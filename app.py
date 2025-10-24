# app.py — YouTube Quote Finder (Cloud, OpenAI-only)
# Streamlit + Supabase + YouTube Data API v3 + Captions-first + OpenAI Whisper
# One-file version with: project language, dedup, robust yt-dlp, defensive Supabase, RPC search + fallback, quote extraction.

import re
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from postgrest import APIError
import httpx
import pandas as pd
import streamlit as st
import yt_dlp
from openai import OpenAI
from supabase import Client, create_client
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi)

# ---------- Streamlit page ----------
st.set_page_config(page_title="YouTube Quote Finder", layout="wide")
st.title("🎬 YouTube Quote Finder")

# ---------- Secrets / Clients ----------
SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

supabase: Client = create_client(SB_URL, SB_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)

# ---------- Supabase helpers ----------
def get_or_create_project(name: str, channel_url: str, lang: str):
    """Create-or-update uden at crashe ved RLS/schema-cache issues."""
    try:
        r = supabase.table("projects").select("*").eq("name", name).execute()
    except APIError as e:
        st.error(f"SELECT projects fejlede: {e}")
        raise

    if r.data:
        row = r.data[0]
        pid = row["id"]
        update = {}
        if row.get("channel_url") != channel_url:
            update["channel_url"] = channel_url
        if "lang" in row and (row.get("lang") or "auto") != (lang or "auto"):
            update["lang"] = lang or "auto"
        if update:
            try:
                supabase.table("projects").update(update).eq("id", pid).execute()
            except APIError as e:
                st.warning(f"UPDATE projects blokeret: {e}")
        return pid

    # INSERT
    payload = {"name": name, "channel_url": channel_url, "lang": lang or "auto"}
    try:
        ins = supabase.table("projects").insert(payload).execute()
        return ins.data[0]["id"]
    except APIError as e:
        # Stale schema cache: prøv uden lang
        payload.pop("lang", None)
        ins = supabase.table("projects").insert(payload).execute()
        return ins.data[0]["id"]

def list_projects():
    r = supabase.table("projects").select("*").order("name", desc=False).execute()
    return r.data or []

def upsert_video(project_id: str, v: dict):
    """Idempotent upsert of a video row by primary key id."""
    row = {
        "id": v["video_id"],
        "project_id": project_id,
        "title": v.get("title"),
        "published_at": v.get("published_at"),
        "url": v["url"],
    }
    supabase.table("videos").upsert(row, on_conflict="id").execute()

def is_already_indexed(project_id: str, video_id: str) -> bool:
    """True hvis videoen allerede har mindst ét segment for projektet eller videos.indexed_at er sat."""
    r = supabase.table("segments").select("video_id").eq("project_id", project_id).eq("video_id", video_id).limit(1).execute()
    if r.data:
        return True
    r2 = supabase.table("videos").select("indexed_at").eq("id", video_id).limit(1).execute()
    return bool(r2.data and r2.data[0].get("indexed_at"))

def insert_segments(project_id: str, video_id: str, segments: list[dict], lang: str | None = None):
    """Batch upsert of segments."""
    if not segments:
        return
    rows = []
    for s in segments:
        txt = (s.get("text") or s.get("content") or "").strip()
        if not txt:
            continue
        rows.append({
            "project_id": project_id,
            "video_id": video_id,
            "start": float(s.get("start", 0)),
            "duration": float(s.get("duration", 0)),
            "speaker": s.get("speaker") or None,
            "content": txt,
            "lang": (s.get("lang") or lang or "auto"),
        })
    for i in range(0, len(rows), 500):
        supabase.table("segments").upsert(rows[i:i + 500], on_conflict="project_id,video_id,start").execute()

# ---------- Quote extraction ----------
QUOTE_MODEL = "gpt-4o-mini"
MAX_WINDOW_CHARS = 8000

EXTRACTION_SYS = (
    "Du er en præcis citatudtrækker og omformulerer. "
    "Opgave: Find korte, citerbare udsagn fra teksten, der deler RÅD/ANBEFALINGER/ERFARINGER om materialer, værktøj, metoder eller valg. "
    "Regler: "
    "1) Du må gerne PARAFRASERE for bedre læsbarhed til artikler/produkt- og kategoritekster, men betydningen må ikke ændres. "
    "2) Medtag også ORIGINALEN (ordret uddrag) som 'original'. "
    "3) Ingen nye fakta; ingen overdrivelser; ingen markedsføring. "
    "4) Brug samme sprog som input. "
    "Output: STRICT JSON: "
    "{\"quotes\":[{\"quote\":\"<parafraseret>\",\"original\":\"<ordret uddrag>\",\"topic\":\"...\",\"tags\":[\"...\"],\"confidence\":0.0}]}"
)

def get_video_segments(project_id: str, video_id: str):
    res = supabase.table("segments").select("start,duration,content,lang")\
        .eq("project_id", project_id).eq("video_id", video_id).order("start").execute()
    return res.data or []

def get_project_display_name(project_id: str) -> str:
    try:
        r = supabase.table("projects").select("name").eq("id", project_id).limit(1).execute()
        if r.data:
            return r.data[0].get("name") or "Kilden"
    except Exception:
        pass
    return "Kilden"

def _windows_from_segments(segments, max_chars=MAX_WINDOW_CHARS):
    cur, cur_chars = [], 0
    for s in segments:
        t = (s.get("content") or "").strip()
        if not t:
            continue
        if cur_chars + len(t) + 1 > max_chars and cur:
            yield cur
            cur, cur_chars = [], 0
        cur.append(s)
        cur_chars += len(t) + 1
    if cur:
        yield cur

def _nearest_timestamp_for_quote(qtext: str, window_segments):
    q = (qtext or "").strip()
    if not q:
        return None
    key = " ".join(q.split()[:4]).lower()
    for seg in window_segments:
        c = (seg.get("content") or "").lower()
        if key and key in c:
            return float(seg.get("start") or 0.0)
    return float(window_segments[0].get("start") or 0.0) if window_segments else None

def extract_quotes_from_video(project_id: str, video_id: str, lang_hint: str, source: str = "asr", max_quotes_per_window: int = 6):
    segs = get_video_segments(project_id, video_id)
    if not segs:
        return 0
    
    attribution = get_project_display_name(project_id)
    total_inserted = 0

    def backfill_quotes_for_project(project_id: str, lang_hint: str):
    """Lav citater for alle videoer i projektet, hvor der ikke findes quotes endnu."""
    # Hent alle video-ids i projektet
    vids = supabase.table("videos").select("id").eq("project_id", project_id).execute().data or []
    processed, created = 0, 0
    for v in vids:
        vid = v["id"]
        # Findes der allerede mindst ét quote for denne video?
        existing = supabase.table("quotes").select("id", count="exact")\
                    .eq("project_id", project_id).eq("video_id", vid).limit(1).execute()
        has_quotes = bool(getattr(existing, "count", 0))
        if has_quotes:
            continue

        # Er der segments at arbejde med?
        segs_exist = supabase.table("segments").select("video_id", count="exact")\
                        .eq("project_id", project_id).eq("video_id", vid).limit(1).execute()
        if not bool(getattr(segs_exist, "count", 0)):
            continue

        # Uddrag citater ud fra segments
        try:
            new_cnt = extract_quotes_from_video(project_id, vid, lang_hint, source="backfill")
            created += int(new_cnt or 0)
            processed += 1
        except Exception as e:
            st.caption(f"Backfill for {vid} skipped: {e}")
    return processed, created

    
    for win in _windows_from_segments(segs):
        text = "\n".join((s.get("content") or "").strip() for s in win if s.get("content"))
        if not text.strip():
            continue

        user_prompt = (
            f"Sprog: {lang_hint or 'auto'}.\n"
            f"Maks {max_quotes_per_window} citater.\n"
            "Tekst:\n---\n" + text + "\n---\n\n"
            "Returnér KUN JSON som beskrevet i systemprompten."
        )

        try:
            resp = oa_client.chat.completions.create(
                model=QUOTE_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYS},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            js = json.loads(resp.choices[0].message.content or "{}")
            items = js.get("quotes") or []
        except Exception as e:
            st.caption(f"Quote-extraction LLM error: {e}")
            continue

        for it in items:
            paraphrase = (it.get("quote") or "").strip()
            original   = (it.get("original") or "").strip()
            if not paraphrase:
                continue
            start = _nearest_timestamp_for_quote(original or paraphrase, win)
            end = None if start is None else start + 6.0

            row = {
                "project_id": project_id,
                "video_id": video_id,
                "start": start,
                "end": end,
                "quote": paraphrase,
                "original": original or None,
                "topic": (it.get("topic") or None),
                "tags": it.get("tags") or [],
                "lang": lang_hint or None,
                "source": source,
                "confidence": float(it.get("confidence") or 0.6),
                "attribution": attribution,
                "paraphrased": True,
            }
            try:
                supabase.table("quotes").insert(row).execute()
                total_inserted += 1
            except Exception:
                # sandsynligvis duplikat pga. unique index
                pass

    return total_inserted

# ---------- HTTP helper ----------
def _http_get(url: str, params: dict, timeout=60):
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            safe_params = {k: v for k, v in params.items() if k != "key"}
            raise RuntimeError(f"HTTP {e.response.status_code} on {url} with params {safe_params}") from e
        return r.json()

# ---------- YouTube Data API ----------
def _extract_bits(inp: str):
    s = (inp or "").strip()
    if s.startswith("@"):
        return s, None, None, None
    m = re.search(r"youtube\.com/(channel/([^/?#]+))", s, re.I)
    if m:
        return None, m.group(2), None, None
    m = re.search(r"youtube\.com/@([^/?#]+)", s, re.I)
    if m:
        return "@" + m.group(1), None, None, None
    m = re.search(r"youtube\.com/user/([^/?#]+)", s, re.I)
    if m:
        return None, None, m.group(1), None
    m = re.search(r"youtube\.com/c/([^/?#]+)", s, re.I)
    if m:
        return None, None, None, m.group(1)
    return None, None, None, None

def _resolve_channel_id(handle_or_url: str) -> str:
    base = "https://www.googleapis.com/youtube/v3"
    handle, ch_id, username, custom = _extract_bits(handle_or_url)
    if ch_id:
        return ch_id
    if handle:
        data = _http_get(f"{base}/channels", {"part": "id", "forHandle": handle, "key": YOUTUBE_API_KEY})
        items = data.get("items", [])
        if items:
            return items[0]["id"]
    if username:
        data = _http_get(f"{base}/channels", {"part": "id", "forUsername": username, "key": YOUTUBE_API_KEY})
        items = data.get("items", [])
        if items:
            return items[0]["id"]
    q = (handle_or_url or "").replace("https://www.youtube.com/", " ").strip()
    data = _http_get(f"{base}/search", {"part": "snippet", "type": "channel", "q": q or "", "maxResults": 1, "key": YOUTUBE_API_KEY})
    items = data.get("items", [])
    if items:
        return items[0]["snippet"]["channelId"]
    raise RuntimeError("Channel not found via YouTube API.")

def list_videos_by_channel_id(channel_id: str):
    base = "https://www.googleapis.com/youtube/v3"
    info = _http_get(f"{base}/channels", {"part": "contentDetails", "id": channel_id, "key": YOUTUBE_API_KEY})
    items = info.get("items", [])
    if not items:
        raise RuntimeError("Channel not found (contentDetails) for provided channelId.")
    uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    out, page = [], None
    while True:
        data = _http_get(
            f"{base}/playlistItems",
            {"part": "snippet,contentDetails", "playlistId": uploads_pl, "maxResults": 50, "pageToken": page, "key": YOUTUBE_API_KEY},
        )
        for it in data.get("items", []):
            vid = it["contentDetails"]["videoId"]
            title = it["snippet"]["title"]
            published_at = it["contentDetails"].get("videoPublishedAt")
            url = f"https://www.youtube.com/watch?v={vid}"
            out.append({"video_id": vid, "title": title, "published_at": published_at, "url": url})
        page = data.get("nextPageToken")
        if not page:
            break
    return out

# ---------- Captions ----------
def fetch_youtube_captions(video_id: str, preferred=("da", "en", "no", "sv")):
    """Try preferred languages, then manual, then generated, then translate to en."""
    try:
        ts_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Preferred languages
        for lang in preferred:
            try:
                tr = ts_list.find_transcript([lang])
                return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr.fetch()]
            except Exception:
                pass

        # Manual > generated
        try:
            all_langs = [t.language_code for t in ts_list]
            tr = ts_list.find_manually_created_transcript(all_langs)
            return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr.fetch()]
        except Exception:
            pass
        try:
            all_langs = [t.language_code for t in ts_list]
            tr = ts_list.find_generated_transcript(all_langs)
            return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr.fetch()]
        except Exception:
            pass

        # Translate to English last
        try:
            for t in ts_list:
                try:
                    tr_en = t.translate("en")
                    return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr_en.fetch()]
                except Exception:
                    continue
        except Exception:
            return None

    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

# ---------- Language helper ----------
def preferred_langs_for(project_lang: str) -> tuple[str, ...]:
    pl = (project_lang or "auto").lower()
    if pl == "da":
        return ("da", "no", "sv", "en")
    if pl == "sv":
        return ("sv", "no", "da", "en")
    if pl == "en":
        return ("en", "da", "sv")
    return ("da", "en", "sv")  # auto fallback

# ---------- Downloader ----------
def _write_cookies_if_any(cookies_text: str) -> str | None:
    if not cookies_text:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="cookies_"))
    cookiefile = tmpdir / "cookies.txt"
    cookiefile.write_text(cookies_text, encoding="utf-8")
    if "# Netscape" not in cookies_text:
        st.caption("⚠️ cookies.txt without Netscape header – proceeding anyway.")
    return str(cookiefile)

def _try_download(url: str, outtmpl: str, fmt: str, cookiefile: str | None, merge_to: str = "m4a") -> Path | None:
    ytdl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 4,
        "fragment_retries": 4,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "geo_bypass": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": merge_to, "preferredquality": "0"}],
        "merge_output_format": merge_to,
        "concurrent_fragments": 3,
        "overwrites": True,
        "force_ipv4": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web", "web_safari"]}},
    }
    if cookiefile:
        ytdl_opts["cookiefile"] = cookiefile
    try:
        with yt_dlp.YoutubeDL(ytdl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.caption(f"yt-dlp error ({fmt}): {e}")
        return None

    outdir = Path(outtmpl).parent
    stem = Path(outtmpl).name.split(".")[0]
    matches = list(outdir.glob(f"{stem}.*"))
    if not matches:
        return None
    best = max(matches, key=lambda p: p.stat().st_size)
    return best if best.exists() and best.stat().st_size > 0 else None

def download_audio_tmp(video_id: str, cookies_text: str = "") -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix=f"yqf_{video_id}_"))
    outtmpl = str(tmpdir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookiefile = _write_cookies_if_any(cookies_text)

    fmt_list = ["140", "251", "bestaudio[ext=m4a]/bestaudio", "bestaudio/best", "best"]

    try:
        for fmt in fmt_list:
            p = _try_download(url, outtmpl, fmt, cookiefile)
            if p:
                return p
        raise RuntimeError("No available format could be downloaded (all strategies failed)")
    except Exception:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if cookiefile:
                shutil.rmtree(Path(cookiefile).parent, ignore_errors=True)
        except Exception:
            pass
        raise

# ---------- OpenAI Whisper (fallback chain) ----------
def transcribe_with_openai(file_path: Path, language: str | None):
    model_candidates = ["gpt-4o-mini-transcribe", "whisper-1"]
    last_err = None
    for mdl in model_candidates:
        try:
            with open(file_path, "rb") as f:
                resp = oa_client.audio.transcriptions.create(
                    model=mdl,
                    file=f,
                    language=language if language else None,
                    response_format="verbose_json",
                )
            segs = []
            segments = getattr(resp, "segments", None)
            if segments:
                for s in segments:
                    start = float(s.get("start", 0))
                    end = float(s.get("end", start))
                    segs.append({
                        "start": start,
                        "duration": max(0.0, end - start),
                        "text": (s.get("text") or "").strip(),
                    })
            elif getattr(resp, "text", None):
                segs = [{"start": 0, "duration": 0, "text": resp.text.strip()}]
            return segs
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"OpenAI transcription failed: {last_err}")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    captions_first = st.toggle("Use captions first", value=True)
    project_lang = st.text_input("Channel language (da/en/sv or 'auto')", value="da").strip().lower() or "auto"
    max_videos = st.number_input("Limit videos (0 = all)", 0, 5000, 0)
    channel_id_override = st.text_input(
        "Channel ID override (optional, starts with 'UC')",
        help="Paste the channelId (UCxxxxxxxxxxxxxxxxxxxxxx) to skip handle/URL detection.",
    )
    cookies_text = st.text_area(
        "Optional cookies.txt (Netscape format)",
        height=120,
        help="Export with 'Get cookies.txt locally' while on youtube.com.",
    )
st.sidebar.caption(f"yt-dlp: {yt_dlp.version.__version__}")

if "pid" not in st.session_state:
    st.session_state.pid = None
if "resolved_channel_id" not in st.session_state:
    st.session_state.resolved_channel_id = None

# ---------- Tabs ----------
tab_idx, tab_search = st.tabs(["📦 Index Videos", "💬 Find Quotes"])

def hhmmss(sec: float):
    s = int(round(sec or 0))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- Index tab ---
with tab_idx:
    try:
        st.subheader("Create or update a project")
        project_name = st.text_input("Project name", placeholder="Client A", key="proj_name")
        channel_url = st.text_input("Channel handle or URL", placeholder="@brand or https://www.youtube.com/@brand", key="chan_url")

        if st.button("Start indexing", type="primary", disabled=not (project_name and channel_url), key="start_idx"):
            pid = get_or_create_project(project_name, channel_url, project_lang)
            st.session_state.pid = pid
            st.info("Fetching video list via YouTube Data API…")

            try:
                chan_id = (channel_id_override or "").strip()
                if chan_id:
                    st.success(f"Using provided Channel ID: {chan_id}")
                    videos = list_videos_by_channel_id(chan_id)
                    st.session_state.resolved_channel_id = chan_id
                else:
                    cid = _resolve_channel_id(channel_url)
                    st.info(f"Resolved Channel ID: {cid}")
                    videos = list_videos_by_channel_id(cid)
                    st.session_state.resolved_channel_id = cid
            except Exception as e:
                st.error(f"Could not list videos: {e}")
                videos = []

            if max_videos > 0:
                videos = videos[: max_videos]

            if not videos:
                st.error("No videos found. Check Channel ID, handle, or visibility.")
            else:
                st.success(f"Found {len(videos)} videos.")
                prog = st.progress(0, text="Indexing…")
                total = len(videos)
                done = 0

                for v in videos:
                    vid = v["video_id"]
                    upsert_video(pid, v)

                    # Dedup
                    if is_already_indexed(pid, vid):
                        st.write(f"⏭️ Skipper allerede indekseret: {v['title']}")
                        done += 1
                        prog.progress(int(done / total * 100), text=f"Indexing… {done}/{total}")
                        continue

                    st.write(f"Processing {v['title']}")
                    segs = None
                    audio_path = None
                    try:
                        # 1) Captions-first
                        if captions_first:
                            segs = fetch_youtube_captions(vid, preferred=preferred_langs_for(project_lang))
                            if segs:
                                insert_segments(pid, vid, segs, lang=project_lang)

                        # 2) Audio download + ASR
                        if not segs:
                            audio_path = download_audio_tmp(vid, cookies_text)
                            forced_lang = None if project_lang == "auto" else project_lang
                            segs = transcribe_with_openai(audio_path, forced_lang)
                            insert_segments(pid, vid, segs, lang=project_lang)

                        if not segs:
                            st.warning(f"No segments for {vid} (captions+ASR failed).")
                        else:
                            # markér som indekseret
                            supabase.table("videos").update({
                                "indexed_at": datetime.now(timezone.utc).isoformat()
                            }).eq("id", vid).execute()

                            # ➕ citatudtræk
                            try:
                                inserted = extract_quotes_from_video(pid, vid, project_lang, source=("captions" if captions_first else "asr"))
                                if inserted:
                                    st.write(f"➕ Extracted {inserted} publish-ready quotes")
                            except Exception as e:
                                st.caption(f"Quote extraction skipped: {e}")

                    except Exception as e:
                        st.warning(f"Skipped {vid}: {e}")
                    finally:
                        try:
                            if audio_path and isinstance(audio_path, Path):
                                shutil.rmtree(audio_path.parent, ignore_errors=True)
                        except Exception:
                            pass

                    done += 1
                    prog.progress(int(done / total * 100), text=f"Indexing… {done}/{total}")

                st.success("✅ Done indexing.")

        st.divider()
        st.subheader("Existing projects")
        prjs = list_projects()
        if prjs:
            st.dataframe(pd.DataFrame(prjs), use_container_width=True)
        else:
            st.info("No projects yet.")
    except Exception as e:
        st.exception(e)

with tab_idx:
    # ... dit eksisterende UI ...
    st.subheader("Backfill quotes")
    prjs = list_projects()
    if prjs:
        options_bf = {p["name"]: p["id"] for p in prjs}
        sel_bf = st.selectbox("Project to backfill", list(options_bf.keys()), key="bf_proj")
        if st.button("Generate missing quotes for this project", key="bf_btn"):
            proc, made = backfill_quotes_for_project(options_bf[sel_bf], project_lang)
            st.success(f"Backfill done: processed {proc} videos, created {made} new quotes.")
    # ...


# --- Find Quotes tab ---
with tab_search:
    search_mode = st.radio("Search in:", ["Segments", "Quotes"], horizontal=True)

    try:
        st.subheader("Search")
        prjs = list_projects()
        if not prjs:
            st.info("Create a project first.")
        else:
            options = {p["name"]: p["id"] for p in prjs}
            sel = st.selectbox("Project", list(options.keys()), key="search_proj")
            q = st.text_input("Search term", value="hammer", key="search_term")
            limit = st.slider("Max results", 10, 500, 50, 10, key="search_limit")

            rows = []
            if st.button("Search", key="search_btn"):
                if search_mode == "Segments":
                    # --- Segment-søgning (RPC + fallback) ---
                    try:
                        res = supabase.rpc("segments_search_by_project", {
                            "p_limit": int(limit),
                            "p_project": options[sel],
                            "p_query": q.strip(),
                        }).execute()
                        rows = res.data or []
                    except APIError as e:
                        st.warning(
                            "RPC failed — falling back to simple LIKE search (no stemming/synonyms).\n"
                            f"code={getattr(e,'code',None)} message={getattr(e,'message',None)}"
                        )
                        seg_rows = supabase.table("segments")\
                            .select("content,start,video_id,project_id")\
                            .eq("project_id", options[sel])\
                            .ilike("content", f"%{q.strip()}%")\
                            .limit(int(limit))\
                            .execute().data or []

                        vid_ids = sorted({r["video_id"] for r in seg_rows})
                        videos_map = {}
                        if vid_ids:
                            vids = supabase.table("videos")\
                                .select("id,title,url")\
                                .in_("id", vid_ids)\
                                .execute().data or []
                            videos_map = {v["id"]: {"title": v.get("title"), "url": v.get("url")} for v in vids}

                        rows = [{
                            "title": videos_map.get(r["video_id"], {}).get("title"),
                            "url": videos_map.get(r["video_id"], {}).get("url"),
                            "content": r.get("content"),
                            "start": r.get("start"),
                        } for r in seg_rows]

                    # visning for segments
                    if not rows:
                        st.info("No results.")
                    else:
                        df = pd.DataFrame(rows)
                        if "start" in df:
                            df["timestamp"] = df["start"].fillna(0).map(hhmmss)
                        cols = [c for c in ("title", "speaker", "content", "timestamp", "url") if c in df.columns]
                        if not cols:
                            cols = [c for c in ("content", "timestamp") if c in df.columns]
                        df = df[cols].rename(columns={"content": "quote"})
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            "search_results.csv",
                            "text/csv",
                        )

                else:
                    # --- Quote-søgning ---
                    st.caption("💬 Searching in AI-extracted quotes…")
                    qb = supabase.table("quotes")\
                        .select("quote,original,topic,tags,lang,start,video_id,project_id,attribution")\
                        .eq("project_id", options[sel])\
                        .order("created_at", desc=True)\
                        .limit(int(limit))

                    rows = qb.execute().data or []
                    if q.strip():
                        ql = q.lower()
                        rows = [r for r in rows if
                                (r.get("quote") and ql in r["quote"].lower()) or
                                (r.get("original") and ql in (r["original"] or "").lower()) or
                                (r.get("topic") and ql in (r["topic"] or "").lower()) or
                                any(ql in (t or "").lower() for t in (r.get("tags") or []))
                               ][:int(limit)]

                    vid_ids = sorted({r["video_id"] for r in rows})
                    videos_map = {}
                    if vid_ids:
                        vids = supabase.table("videos").select("id,title,url").in_("id", vid_ids).execute().data or []
                        videos_map = {v["id"]: {"title": v.get("title"), "url": v.get("url")} for v in vids}

                    def ts_link(url, start):
                        if not url: return None
                        try:
                            s = int(round(float(start or 0)))
                        except Exception:
                            s = 0
                        sep = "&" if "?" in url else "?"
                        return f"{url}{sep}t={s}s"

                    df = pd.DataFrame([{
                        "title": videos_map.get(r["video_id"], {}).get("title"),
                        "url": ts_link(videos_map.get(r["video_id"], {}).get("url"), r.get("start")),
                        "quote": r.get("quote"),
                        "original": r.get("original"),
                        "attribution": r.get("attribution"),
                        "topic": r.get("topic"),
                        "tags": ", ".join(r.get("tags") or []),
                        "lang": r.get("lang"),
                    } for r in rows])

                    if df.empty:
                        st.info("No quotes yet.")
                    else:
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            "quotes.csv",
                            "text/csv",
                        )

    except Exception as e:
        st.exception(e)

# ---------- Diagnostics (optional expander) ----------
with st.expander("Diagnostics"):
    def diagnose_video(video_id: str, cookies_text: str = "") -> dict:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cookiefile = _write_cookies_if_any(cookies_text)
        opts = {"quiet": True, "skip_download": True, "noplaylist": True,
                "http_headers": {"User-Agent": "Mozilla/5.0"}, "force_ipv4": True}
        if cookiefile:
            opts["cookiefile"] = cookiefile
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        ps = info.get("playability_status") or {}
        return {
            "ok": True,
            "formats": len(info.get("formats") or []),
            "age_limit": info.get("age_limit"),
            "availability": info.get("availability"),
            "playable_in_embed": info.get("playable_in_embed"),
            "status": ps.get("status"),
            "reason": ps.get("reason"),
            "uploader": info.get("uploader"),
            "title": info.get("title"),
        }

    vid = st.text_input("Video ID", value="SDm_sdmz-FU")
    if st.button("Run diagnostics"):
        st.json(diagnose_video(vid, ""))
