# app.py — YouTube Quote Finder (Cloud, OpenAI-only)
# Streamlit + Supabase + YouTube Data API v3 + Captions-first + OpenAI Whisper
# 2025-ready edition with robust yt-dlp fallbacks, Whisper fallback, idempotent DB upserts, temp cleanup
# One-file version with: project language, dedup (never index same video twice), and stable yt-dlp options.

import re
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
        # VIGTIGT: vælg * i stedet for at nævne kolonner (lang kan mangle i cache)
        r = supabase.table("projects").select("*").eq("name", name).execute()
    except APIError as e:
        st.error(
            "SELECT projects fejlede.\n"
            f"code={getattr(e,'code',None)} message={getattr(e,'message',None)} "
            f"details={getattr(e,'details',None)} hint={getattr(e,'hint',None)}"
        )
        raise

    if r.data:
        row = r.data[0]
        pid = row["id"]
        update = {}
        if row.get("channel_url") != channel_url:
            update["channel_url"] = channel_url
        # Opdater kun lang hvis kolonnen faktisk findes i denne række
        if "lang" in row and (row.get("lang") or "auto") != (lang or "auto"):
            update["lang"] = lang or "auto"

        if update:
            try:
                supabase.table("projects").update(update).eq("id", pid).execute()
            except APIError as e:
                st.warning(
                    "UPDATE projects blokeret – fortsætter uden at stoppe appen.\n"
                    f"code={getattr(e,'code',None)} message={getattr(e,'message',None)}"
                )
        return pid

    # INSERT – prøv med lang, og fald tilbage uden hvis schema-cache ikke kender den
    payload = {"name": name, "channel_url": channel_url, "lang": lang or "auto"}
    try:
        ins = supabase.table("projects").insert(payload).execute()
        return ins.data[0]["id"]
    except APIError as e:
        msg = (getattr(e, "message", "") or "").lower()
        if "lang" in payload and ("lang" in msg or getattr(e, "code", "") == "PGRST204"):
            st.info("Schema-cache kender ikke 'lang' endnu. Prøver INSERT uden 'lang'.")
            payload.pop("lang", None)
            ins = supabase.table("projects").insert(payload).execute()
            return ins.data[0]["id"]
        raise

    # Opret nyt projekt
    payload = {"name": name, "channel_url": channel_url, "lang": lang or "auto"}
    try:
        ins = supabase.table("projects").insert(payload).execute()
        return ins.data[0]["id"]
    except APIError as e:
        # Hvis schema-cache ikke kender 'lang', så prøv igen uden 'lang'
        msg = (getattr(e, "message", "") or "").lower()
        if "lang" in payload and ("lang" in msg or getattr(e, "code", "") == "PGRST204"):
            st.info("Schema-cache kender ikke kolonnen 'lang' endnu ved INSERT. Prøver igen uden 'lang'.")
            payload.pop("lang", None)
            ins = supabase.table("projects").insert(payload).execute()
            return ins.data[0]["id"]
        # Andre fejl – bobler op så vi kan se dem i UI
        raise
        return pid

    # Insert hvis ikke findes
    payload = {"name": name, "channel_url": channel_url, "lang": lang or "auto"}
    try:
        ins = supabase.table("projects").insert(payload).execute()
    except APIError as e:
        # Samme fallback ved stale cache på INSERT
        msg = (getattr(e, "message", "") or "").lower()
        if "lang" in payload and ("lang" in msg or getattr(e, "code", "") == "PGRST204"):
            st.info("Schema-cache kender ikke kolonnen 'lang' endnu ved INSERT. Prøver uden 'lang'.")
            payload.pop("lang", None)
            ins = supabase.table("projects").insert(payload).execute()
        else:
            raise
    return ins.data[0]["id"]



def list_projects():
    try:
        r = supabase.table("projects").select("*").order("name", desc=False).execute()
        return r.data or []
    except APIError as e:
        st.error(
            "Supabase APIError on SELECT projects\n"
            f"message: {getattr(e, 'message', None)}\n"
            f"details: {getattr(e, 'details', None)}\n"
            f"hint: {getattr(e, 'hint', None)}\n"
            f"code: {getattr(e, 'code', None)}"
        )
        raise

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
    """Return True if the video already has at least one segment for this project or videos.indexed_at is set."""
    r = supabase.table("segments").select("video_id").eq("project_id", project_id).eq("video_id", video_id).limit(1).execute()
    if r.data:
        return True
    r2 = supabase.table("videos").select("indexed_at").eq("id", video_id).limit(1).execute()
    return bool(r2.data and r2.data[0].get("indexed_at"))

def insert_segments(project_id: str, video_id: str, segments: list[dict], lang: str | None = None):
    """Batch upsert of segments. Requires a unique index on (project_id, video_id, start) in DB."""
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

# ---------- HTTP helper ----------

def _http_get(url: str, params: dict, timeout=60):
    """GET JSON with context-rich error messages (key redacted)."""
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
    if s.startswith("@"):  # handle
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
    """Resolve channelId from @handle, channel URL, username, or a search fallback."""
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
    """Return a list of uploads with id/title/published_at/url for a channel."""
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
    """Return a tuple of preferred caption languages based on the project's language."""
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
    """Write a cookies.txt file into a temp dir and return its path; accept both Netscape and raw outputs."""
    if not cookies_text:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="cookies_"))
    cookiefile = tmpdir / "cookies.txt"
    cookiefile.write_text(cookies_text, encoding="utf-8")
    if "# Netscape" not in cookies_text:
        st.caption("⚠️ cookies.txt without Netscape header – proceeding anyway.")
    return str(cookiefile)

def _try_download(url: str, outtmpl: str, fmt: str, cookiefile: str | None, merge_to: str = "m4a") -> Path | None:
    """Attempt a single yt-dlp download with FFmpeg postprocessing to a consistent container."""
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
    """Download audio to a temp directory with robust format fallbacks; returns an audio file (m4a) or raises."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f"yqf_{video_id}_"))
    outtmpl = str(tmpdir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookiefile = _write_cookies_if_any(cookies_text)

    # Aggressive yet common format fallbacks: known IDs first, then generic bestaudio.
    fmt_list = [
        "140",                         # m4a 128kbps (very common)
        "251",                         # webm/opus ~160kbps
        "bestaudio[ext=m4a]/bestaudio",
        "bestaudio/best",
        "best",
    ]

    try:
        for fmt in fmt_list:
            p = _try_download(url, outtmpl, fmt, cookiefile)
            if p:
                return p
        raise RuntimeError("No available format could be downloaded (all strategies failed)")
    except Exception:
        # Cleanup temp dirs if we fail entirely
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if cookiefile:
                shutil.rmtree(Path(cookiefile).parent, ignore_errors=True)
        except Exception:
            pass
        raise

# ---------- OpenAI Whisper (fallback chain) ----------

def transcribe_with_openai(file_path: Path, language: str | None):
    """Try newer transcription model then fallback to whisper-1; return list of segment dicts."""
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

# ---------- UI ----------

st.set_page_config(page_title="YouTube Quote Finder", layout="wide")
st.title("🎬 YouTube Quote Finder")

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

tab_idx, tab_search = st.tabs(["📦 Index", "🔎 Search"])

def hhmmss(sec: float):
    s = int(round(sec or 0))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

with tab_idx:
    st.subheader("Create or update a project")
    project_name = st.text_input("Project name", placeholder="Client A")
    channel_url = st.text_input("Channel handle or URL", placeholder="@brand or https://www.youtube.com/@brand")

    if st.button("Start indexing", type="primary", disabled=not (project_name and channel_url)):
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

                # ⛔ Dedup: skip if already indexed for this project
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
                        # ✅ Mark video as indexed
                        supabase.table("videos").update({
                            "indexed_at": datetime.now(timezone.utc).isoformat()
                        }).eq("id", vid).execute()

                except Exception as e:
                    st.warning(f"Skipped {vid}: {e}")

                finally:
                    # Cleanup per-video temp dir
                    try:
                        if audio_path and isinstance(audio_path, Path):
                            tmp_parent = audio_path.parent
                            shutil.rmtree(tmp_parent, ignore_errors=True)
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

with tab_search:
    st.subheader("Search quotes")
    prjs = list_projects()
    if not prjs:
        st.info("Create a project first.")
    else:
        options = {p["name"]: p["id"] for p in prjs}
        sel = st.selectbox("Project", list(options.keys()))
        q = st.text_input("Search term", value="hammer")
        limit = st.slider("Max results", 10, 500, 50, 10)
        if st.button("Search"):
try:
    res = supabase.rpc("segments_search_by_project", {
        "p_project": options[sel],
        "p_query": q.strip(),
        "p_limit": int(limit),
    }).execute()
    rows = res.data or []
except APIError:
    st.warning("RPC failed — falling back to simple LIKE search (no stemming/synonyms).")
    # Simple fallback: case-insensitive 'content ILIKE %q%'
    rows = supabase.table("segments")\
        .select("content,start,video_id,project_id")\
        .eq("project_id", options[sel])\
        .ilike("content", f"%{q.strip()}%")\
        .limit(int(limit))\
        .execute().data or []
    # join på videos for title/url hvis du vil pifte det op:
    # (kan laves med yderligere kald)


            if not rows:
                st.info("No results.")
            else:
                df = pd.DataFrame(rows)
                if "start" in df:
                    df["timestamp"] = df["start"].fillna(0).map(hhmmss)
                cols = [c for c in ("title", "speaker", "content", "timestamp", "url") if c in df.columns]
                df = df[cols].rename(columns={"content": "quote"})
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "search_results.csv",
                    "text/csv",
                )

# ---------- Diagnostics ----------

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

with st.expander("Diagnostics"):
    vid = st.text_input("Video ID", value="SDm_sdmz-FU")
    if st.button("Run diagnostics"):
        st.json(diagnose_video(vid, cookies_text))

# ---------- Notes ----------
# DB requirements (run in Supabase SQL editor, once):
# 1) Ensure projects.lang and segments.lang exist; add videos.indexed_at
#    alter table public.projects add column if not exists lang text default 'auto';
#    alter table public.segments add column if not exists lang text;
#    alter table public.videos   add column if not exists indexed_at timestamptz;
# 2) Unique + performance:
#    create unique index if not exists segments_unique on public.segments(project_id, video_id, start);
#    create index if not exists segments_proj_vid_idx on public.segments(project_id, video_id);
# 3) FTS indexes (optional but recommended) + RPC 'segments_search_by_project' as previously provided.
