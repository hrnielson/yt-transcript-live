import os, io, tempfile, time, datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from supabase import create_client, Client
from openai import OpenAI

# ----------------- Config from Streamlit secrets -----------------
SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

supabase: Client = create_client(SB_URL, SB_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)

# ----------------- Supabase helpers -----------------
def get_or_create_project(name: str, channel_url: str):
    res = supabase.table("projects").select("*").eq("name", name).execute()
    if res.data:
        pid = res.data[0]["id"]
        if res.data[0].get("channel_url") != channel_url:
            supabase.table("projects").update({"channel_url": channel_url}).eq("id", pid).execute()
        return pid
    ins = supabase.table("projects").insert({"name": name, "channel_url": channel_url}).execute()
    return ins.data[0]["id"]

def list_projects():
    res = supabase.table("projects").select("id,name,channel_url,created_at").order("created_at", desc=True).execute()
    return res.data or []

def upsert_video(project_id: str, v: dict):
    row = {
        "id": v["video_id"],
        "project_id": project_id,
        "title": v.get("title"),
        "published_at": v.get("published_at"),
        "url": v["url"],
    }
    supabase.table("videos").upsert(row).execute()

def insert_segments(project_id: str, video_id: str, segments: list[dict]):
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
        })
    CHUNK = 500
    for i in range(0, len(rows), CHUNK):
        supabase.table("segments").insert(rows[i:i+CHUNK]).execute()

def search_segments(project_id: str, q: str, limit: int = 100):
    res = supabase.rpc("segments_search", {"p_project": project_id, "p_query": q, "p_limit": limit}).execute()
    return res.data or []

# ----------------- YouTube: list videos (no API key) -----------------
def list_videos_no_api(channel_or_playlist_url: str):
    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    out = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_or_playlist_url, download=False)
        entries = info.get("entries", []) or []
        for e in entries:
            if e.get("_type") == "url" and "watch" in (e.get("url") or ""):
                vid = e.get("id"); url = e.get("url"); title = e.get("title") or ""
                ts = e.get("timestamp"); published_at = None
                if ts: published_at = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
                if vid:
                    out.append({"video_id": vid, "title": title, "published_at": published_at,
                                "url": url or f"https://www.youtube.com/watch?v={vid}"})
            elif e.get("_type") == "playlist" and e.get("entries"):
                for ve in e["entries"]:
                    vid = ve.get("id"); title = ve.get("title") or ""
                    ts = ve.get("timestamp"); published_at = None
                    if ts: published_at = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
                    if vid:
                        out.append({"video_id": vid, "title": title, "published_at": published_at,
                                    "url": f"https://www.youtube.com/watch?v={vid}"})
            else:
                if e.get("id") and e.get("webpage_url"):
                    vid = e["id"]; title = e.get("title") or ""
                    ts = e.get("timestamp"); published_at = None
                    if ts: published_at = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
                    out.append({"video_id": vid, "title": title, "published_at": published_at, "url": e["webpage_url"]})
    seen = set(); uniq = []
    for v in out:
        if v["video_id"] in seen: continue
        seen.add(v["video_id"]); uniq.append(v)
    return uniq

# ----------------- Captions first -----------------
def fetch_youtube_captions(video_id: str, preferred=("da","en","no","sv")):
    try:
        ts_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for lang in preferred:
            try:
                tr = ts_list.find_transcript([lang])
                return [{"start": s["start"], "duration": s.get("duration",0), "text": s.get("text","")} for s in tr.fetch()]
            except Exception:
                pass
        try:
            tr = ts_list.find_manually_created_transcript([t.language_code for t in ts_list])
            return [{"start": s["start"], "duration": s.get("duration",0), "text": s.get("text","")} for s in tr.fetch()]
        except Exception:
            tr = ts_list.find_generated_transcript([t.language_code for t in ts_list])
            return [{"start": s["start"], "duration": s.get("duration",0), "text": s.get("text","")} for s in tr.fetch()]
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

# ----------------- Robust format probing & download -----------------
def pick_best_available_format(info: dict) -> tuple[str, str]:
    """
    Choose an actually existing format id:
      1) audio-only (prefer m4a/webm, higher abr/asr)
      2) fallback to progressive (audio+video)
    Returns (format_id, ext)
    """
    formats = info.get("formats") or []
    audio = []
    for f in formats:
        if f.get("acodec") and f["acodec"] != "none" and (f.get("vcodec") in (None, "none")):
            abr = f.get("abr") or 0
            asr = f.get("asr") or 0
            ext = f.get("ext") or ""
            pref_ext = 2 if ext == "m4a" else (1 if ext == "webm" else 0)
            audio.append((pref_ext, abr, asr, f))
    if audio:
        audio.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        f = audio[0][3]
        return f["format_id"], f.get("ext") or "m4a"

    prog = []
    for f in formats:
        if (f.get("acodec") and f["acodec"] != "none") and (f.get("vcodec") and f["vcodec"] != "none"):
            height = f.get("height") or 0
            tbr = f.get("tbr") or 0
            prog.append((height, tbr, f))
    if prog:
        prog.sort(key=lambda x: (x[0], x[1]), reverse=True)
        f = prog[0][2]
        return f["format_id"], f.get("ext") or "mp4"

    if formats:
        f = formats[-1]
        return f.get("format_id") or "best", f.get("ext") or "mp4"

    raise RuntimeError("No downloadable formats found")

def write_cookies_if_any(cookies_text: str) -> str | None:
    if not cookies_text or "# Netscape" not in cookies_text:
        return None
    tmp = Path(tempfile.mkdtemp()) / "cookies.txt"
    tmp.write_text(cookies_text, encoding="utf-8")
    return str(tmp)

def download_audio_tmp(video_id: str, cookies_text: str = "") -> Path:
    """
    Download a *real* available format:
      - prefer audio-only; fallback to progressive.
      - return path to downloaded file (.m4a/.webm/.mp4…)
    """
    tmpdir = Path(tempfile.mkdtemp())
    base = tmpdir / f"{video_id}.%(ext)s"
    url = f"https://www.youtube.com/watch?v={video_id}"

    cookiefile = write_cookies_if_any(cookies_text)

    # Probe formats first
    probe_opts = {
        "quiet": True,
        "skip_download": True,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Referer": "https://www.youtube.com/"},
        "geo_bypass": True,
    }
    if cookiefile:
        probe_opts["cookiefile"] = cookiefile
    with yt_dlp.YoutubeDL(probe_opts) as y:
        info = y.extract_info(url, download=False)

    fmt_id, ext = pick_best_available_format(info)

    # Download that exact format id
    ydl_opts = {
        "format": fmt_id,
        "outtmpl": str(base),
        "noplaylist": True,
        "quiet": True,
        "retries": 10,
        "fragment_retries": 10,
        "sleep_requests": 1,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Referer": "https://www.youtube.com/"},
        "geo_bypass": True,
        "concurrent_fragment_downloads": 1,
        "socket_timeout": 30,
    }
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    files = list(tmpdir.glob(f"{video_id}.*"))
    if not files:
        raise RuntimeError("Download produced no files")
    f = max(files, key=lambda p: p.stat().st_size)
    if f.stat().st_size <= 0:
        raise RuntimeError("Downloaded file is empty (0 bytes)")
    return f

# ----------------- ASR via OpenAI Whisper API -----------------
def transcribe_with_openai(file_path: Path, language: str | None):
    with open(file_path, "rb") as f:
        tr = oa_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language if language else None,
            response_format="verbose_json",
        )
    segs = []
    # OpenAI SDK returns dict-like object; access with keys
    segments = getattr(tr, "segments", None) or tr.get("segments") if isinstance(tr, dict) else None
    text = getattr(tr, "text", None) or tr.get("text") if isinstance(tr, dict) else None
    if segments:
        for s in segments:
            start = float(s["start"]); end = float(s["end"])
            segs.append({"start": start, "duration": end - start, "text": s["text"].strip()})
    elif text:
        segs = [{"start": 0.0, "duration": 0.0, "text": text}]
    return segs

# ----------------- UI -----------------
st.set_page_config(page_title="YouTube Quote Finder (Cloud)", layout="wide")
st.title("YouTube Quote Finder — Cloud")

with st.sidebar:
    st.header("Settings")
    captions_first = st.toggle("Use captions first", value=True)
    allow_asr = st.toggle("Allow ASR fallback (OpenAI Whisper API)", value=True)
    language = st.text_input("Language (e.g., 'da' or leave empty for auto)", value="da")
    max_videos = st.number_input("Limit videos (0 = all)", min_value=0, max_value=5000, value=0, step=1)
    st.markdown("**Optional cookies.txt** (Netscape format)")
    cookies_text = st.text_area(
        "Paste cookies (must start with '# Netscape HTTP Cookie File')",
        value="",
        height=120,
        help="Export with a browser extension like 'Get cookies.txt locally' while on youtube.com"
    )
    st.caption("Data is stored in Supabase and shared with your team.")

tab_idx, tab_search = st.tabs(["📦 Index", "🔎 Search"])

def hhmmss(sec: int):
    hh, r = divmod(sec, 3600)
    mm, ss = divmod(r, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

with tab_idx:
    st.subheader("Create or update a project")
    project_name = st.text_input("Project name", placeholder="Client A")
    channel_url = st.text_input("Channel or playlist URL", placeholder="https://www.youtube.com/@CHANNEL")
    if st.button("Start indexing", type="primary", disabled=not (project_name and channel_url)):
        pid = get_or_create_project(project_name, channel_url)
        st.info("Fetching video list…")
        try:
            videos = list_videos_no_api(channel_url)
        except Exception as e:
            st.error(f"Could not list videos: {e}")
            videos = []
        if max_videos and max_videos > 0:
            videos = videos[:max_videos]
        if not videos:
            st.error("No videos found. Is the URL correct and public?")
        else:
            st.success(f"Found {len(videos)} videos.")
            prog = st.progress(0, text="Indexing…")
            status = st.empty()
            total = len(videos); processed = 0
            for v in videos:
                vid = v["video_id"]
                upsert_video(pid, v)
                status.markdown(f"**Processing:** `{v.get('title','(no title)')}` (`{vid}`)")
                try:
                    segs = None
                    if captions_first:
                        segs = fetch_youtube_captions(vid)
                        if segs:
                            insert_segments(pid, vid, segs)
                    if (not segs) and allow_asr:
                        audio = download_audio_tmp(vid, cookies_text)
                        segs = transcribe_with_openai(audio, language.strip() or None)
                        insert_segments(pid, vid, segs)
                except Exception as e:
                    st.warning(f"Skipped {vid}: {e}")
                processed += 1
                prog.progress(int(processed/total*100), text=f"Indexing… ({processed}/{total})")
            status.markdown("**Done.**")

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
        options = {f"{p['name']}": p["id"] for p in prjs}
        sel_name = st.selectbox("Project", list(options.keys()))
        q = st.text_input("Search (FTS term/phrase)", value="hammer")
        limit = st.slider("Max results", 10, 500, 50, step=10)
        if st.button("Search"):
            rows = search_segments(options[sel_name], q.strip(), limit)
            if not rows:
                st.info("No results.")
            else:
                df = pd.DataFrame(rows)
                df["timestamp"] = df["start"].fillna(0).astype(int).map(hhmmss)
                df = df[["title","speaker","content","timestamp","url"]].rename(columns={"content":"quote"})
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "search_results.csv",
                                   "text/csv")
