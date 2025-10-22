# app.py — YouTube Quote Finder (Cloud, OpenAI-only)
# Streamlit Cloud + Supabase + Captions-first + OpenAI Whisper ASR (file upload)
# Robust yt-dlp download (audio -> progressive -> HLS) + cookies support.

import os, time, datetime, tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import httpx
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from supabase import create_client, Client
from openai import OpenAI

# ----------------- Secrets / Clients -----------------
SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

supabase: Client = create_client(SB_URL, SB_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)

# ----------------- Supabase helpers -----------------
def get_or_create_project(name: str, channel_url: str):
    r = supabase.table("projects").select("*").eq("name", name).execute()
    if r.data:
        pid = r.data[0]["id"]
        if r.data[0].get("channel_url") != channel_url:
            supabase.table("projects").update({"channel_url": channel_url}).eq("id", pid).execute()
        return pid
    ins = supabase.table("projects").insert({"name": name, "channel_url": channel_url}).execute()
    return ins.data[0]["id"]

def list_projects():
    r = supabase.table("projects").select("id,name,channel_url,created_at").order("created_at", desc=True).execute()
    return r.data or []

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
    r = supabase.rpc("segments_search", {"p_project": project_id, "p_query": q, "p_limit": limit}).execute()
    return r.data or []

# ----------------- YouTube Data API v3 (stable listing) -----------------
def _http_get(url: str, params: dict, timeout=60):
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()

def _resolve_channel_id(handle_or_url: str) -> str:
    if "/channel/" in handle_or_url:
        return handle_or_url.split("/channel/")[1].split("/")[0]
    base = "https://www.googleapis.com/youtube/v3"
    data = _http_get(f"{base}/search", {
        "part": "snippet",
        "q": handle_or_url.replace("https://www.youtube.com/", "").replace("@",""),
        "type": "channel",
        "maxResults": 1,
        "key": YOUTUBE_API_KEY
    })
    items = data.get("items", [])
    if not items:
        raise RuntimeError("Channel not found via YouTube API.")
    return items[0]["snippet"]["channelId"]

def list_videos_via_youtube_api(channel_url_or_handle: str):
    base = "https://www.googleapis.com/youtube/v3"
    ch_id = _resolve_channel_id(channel_url_or_handle)

    info = _http_get(f"{base}/channels", {
        "part": "contentDetails",
        "id": ch_id,
        "key": YOUTUBE_API_KEY
    })
    items = info.get("items", [])
    if not items:
        raise RuntimeError("Channel not found (contentDetails).")
    uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    out, page = [], None
    while True:
        data = _http_get(f"{base}/playlistItems", {
            "part": "snippet,contentDetails",
            "playlistId": uploads_pl,
            "maxResults": 50,
            "pageToken": page,
            "key": YOUTUBE_API_KEY
        })
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

# ----------------- Robust downloader (OpenAI path) -----------------
def _write_cookies_if_any(cookies_text: str) -> str | None:
    if not cookies_text or "# Netscape" not in cookies_text:
        return None
    tmp = Path(tempfile.mkdtemp()) / "cookies.txt"
    tmp.write_text(cookies_text, encoding="utf-8")
    return str(tmp)

def _try_download_with_format(url: str, outtmpl: str, fmt: str, cookiefile: str | None, client: str | None) -> Path | None:
    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 12,
        "fragment_retries": 12,
        "sleep_requests": 1,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Referer": "https://www.youtube.com/"},
        "geo_bypass": True,
        "concurrent_fragment_downloads": 1,
        "socket_timeout": 45,
        "format_sort": ["proto:https", "proto:m3u8", "codec:avc:m4a", "res:480", "vbr", "abr", "filesize"],
        "hls_prefer_native": True,
        "nocheckcertificate": True,
        "ignoreerrors": "only_download",
    }
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    if client in ("web", "mweb", "web_embedded", "ios"):
        ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        return None

    outdir = Path(outtmpl).parent
    stem = Path(outtmpl).name.split(".")[0]
    matches = list(outdir.glob(f"{stem}.*"))
    if not matches:
        return None
    best = max(matches, key=lambda p: p.stat().st_size)
    return best if best.exists() and best.stat().st_size > 0 else None

def download_audio_tmp(video_id: str, cookies_text: str = "") -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    outtmpl = str(tmpdir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookiefile = _write_cookies_if_any(cookies_text)

    fmt_list = [
        # audio-only først (https, derefter hls)
        "bestaudio[ext=m4a][protocol^=https]/bestaudio[ext=webm][protocol^=https]/bestaudio[protocol^=https]/bestaudio/best[acodec!=none]",
        "bestaudio[protocol^=m3u8]/bestaudio/best[acodec!=none]",
        # progressive fallback
        "best[acodec!=none][vcodec!=none][protocol^=https]/best[acodec!=none][vcodec!=none]",
        "best[protocol^=m3u8]/best",
    ]
    clients = ["web", "mweb", "web_embedded", "ios"]

    for cli in clients:
        for fmt in fmt_list:
            p = _try_download_with_format(url, outtmpl, fmt, cookiefile, cli)
            if p:
                return p
    raise RuntimeError("No available format could be downloaded (all strategies failed)")

# ----------------- ASR (OpenAI Whisper) -----------------
def transcribe_with_openai(file_path: Path, language: str | None):
    with open(file_path, "rb") as f:
        resp = oa_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language if language else None,
            response_format="verbose_json",
        )
    segs = []
    try:
        segments = getattr(resp, "segments", None)
        text = getattr(resp, "text", None)
    except Exception:
        segments = None; text = None
    if segments:
        for s in segments:
            start = float(s["start"]); end = float(s["end"])
            segs.append({"start": start, "duration": end - start, "text": s["text"].strip()})
    elif text:
        segs = [{"start": 0.0, "duration": 0.0, "text": text}]
    return segs

# ----------------- UI -----------------
st.set_page_config(page_title="YouTube Quote Finder (Cloud)", layout="wide")
st.title("YouTube Quote Finder — Cloud (OpenAI only)")

with st.sidebar:
    st.header("Settings")
    captions_first = st.toggle("Use captions first", value=True)
    language = st.text_input("Language (e.g., 'da' or leave empty for auto)", value="da")
    max_videos = st.number_input("Limit videos (0 = all)", min_value=0, max_value=5000, value=0, step=1)
    st.markdown("**Optional cookies.txt** (Netscape format)")
    cookies_text = st.text_area(
        "Paste cookies (must start with '# Netscape HTTP Cookie File')",
        value="",
        height=120,
        help="Export with 'Get cookies.txt locally' while on youtube.com"
    )
    st.caption("Data is stored in Supabase and shared across your team.")

tab_idx, tab_search = st.tabs(["📦 Index", "🔎 Search"])

def hhmmss(sec: float):
    s = int(sec or 0)
    hh, r = divmod(s, 3600)
    mm, ss = divmod(r, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

with tab_idx:
    st.subheader("Create or update a project")
    project_name = st.text_input("Project name", placeholder="Client A")
    channel_url = st.text_input("Channel handle/URL", placeholder="https://www.youtube.com/@CHANNEL or /channel/ID")
    if st.button("Start indexing", type="primary", disabled=not (project_name and channel_url)):
        pid = get_or_create_project(project_name, channel_url)
        st.info("Fetching video list via YouTube API…")
        try:
            videos = list_videos_via_youtube_api(channel_url)
        except Exception as e:
            st.error(f"Could not list videos: {e}")
            videos = []
        if max_videos and max_videos > 0:
            videos = videos[:max_videos]
        if not videos:
            st.error("No videos found. Check the handle/URL and visibility.")
        else:
            st.success(f"Found {len(videos)} videos.")
            prog = st.progress(0, text="Indexing…")
            status = st.empty()
            total = len(videos); processed = 0
            for v in videos:
                vid = v["video_id"]; video_url = v["url"]
                upsert_video(pid, v)
                status.markdown(f"**Processing:** `{v.get('title','(no title)')}` (`{vid}`)")
                try:
                    segs = None
                    if captions_first:
                        segs = fetch_youtube_captions(vid)
                        if segs:
                            insert_segments(pid, vid, segs)
                    if not segs:
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
                df["timestamp"] = df["start"].fillna(0).map(hhmmss)
                df = df[["title","speaker","content","timestamp","url"]].rename(columns={"content":"quote"})
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "search_results.csv",
                                   "text/csv")
