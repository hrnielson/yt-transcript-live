# app.py — YouTube Quote Finder (Cloud, OpenAI-only)
# Streamlit + Supabase + YouTube Data API v3 + Captions-first + OpenAI Whisper
# 2025-ready edition with robust yt-dlp, caption fallbacks, Whisper fallback, idempotent DB, and temp cleanup

import os, re, time, datetime, tempfile, shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import httpx
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from supabase import create_client, Client
from openai import OpenAI

# ---------- Secrets / Clients ----------
SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

supabase: Client = create_client(SB_URL, SB_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)

# ---------- Supabase helpers ----------
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
    # requires PK/unique on id
    supabase.table("videos").upsert(row, on_conflict="id").execute()

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
    # requires unique index on (project_id, video_id, start)
    for i in range(0, len(rows), 500):
        supabase.table("segments").upsert(rows[i:i+500], on_conflict="project_id,video_id,start").execute()

def search_segments(project_id: str, q: str, limit: int = 100):
    r = supabase.rpc("segments_search", {"p_project": project_id, "p_query": q, "p_limit": limit}).execute()
    return r.data or []

# ---------- HTTP helper ----------
def _http_get(url: str, params: dict, timeout=60):
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code} on {url.split('/')[-1]} with params: { {k:v for k,v in params.items() if k!='key'} }") from e
        return r.json()

# ---------- YouTube Data API ----------
def _extract_bits(inp: str):
    s = (inp or "").strip()
    if s.startswith("@"): return s, None, None, None
    m = re.search(r"youtube\.com/(channel/([^/?#]+))", s, re.I)
    if m: return None, m.group(2), None, None
    m = re.search(r"youtube\.com/@([^/?#]+)", s, re.I)
    if m: return "@"+m.group(1), None, None, None
    m = re.search(r"youtube\.com/user/([^/?#]+)", s, re.I)
    if m: return None, None, m.group(1), None
    m = re.search(r"youtube\.com/c/([^/?#]+)", s, re.I)
    if m: return None, None, None, m.group(1)
    return None, None, None, None

def _resolve_channel_id(handle_or_url: str) -> str:
    base = "https://www.googleapis.com/youtube/v3"
    handle, ch_id, username, custom = _extract_bits(handle_or_url)
    if ch_id: return ch_id
    if handle:
        data = _http_get(f"{base}/channels", {"part":"id","forHandle":handle,"key":YOUTUBE_API_KEY})
        items = data.get("items", [])
        if items: return items[0]["id"]
    if username:
        data = _http_get(f"{base}/channels", {"part":"id","forUsername":username,"key":YOUTUBE_API_KEY})
        items = data.get("items", [])
        if items: return items[0]["id"]
    q = (handle_or_url or "").replace("https://www.youtube.com/"," ").strip()
    data = _http_get(f"{base}/search", {"part":"snippet","type":"channel","q":q or "","maxResults":1,"key":YOUTUBE_API_KEY})
    items = data.get("items", [])
    if items: return items[0]["snippet"]["channelId"]
    raise RuntimeError("Channel not found via YouTube API.")

def list_videos_by_channel_id(channel_id: str):
    base = "https://www.googleapis.com/youtube/v3"
    info = _http_get(f"{base}/channels", {"part":"contentDetails","id":channel_id,"key":YOUTUBE_API_KEY})
    items = info.get("items", [])
    if not items:
        raise RuntimeError("Channel not found (contentDetails) for provided channelId.")
    uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    out, page = [], None
    while True:
        data = _http_get(f"{base}/playlistItems", {
            "part":"snippet,contentDetails",
            "playlistId":uploads_pl,
            "maxResults":50,
            "pageToken":page,
            "key":YOUTUBE_API_KEY
        })
        for it in data.get("items", []):
            vid = it["contentDetails"]["videoId"]
            title = it["snippet"]["title"]
            published_at = it["contentDetails"].get("videoPublishedAt")
            url = f"https://www.youtube.com/watch?v={vid}"
            out.append({"video_id":vid,"title":title,"published_at":published_at,"url":url})
        page = data.get("nextPageToken")
        if not page: break
    return out

# ---------- Captions ----------
def fetch_youtube_captions(video_id: str, preferred=("da","en","no","sv")):
    try:
        ts_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # 1) Direkte match på foretrukne sprog
        for lang in preferred:
            try:
                tr = ts_list.find_transcript([lang])
                return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr.fetch()]
            except Exception:
                pass

        # 2) Manuelt lavede > auto-generated
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

        # 3) Oversæt til engelsk som sidste fallback (hvis muligt)
        try:
            for t in ts_list:
                try:
                    tr_en = t.translate('en')
                    return [{"start": s["start"], "duration": s.get("duration", 0), "text": s.get("text", "")} for s in tr_en.fetch()]
                except Exception:
                    continue
        except Exception:
            return None

    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

# ---------- Downloader ----------
def _try_download(url, outtmpl, fmt, cookiefile, merge_to="m4a"):
    ytdl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 4,
        "fragment_retries": 4,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "geo_bypass": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": merge_to, "preferredquality": "0"}
        ],
        "merge_output_format": merge_to,
        "concurrent_fragment_downloads": 3,
        "overwrites": True,
        "force_ipv4": True,
    }
    if cookiefile:
        ytdl_opts["cookiefile"] = cookiefile

    try:
        with yt_dlp.YoutubeDL(ytdl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.caption(f"yt-dlp fejl ({fmt}): {e}")
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

    # Hårdfør rækkefølge: specifikke format-id'er -> brede fallbacks
    fmt_list = [
        "140",                         # m4a 128kbps (meget udbredt)
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
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if cookiefile:
                shutil.rmtree(Path(cookiefile).parent, ignore_errors=True)
        except Exception:
            pass
        raise

# ---------- UI ----------
st.set_page_config(page_title="YouTube Quote Finder", layout="wide")
st.title("🎬 YouTube Quote Finder")

with st.sidebar:
    st.header("Settings")
    captions_first = st.toggle("Use captions first", value=True)
    language = st.text_input("Language (e.g., 'da' or leave empty)", value="da")
    max_videos = st.number_input("Limit videos (0 = all)", 0, 5000, 0)
    channel_id_override = st.text_input(
        "Channel ID override (optional, starts with 'UC')",
        help="Paste the channelId (UCxxxxxxxxxxxxxxxxxxxxxx) to skip handle/URL detection."
    )
    cookies_text = st.text_area(
        "Optional cookies.txt (Netscape format)",
        height=120, help="Export with 'Get cookies.txt locally' while on youtube.com"
    )

if "pid" not in st.session_state:
    st.session_state.pid = None
if "resolved_channel_id" not in st.session_state:
    st.session_state.resolved_channel_id = None


tab_idx, tab_search = st.tabs(["📦 Index", "🔎 Search"])

def hhmmss(sec: float):
    s = int(round(sec or 0))
    h, r = divmod(s,3600)
    m, s = divmod(r,60)
    return f"{h:02d}:{m:02d}:{s:02d}"

with tab_idx:
    st.subheader("Create or update a project")
    project_name = st.text_input("Project name", placeholder="Client A")
    channel_url = st.text_input("Channel handle or URL", placeholder="@brand or https://www.youtube.com/@brand")
    if st.button("Start indexing", type="primary", disabled=not(project_name and channel_url)):
        pid = get_or_create_project(project_name, channel_url)
        st.session_state.pid = pid
        st.info("Fetching video list via YouTube Data API…")
        try:
            chan_id = channel_id_override.strip()
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

        if max_videos>0: videos = videos[:max_videos]
        if not videos:
            st.error("No videos found. Check Channel ID, handle, or visibility.")
        else:
            st.success(f"Found {len(videos)} videos.")
            prog = st.progress(0, text="Indexing…")
            total=len(videos); done=0
            for v in videos:
                vid=v["video_id"]; upsert_video(pid,v)
                st.write(f"Processing {v['title']}")
                segs=None
                audio_path=None
                cookiefile=None
                try:
                    if captions_first:
                        segs=fetch_youtube_captions(vid)
                        if segs:
                            insert_segments(pid,vid,segs)
                    if not segs:
                        audio_path=download_audio_tmp(vid,cookies_text)
                        segs=transcribe_with_openai(audio_path, (language or "").split("-")[0].strip() or None)
                        insert_segments(pid,vid,segs)
                    if not segs:
                        st.warning(f"No segments for {vid} (captions+ASR failed).")
                except Exception as e:
                    st.warning(f"Skipped {vid}: {e}")
                finally:
                    # cleanup temp dir for this video
                    try:
                        if audio_path and isinstance(audio_path, Path):
                            tmp_parent = audio_path.parent
                            shutil.rmtree(tmp_parent, ignore_errors=True)
                    except Exception:
                        pass
                done+=1
                prog.progress(int(done/total*100), text=f"Indexing… {done}/{total}")
            st.success("✅ Done indexing.")

    st.divider()
    st.subheader("Existing projects")
    prjs=list_projects()
    if prjs:
        st.dataframe(pd.DataFrame(prjs), use_container_width=True)
    else:
        st.info("No projects yet.")

with tab_search:
    st.subheader("Search quotes")
    prjs=list_projects()
    if not prjs: st.info("Create a project first.")
    else:
        options={p["name"]:p["id"] for p in prjs}
        sel=st.selectbox("Project", list(options.keys()))
        q=st.text_input("Search term", value="hammer")
        limit=st.slider("Max results",10,500,50,10)
        if st.button("Search"):
            rows=search_segments(options[sel], q.strip(), limit)
            if not rows: st.info("No results.")
            else:
                df=pd.DataFrame(rows)
                if "start" in df:
                    df["timestamp"]=df["start"].fillna(0).map(hhmmss)
                cols=[c for c in ["title","speaker","content","timestamp","url"] if c in df.columns]
                df=df[cols].rename(columns={"content":"quote"})
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "search_results.csv","text/csv")
