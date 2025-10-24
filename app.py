# app.py — YouTube Quote Finder (Cloud, OpenAI-only)
# Streamlit + Supabase + YouTube Data API v3 + Captions-first + OpenAI Whisper
# One-file version with: project language, dedup, robust yt-dlp, defensive Supabase, RPC search + rich fallback.

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
import json

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
        if "lang" in payload:
            payload.pop("lang", None)
            ins = supabase.table("projects").insert(payload).execute()
            return ins.data[0]["id"]
        raise


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

    for win in _windows_from_segments(segs):
        text = "\n".join((s.get("content") or "").strip() for s in win if s.get("content"))
        if not text.strip():
            continue
        user_prompt = f"Sprog: {lang_hint or 'auto'}.\nTekst:\n{text}\n\nReturnér KUN JSON som beskrevet i systemprompten."
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
                pass
    return total_inserted

# ---------- Rest af koden (uændret) ---------- 
# ... [behold alt fra YouTube API, fetch_youtube_captions, etc. som i dit eksisterende script]
#  ↓↓↓

# (indsæt din eksisterende kode herfra ned til indekseringsloopen — uændret)

# ---------- INDEXING LOOP TILFØJELSE ----------
# Efter du har indsat segments og markeret videoen som indekseret, indsæt dette:

    try:
        inserted = extract_quotes_from_video(pid, vid, project_lang, source=("captions" if captions_first else "asr"))
        if inserted:
            st.write(f"➕ Extracted {inserted} publish-ready quotes")
    except Exception as e:
        st.caption(f"Quote extraction skipped: {e}")

# ---------- UI: Ny tab for Quotes ----------

tab_idx, tab_search, tab_quotes = st.tabs(["📦 Index", "🔎 Search", "💬 Quotes"])

# Efter Search-tab, tilføj denne nye sektion:

with tab_quotes:
    st.subheader("Quotes")
    prjs = list_projects()
    if not prjs:
        st.info("Create a project first.")
    else:
        options = {p["name"]: p["id"] for p in prjs}
        sel = st.selectbox("Project", list(options.keys()))
        q = st.text_input("Filter (ILIKE over quote/topic/tags/attribution)", value="")
        limit = st.slider("Max rows", 10, 500, 100, 10)

        qb = supabase.table("quotes")\
            .select("quote,original,topic,tags,lang,start,video_id,project_id,attribution,paraphrased")\
            .eq("project_id", options[sel])\
            .order("created_at", desc=True)\
            .limit(int(limit))

        rows = qb.execute().data or []
        if q.strip():
            like = f"%{q.strip()}%"
            rows = [r for r in rows if
                    (r.get("quote") and q.lower() in r["quote"].lower()) or
                    (r.get("topic") and q.lower() in r["topic"].lower()) or
                    (r.get("attribution") and q.lower() in r["attribution"].lower()) or
                    any(q.lower() in (t or "").lower() for t in (r.get("tags") or []))
                   ][:int(limit)]

        vid_ids = sorted({r["video_id"] for r in rows})
        videos_map = {}
        if vid_ids:
            vids = supabase.table("videos").select("id,title,url").in_("id", vid_ids).execute().data or []
            videos_map = {v["id"]: {"title": v.get("title"), "url": v.get("url")} for v in vids}

        if not rows:
            st.info("No quotes yet.")
        else:
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

            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "quotes.csv", "text/csv")
