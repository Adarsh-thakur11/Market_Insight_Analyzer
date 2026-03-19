#!/usr/bin/env python3
from __future__ import annotations

import re
import html
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta


# ----------------------------
# Robust project root + dirs
# ----------------------------
PROJECT_ROOT = Path.cwd()
# if you run from MIANALYZER/pipeline, cwd will already be MIANALYZER or pipeline depending on how you launch
# handle both cases:
if PROJECT_ROOT.name == "pipeline":
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name == "frontend":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR  = PROJECT_ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Data dir:", DATA_DIR)
print("Artifacts dir:", ART_DIR)


# ----------------------------
# Config
# ----------------------------
ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"

KEYWORDS_TO_TRACK = ["film", "platform", "creator", "monetization"]
FETCH_DAYS = 7
MAX_PAGES = 5
HITS_PER_PAGE = 100
SLEEP_S = 0.2

MIN_CLEAN_LEN = 40


LUNIM_KEYWORDS = [
    "market gap", "customer need", "user need",
    "pain point", "problem", "friction",
    "platform", "tool", "product", "solution", "alternative",
    "use case", "value", "benefit", "advantage",
    "adoption", "workflow", "efficiency", "automation",
    "pricing", "cost", "revenue", "monetization",
]

NOISE_KEYWORDS = [
    # force job stuff to False
    "who is hiring", "hiring", "job", "role", "position", "resume", "cv",
    # common HN noise
    "kernel", "compiler", "filesystem", "mmap", "rust", "c++", "linux",
    "crypto", "token", "blockchain",
]


# ----------------------------
# Fetch
# ----------------------------
def fetch_hn(
    keyword: str,
    tag: str = "story",
    hits_per_page: int = 100,
    max_pages: int = 4,
    sleep_s: float = 0.2,
    days: int | None = None,
) -> list[dict]:
    rows: list[dict] = []

    cutoff_ts = None
    if days is not None:
        cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    for page in range(max_pages):
        params = {
            "query": keyword,
            "tags": tag,  # ONE tag only: "story" or "comment"
            "hitsPerPage": hits_per_page,
            "page": page,
        }
        if cutoff_ts is not None:
            params["numericFilters"] = [f"created_at_i>={cutoff_ts}"]

        r = requests.get(ALGOLIA_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for hit in data.get("hits", []):
            rows.append(
                {
                    "source": "HackerNews",
                    "keyword": keyword,
                    "content_type": tag,
                    "objectID": hit.get("objectID"),
                    "created_at": hit.get("created_at"),
                    "author": hit.get("author"),
                    "url": hit.get("url"),
                    "story_title": hit.get("title") or hit.get("story_title"),
                    "text": (hit.get("comment_text") or hit.get("story_text") or ""),
                }
            )

        # stop if last available page
        if page >= data.get("nbPages", 1) - 1:
            break

        if sleep_s:
            time.sleep(sleep_s)

    return rows


# ----------------------------
# Cleaning + relevance
# ----------------------------
def clean_text_fn(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)          # remove HTML tags
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = re.sub(r"\s+", " ", s).strip()      # collapse whitespace
    return s


def mark_lunim_relevance(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()

    if any(nk in t for nk in NOISE_KEYWORDS):
        return False

    return any(lk in t for lk in LUNIM_KEYWORDS)


def main() -> None:
    # Fetch all keywords for stories + comments
    all_rows: list[dict] = []

    for kw in KEYWORDS_TO_TRACK:
        story_rows = fetch_hn(
            kw, tag="story",
            hits_per_page=HITS_PER_PAGE,
            max_pages=MAX_PAGES,
            sleep_s=SLEEP_S,
            days=FETCH_DAYS
        )
        comment_rows = fetch_hn(
            kw, tag="comment",
            hits_per_page=HITS_PER_PAGE,
            max_pages=MAX_PAGES,
            sleep_s=SLEEP_S,
            days=FETCH_DAYS
        )

        print(f"{kw}: stories={len(story_rows)} comments={len(comment_rows)}")
        all_rows.extend(story_rows)
        all_rows.extend(comment_rows)

    df_raw = pd.DataFrame(all_rows)
    if df_raw.empty:
        raise RuntimeError("No rows fetched from Algolia. Check network / API / keywords.")

    # Stable dedupe key
    df_raw["dedupe_id"] = (
        df_raw["content_type"].astype(str).fillna("unknown") + ":" +
        df_raw["objectID"].astype(str).fillna("missing")
    )
    df_raw = df_raw.drop_duplicates(subset=["dedupe_id"]).copy()
    df_raw["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # Clean
    df = df_raw.copy()
    df["clean_text"] = (
        df["story_title"].fillna("").astype(str) + ". " +
        df["text"].fillna("").astype(str)
    ).apply(clean_text_fn)

    df["clean_length"] = df["clean_text"].str.len()
    df = df[df["clean_length"] >= MIN_CLEAN_LEN].copy()

    # Relevance
    df["is_relevant"] = df["clean_text"].apply(mark_lunim_relevance)

    df_relevant = df[df["is_relevant"] == True].copy()
    df_noise = df[df["is_relevant"] == False].copy()

    # Save outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    df.to_csv(DATA_DIR / f"hn_market_labeled_{ts}.csv", index=False)
    df_relevant.to_csv(DATA_DIR / f"hn_market_relevant_{ts}.csv", index=False)
    df_noise.to_csv(DATA_DIR / f"hn_market_noise_{ts}.csv", index=False)

    df.to_csv(DATA_DIR / "hn_market_labeled_latest.csv", index=False)
    df_relevant.to_csv(DATA_DIR / "hn_market_relevant_latest.csv", index=False)
    df_noise.to_csv(DATA_DIR / "hn_market_noise_latest.csv", index=False)

    # Summary
    newest = pd.to_datetime(df_relevant["created_at"], utc=True, errors="coerce").max()
    print("\n✅ Fetch+Clean complete")
    print("Total rows (after dedupe):", len(df_raw))
    print("Rows after length filter :", len(df))
    print("Relevant rows            :", len(df_relevant))
    print("Noise rows               :", len(df_noise))
    print("Newest relevant created_at:", newest)
    print("Saved latest relevant ->", DATA_DIR / "hn_market_relevant_latest.csv")


if __name__ == "__main__":
    main()