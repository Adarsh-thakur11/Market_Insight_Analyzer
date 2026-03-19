#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Robust project root + dirs
# ----------------------------
PROJECT_ROOT = Path.cwd()
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
# Paths
# ----------------------------
IN_PATH = DATA_DIR / "hn_market_relevant_latest.csv"

OUT_SCORED   = DATA_DIR / "hn_scored_latest.csv"
OUT_INSIGHTS = DATA_DIR / "insights_latest.csv"
OUT_QUOTES   = DATA_DIR / "quotes_latest.csv"

VECTORIZER_PATH = ART_DIR / "tfidf_vectorizer.joblib"
KMEANS_PATH     = ART_DIR / "kmeans_model.joblib"
THEME_MAP_PATH  = ART_DIR / "cluster_name_map.json"


# ----------------------------
# Config
# ----------------------------
K = 8
RANDOM_STATE = 42

MIN_CLEAN_LEN = 40
MIN_QUOTE_LEN = 60

TFIDF_MAX_FEATURES = 6000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3


# ----------------------------
# Theme map (cluster -> label)
# ----------------------------
CLUSTER_NAME_MAP = {
    0: "Pricing & Monetization Concerns",
    1: "Early Product Feedback & Tool Discovery",
    2: "Platform Perception & Social Sentiment",
    3: "Ad-Based Monetization Trade-offs",
    4: "Developer Ecosystem & Integrations",
    5: "AI Content Platforms & Automation",
    6: "Creator Economy & AI Disruption",
    7: "Open Platform & Trust Signals",
}


# ----------------------------
# Helpers
# ----------------------------
def add_hn_item_url(df: pd.DataFrame, objectid_col="objectID", url_col="url") -> pd.DataFrame:
    df = df.copy()
    if objectid_col in df.columns:
        hn_link = "https://news.ycombinator.com/item?id=" + df[objectid_col].astype(str)
        if url_col not in df.columns:
            df[url_col] = hn_link
        else:
            df[url_col] = df[url_col].fillna(hn_link)
    return df


def representative_quotes(
    df: pd.DataFrame,
    theme_col="market_theme",
    text_col="clean_text",
    n_quotes=3,
    min_len=60,
) -> pd.DataFrame:
    results = []
    for theme, g in df.dropna(subset=[theme_col, text_col]).groupby(theme_col):
        g = g[g[text_col].astype(str).str.len() >= min_len].copy()
        if g.empty:
            continue

        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=TFIDF_MAX_FEATURES,
        )
        X = vec.fit_transform(g[text_col].astype(str).tolist())
        centroid = np.asarray(X.mean(axis=0))
        sims = cosine_similarity(X, centroid).ravel()

        top_idx = np.argsort(sims)[::-1][:n_quotes]
        for rank, idx in enumerate(top_idx, start=1):
            row = g.iloc[idx]
            results.append(
                {
                    "market_theme": theme,
                    "rank": rank,
                    "keyword": row.get("keyword", ""),
                    "story_title": row.get("story_title", ""),
                    "author": row.get("author", ""),
                    "url": row.get("url", ""),
                    "objectID": row.get("objectID", ""),
                    "clean_text": row.get(text_col, ""),
                }
            )

    if not results:
        return pd.DataFrame(columns=["market_theme","rank","keyword","story_title","author","url","objectID","clean_text"])

    return pd.DataFrame(results).sort_values(["market_theme", "rank"])


def theme_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["created_at"] = pd.to_datetime(tmp.get("created_at", None), errors="coerce", utc=True)
    stats = (
        tmp.groupby("market_theme")
        .agg(
            mentions=("clean_text", "count"),
            newest=("created_at", "max"),
            oldest=("created_at", "min"),
            unique_authors=("author", "nunique"),
        )
        .reset_index()
    )
    stats["days_span"] = (stats["newest"] - stats["oldest"]).dt.days
    return stats.sort_values("mentions", ascending=False)


def top_signal_phrases(df: pd.DataFrame, theme: str, text_col="clean_text", top_k=8) -> list[str]:
    g = df[df["market_theme"] == theme].copy()
    texts = g[text_col].astype(str).tolist()
    if len(texts) < 3:
        return []

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=8000)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    top_idx = np.argsort(mean_scores)[::-1][:top_k]
    return terms[top_idx].tolist()


def top_keywords_for_theme(df: pd.DataFrame, theme: str, n=3) -> list[str]:
    vals = df.loc[df["market_theme"] == theme, "keyword"].dropna().astype(str)
    if len(vals) == 0:
        return []
    return [k for k, _ in Counter(vals).most_common(n)]


def strip_urls(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = strip_urls(text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if 40 <= len(s.strip()) <= 240 and "hn:" not in s.lower()]


PAIN_WORDS = ["problem","pain","hard","difficult","issue","friction","annoy","struggle","fails","limitation","slow"]
OPP_WORDS  = ["opportunity","could","should","would","improve","better","build","need","demand","use case","workflow","automate","simplify","value"]
RISK_WORDS = ["risk","concern","legal","privacy","scam","fraud","ban","abuse","unsafe","security","trust","misuse","policy","compliance"]


def build_actionable_insights(df: pd.DataFrame, theme: str, n_each=3, max_rows=600) -> dict:
    g = df[df["market_theme"] == theme].copy()
    if g.empty:
        return {"pain_signals": "", "opportunity_signals": "", "risk_signals": ""}

    g = g.sample(min(len(g), max_rows), random_state=42)

    rows = []
    for _, r in g.iterrows():
        for s in split_sentences(r.get("clean_text", "")):
            rows.append({"sent": s})
    sent_df = pd.DataFrame(rows).drop_duplicates(subset=["sent"])
    if sent_df.empty:
        return {"pain_signals": "", "opportunity_signals": "", "risk_signals": ""}

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=TFIDF_MAX_FEATURES)
    X = vec.fit_transform(sent_df["sent"].tolist())
    centroid = np.asarray(X.mean(axis=0))
    sims = cosine_similarity(X, centroid).ravel()
    sent_df["score"] = sims

    def contains_any(sent: str, words: list[str]) -> bool:
        sl = sent.lower()
        return any(w in sl for w in words)

    def pick_top(words: list[str]) -> list[str]:
        subset = sent_df[sent_df["sent"].apply(lambda s: contains_any(s, words))].sort_values("score", ascending=False)
        out = []
        for s in subset["sent"].tolist():
            if s.lower() not in [x.lower() for x in out]:
                out.append(s)
            if len(out) >= n_each:
                break
        return out

    def fallback() -> list[str]:
        return sent_df.sort_values("score", ascending=False)["sent"].head(n_each).tolist()

    pain = pick_top(PAIN_WORDS) or fallback()
    opp  = pick_top(OPP_WORDS)  or fallback()
    risk = pick_top(RISK_WORDS) or fallback()

    return {
        "pain_signals": " | ".join(pain[:n_each]),
        "opportunity_signals": " | ".join(opp[:n_each]),
        "risk_signals": " | ".join(risk[:n_each]),
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if set(CLUSTER_NAME_MAP.keys()) != set(range(K)):
        raise ValueError("CLUSTER_NAME_MAP keys must be 0..K-1")

    # Load
    df = pd.read_csv(IN_PATH)
    if "clean_text" not in df.columns:
        raise ValueError("Missing required column: clean_text")

    df["clean_text"] = df["clean_text"].astype(str).fillna("")
    df = df[df["clean_text"].str.len() >= MIN_CLEAN_LEN].copy()

    print("Training/scoring rows:", len(df))

    # Train TF-IDF + KMeans
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )
    X = vectorizer.fit_transform(df["clean_text"].tolist())

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    clusters = kmeans.fit_predict(X)

    # Save models + map
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    with open(THEME_MAP_PATH, "w") as f:
        json.dump(CLUSTER_NAME_MAP, f, indent=2)

    # Apply scoring
    df["cluster"] = clusters
    df["market_theme"] = df["cluster"].map(CLUSTER_NAME_MAP)
    df = add_hn_item_url(df)

    # Quotes + insights
    quotes_df = representative_quotes(df, n_quotes=3, min_len=MIN_QUOTE_LEN)

    stats = theme_stats(df)
    rows = []
    for _, r in stats.iterrows():
        theme = r["market_theme"]
        phrases = top_signal_phrases(df, theme, top_k=8)
        top_keys = top_keywords_for_theme(df, theme, n=3)
        q = quotes_df[quotes_df["market_theme"] == theme].sort_values("rank")
        sig = build_actionable_insights(df, theme)

        rows.append(
            {
                "market_theme": theme,
                "mentions": int(r["mentions"]),
                "unique_authors": int(r["unique_authors"]) if pd.notna(r["unique_authors"]) else 0,
                "newest": r["newest"],
                "days_span": r["days_span"],
                "top_signal_phrases": ", ".join(phrases),
                "insight_summary": f"Conversation clusters around {', '.join(phrases[:4])}. Top source keywords: {', '.join(top_keys)}.",
                "quote_1": q.iloc[0]["clean_text"] if len(q) > 0 else "",
                "quote_1_title": q.iloc[0]["story_title"] if len(q) > 0 else "",
                "quote_1_url": q.iloc[0]["url"] if len(q) > 0 else "",
                "quote_2": q.iloc[1]["clean_text"] if len(q) > 1 else "",
                "quote_2_title": q.iloc[1]["story_title"] if len(q) > 1 else "",
                "quote_2_url": q.iloc[1]["url"] if len(q) > 1 else "",
                **sig,
            }
        )

    insights_df = pd.DataFrame(rows).sort_values("mentions", ascending=False)

    # Save
    df.to_csv(OUT_SCORED, index=False)
    insights_df.to_csv(OUT_INSIGHTS, index=False)
    quotes_df.to_csv(OUT_QUOTES, index=False)

    # Summary
    newest = pd.to_datetime(df.get("created_at", None), utc=True, errors="coerce").max()
    print("\n✅ Train+Score complete")
    print("Saved scored  ->", OUT_SCORED)
    print("Saved insights->", OUT_INSIGHTS)
    print("Saved quotes  ->", OUT_QUOTES)
    print("Saved models  ->", VECTORIZER_PATH, KMEANS_PATH, THEME_MAP_PATH)
    print("Rows scored:", len(df), "| Newest created_at:", newest)


if __name__ == "__main__":
    main()