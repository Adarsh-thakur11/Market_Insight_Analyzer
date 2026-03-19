#!/usr/bin/env python3
from __future__ import annotations

import pickle
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


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
# Paths + config
# ----------------------------
INPUT_PATH = DATA_DIR / "hn_scored_latest.csv"
INDEX_PATH = ART_DIR / "hn_rag.index"
META_PATH  = ART_DIR / "hn_rag_meta.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COL = "clean_text"
THEME_COL = "market_theme"

MIN_LEN = 40


def main() -> None:
    print("Reading scored:", INPUT_PATH)
    print("Writing index :", INDEX_PATH)
    print("Writing meta  :", META_PATH)

    # ---------- Load ----------
    df = pd.read_csv(INPUT_PATH)
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df = df[df[TEXT_COL].str.len() >= MIN_LEN].copy()

    # ---------- Build docs ----------
    docs = []
    meta = []

    for _, r in df.iterrows():
        text = r.get(TEXT_COL, "")
        title = str(r.get("story_title", "") or "")
        theme = str(r.get(THEME_COL, "") or "")
        url = str(r.get("url", "") or "")
        created_at = str(r.get("created_at", "") or "")
        keyword = str(r.get("keyword", "") or "")
        objectID = str(r.get("objectID", "") or "")

        doc = (
            f"Title: {title}\n"
            f"Theme: {theme}\n"
            f"Keyword: {keyword}\n"
            f"Date: {created_at}\n"
            f"Text: {text}\n"
            f"URL: {url}"
        )

        docs.append(doc)
        meta.append(
            {
                "story_title": title,
                "market_theme": theme,
                "keyword": keyword,
                "created_at": created_at,
                "url": url,
                "objectID": objectID,
            }
        )

    print("Docs:", len(docs))

    # ---------- Embed ----------
    embedder = SentenceTransformer(MODEL_NAME)
    emb = embedder.encode(
        docs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ---------- FAISS index ----------
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity if vectors are normalized
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"meta": meta, "docs": docs}, f)

    print("\n✅ Build index complete")
    print("INDEX_PATH:", INDEX_PATH)
    print("META_PATH :", META_PATH)
    print("FAISS ntotal:", index.ntotal)
    print("Docs:", len(docs), "Meta:", len(meta))


if __name__ == "__main__":
    main()