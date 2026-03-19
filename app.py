import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime, timezone, timedelta

# --- paths (use your real paths) ---
INDEX_PATH = "hn_rag.index"
META_PATH = "hn_rag_meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_DEFAULT = 8

# --- load once (important for speed) ---
@st.cache_resource
def load_rag():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        store = pickle.load(f)
    embedder = SentenceTransformer(MODEL_NAME)
    return index, store["docs"], store["meta"], embedder

index, docs, meta, embedder = load_rag()

PAIN_CUES = [
    "pain", "pain point", "problem", "issue", "hard", "difficult", "friction",
    "annoy", "struggle", "fails", "confusing", "complain", "pitfall",
    "refund", "refunds", "chargeback", "chargebacks", "compliance",
    "access control", "entitlement", "entitlements", "permissions", "oauth"
]

def has_pain_language(text: str) -> bool:
    t = str(text).lower()
    return any(w in t for w in PAIN_CUES)

def is_within_days(created_at, days=7) -> bool:
    dt = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.isna(dt):
        return False
    return dt >= (datetime.now(timezone.utc) - timedelta(days=days))

def rag_search(query: str, top_k: int, theme_filter: str|None, days: int|None, require_pain: bool):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    overfetch = max(top_k * 50, 200)
    scores, idx = index.search(q, overfetch)

    results = []
    for sc, i in zip(scores[0], idx[0]):
        if i == -1:
            continue
        m = meta[i]
        doc = docs[i]

        if days is not None and not is_within_days(m.get("created_at", ""), days=days):
            continue
        if theme_filter and m.get("market_theme") != theme_filter:
            continue
        if require_pain and not has_pain_language(doc):
            continue

        results.append((float(sc), m, doc))
        if len(results) >= top_k:
            break
    return results

# --- UI ---
st.title("Market Insight RAG (HN)")

query = st.text_input("Ask a question", placeholder="e.g., What are people complaining about in creator platforms this week?")
col1, col2, col3 = st.columns(3)

with col1:
    top_k = st.slider("Top K", 3, 15, TOP_K_DEFAULT)
with col2:
    days = st.selectbox("Time window", [7, 14, 30, None], format_func=lambda x: "All time" if x is None else f"Last {x} days")
with col3:
    require_pain = st.checkbox("Require pain language", value=True)

# Optional: theme filter (read from your stored meta)
all_themes = sorted({m.get("market_theme") for m in meta if m.get("market_theme")})
theme_filter = st.selectbox("Theme filter (optional)", ["(Any)"] + all_themes)
theme_filter = None if theme_filter == "(Any)" else theme_filter

if st.button("Search") and query.strip():
    results = rag_search(query.strip(), top_k, theme_filter, days, require_pain)

    if not results:
        st.warning("No results found with these filters.")
    else:
        for rank, (sc, m, doc) in enumerate(results, start=1):
            st.markdown(f"### {rank}) {m.get('story_title','(no title)')}")
            st.caption(f"score={sc:.3f} | theme={m.get('market_theme')} | keyword={m.get('keyword')} | date={m.get('created_at')}")
            st.write(m.get("url"))
            text_part = doc.split("\nText:", 1)[-1].strip()
            st.write(text_part[:700] + ("..." if len(text_part) > 700 else ""))
            st.divider()