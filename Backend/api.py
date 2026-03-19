from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
from typing import Optional

import subprocess
import threading
import uuid
import os


# ================================
# PATHS
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = PROJECT_ROOT / "artifacts"

INDEX_PATH = ART_DIR / "hn_rag.index"
META_PATH = ART_DIR / "hn_rag_meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ================================
# APP SETUP
# ================================

app = FastAPI(title="Market Insight RAG API")
print("api.py imported successfully")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://mianalyzer.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# GLOBALS
# ================================

index = None
docs = None
meta = None
embedder = None
loaded_at = None

PIPELINE_SECRET = os.environ.get("PIPELINE_SECRET", "12341234")

PIPELINE_JOBS = {}
PIPELINE_LOCK = threading.Lock()


PAIN_CUES = [
    "pain", "pain point", "problem", "issue", "hard", "difficult", "friction",
    "annoy", "struggle", "fails", "confusing", "complain", "pitfall",
    "refund", "refunds", "chargeback", "chargebacks", "compliance",
    "access control", "entitlement", "entitlements", "permissions", "oauth"
]


# ================================
# UTILITY FUNCTIONS
# ================================

def has_pain_language(text: str) -> bool:
    t = str(text).lower()
    return any(w in t for w in PAIN_CUES)


def is_within_days(created_at, days=7) -> bool:
    dt = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.isna(dt):
        return False
    return dt >= (datetime.now(timezone.utc) - timedelta(days=days))


def load_assets():
    global index, docs, meta, embedder, loaded_at

    import faiss
    from sentence_transformers import SentenceTransformer

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing index file: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta file: {META_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "rb") as f:
        store = pickle.load(f)

    docs = store["docs"]
    meta = store["meta"]

    embedder = SentenceTransformer(MODEL_NAME)
    loaded_at = datetime.now(timezone.utc).isoformat()


def ensure_assets_loaded():
    global index, docs, meta, embedder
    if index is None or docs is None or meta is None or embedder is None:
        load_assets()

# ================================
# STARTUP
# ================================

@app.on_event("startup")
def _startup():
    global loaded_at
    loaded_at = "startup_pending"


# ================================
# PIPELINE BACKGROUND RUNNER
# ================================

def _run_pipeline_job(job_id: str):
    steps = [
        ["python", "pipeline/01_fetch_clean.py"],
        ["python", "pipeline/02_train_score.py"],
        ["python", "pipeline/03_build_index.py"],
    ]

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        with PIPELINE_LOCK:
            PIPELINE_JOBS[job_id]["log"] = "\n".join(log_lines)

    with PIPELINE_LOCK:
        PIPELINE_JOBS[job_id] = {"status": "running", "log": ""}

    try:
        for cmd in steps:
            log(f"Running: {' '.join(cmd)}")

            p = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in p.stdout:
                log(line.rstrip())

            if p.wait() != 0:
                raise RuntimeError("Pipeline step failed")

        log("Reloading RAG index...")
        load_assets()
        log("Pipeline complete ✅")

        with PIPELINE_LOCK:
            PIPELINE_JOBS[job_id]["status"] = "success"

    except Exception as e:
        log(f"ERROR: {str(e)}")
        with PIPELINE_LOCK:
            PIPELINE_JOBS[job_id]["status"] = "error"


# ================================
# API MODELS
# ================================

class SearchReq(BaseModel):
    query: str
    top_k: int = 8
    days: Optional[int] = 30
    theme_filter: Optional[str] = None
    require_pain: bool = True

@app.get("/")
def root():
    return {"message": "Market Insight backend is live"}
# ================================
# HEALTH & RELOAD
# ================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "docs": 0 if docs is None else len(docs),
        "ntotal": int(index.ntotal) if index is not None else 0,
        "loaded_at": loaded_at,
    }


@app.post("/reload")
def reload():
    load_assets()
    return {"ok": True, "loaded_at": loaded_at}


# ================================
# PIPELINE ENDPOINTS (SECURE)
# ================================

@app.post("/pipeline/run")
def pipeline_run(x_api_key: str = Header(None)):
    if x_api_key != PIPELINE_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid.uuid4())

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id,),
        daemon=True,
    )
    thread.start()

    return {"ok": True, "job_id": job_id}


@app.get("/pipeline/status/{job_id}")
def pipeline_status(job_id: str):
    with PIPELINE_LOCK:
        job = PIPELINE_JOBS.get(job_id)

    if not job:
        return {"ok": False, "error": "Invalid job id"}

    return {"ok": True, **job}


# ================================
# SEARCH & THEMES
# ================================

@app.get("/themes")
def themes():
    ensure_assets_loaded()
    return sorted({m.get("market_theme") for m in meta if m.get("market_theme")})


@app.post("/search")
def search(req: SearchReq):
    ensure_assets_loaded()

    q = embedder.encode(
        [req.query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    overfetch = max(req.top_k * 50, 200)
    scores, idx = index.search(q, overfetch)

    out = []

    for sc, i in zip(scores[0], idx[0]):
        if i == -1:
            continue

        m = meta[i]
        doc = docs[i]

        if req.days is not None and not is_within_days(m.get("created_at", ""), days=req.days):
            continue
        if req.theme_filter and m.get("market_theme") != req.theme_filter:
            continue
        if req.require_pain and not has_pain_language(doc):
            continue

        text_part = doc.split("\nText:", 1)[-1].split("\nURL:", 1)[0].strip()

        out.append({
            "score": float(sc),
            "title": m.get("story_title"),
            "theme": m.get("market_theme"),
            "keyword": m.get("keyword"),
            "created_at": m.get("created_at"),
            "url": m.get("url"),
            "preview": text_part[:700],
        })

        if len(out) >= req.top_k:
            break

    return {"results": out}