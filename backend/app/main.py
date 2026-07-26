"""
Fin-Agent India — FastAPI backend
6-agent/model pipeline: anomaly detection -> Indian news retrieval ->
FinBERT sentiment -> hybrid ML forecasting -> BUY/HOLD/SELL verdict ->
SHAP + Granger explainability.
"""
from __future__ import annotations
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from .pipeline import orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finagent")

app = FastAPI(title="Fin-Agent India API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)
_running: set[str] = set()

# No local model to preload — FinBERT sentiment runs through the Hugging
# Face Inference API (see pipeline/model1_sentiment.py). The backend has
# no torch/transformers dependency and needs no warm-up at boot.
if not config.HF_API_TOKEN:
    logger.warning(
        "HF_API_TOKEN is not set — sentiment scoring will use the lexicon "
        "fallback instead of real FinBERT. Set HF_API_TOKEN to enable it."
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sentiment_engine": "finbert_hf_api" if config.HF_API_TOKEN else "lexicon_fallback (no HF_API_TOKEN set)",
    }


@app.get("/api/tickers")
def get_tickers():
    return [{"ticker": t, "company": name} for t, name in config.TICKERS.items()]


@app.get("/api/dashboard")
def get_dashboard():
    """Returns cached latest results for all tickers (fast — no live compute)."""
    return db.list_cached_summaries()


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str, force: bool = False):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'. See /api/tickers.")

    if not force:
        cached = db.get_cached(ticker)
        if cached:
            return cached

    if ticker in _running:
        raise HTTPException(status_code=409, detail="Analysis already in progress for this ticker — try again shortly.")

    try:
        _running.add(ticker)
        result = orchestrator.run_full_pipeline(ticker)
        if "error" not in result:
            db.set_cached(ticker, result)
        return result
    except Exception as e:
        logger.error("Pipeline failed for %s:\n%s", ticker, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        _running.discard(ticker)


@app.post("/api/refresh-all")
def refresh_all(background_tasks: BackgroundTasks):
    """Kicks off a background refresh of every ticker — for a scheduled cron hit."""
    def _run_all():
        for ticker in config.TICKERS:
            try:
                result = orchestrator.run_full_pipeline(ticker)
                if "error" not in result:
                    db.set_cached(ticker, result)
            except Exception:
                logger.error("Refresh failed for %s:\n%s", ticker, traceback.format_exc())

    background_tasks.add_task(_run_all)
    return {"status": "refresh_started", "tickers": list(config.TICKERS.keys())}
