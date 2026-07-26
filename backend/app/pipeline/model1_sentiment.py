"""
Model 1 — FinBERT NLP sentiment + daily sentiment index
Ported from 03_ML1_MODEL_NLP.ipynb, with one deliberate change from the
original notebook: FinBERT inference runs on Hugging Face's hosted
Inference API instead of a locally-loaded torch model. This keeps the
backend free to host (no ~600MB model in server RAM, no torch/transformers
dependency) at the cost of a bit of network latency per call and being
subject to HF's free-tier rate limits / shared-model cold starts.

If HF_API_TOKEN is unset, sentiment falls back to a lightweight lexicon
scorer (see `_lexicon_sentiment`) so the app still runs end-to-end without
any keys configured — clearly flagged as a fallback in the response, never
silently passed off as FinBERT output.
"""
from __future__ import annotations
import re
import time
import requests
import pandas as pd
import numpy as np

from .. import config

# Words the fallback lexicon scorer uses when no HF token is configured.
_POS_WORDS = {"beat", "beats", "surge", "surges", "record", "profit", "growth", "upgrade",
              "rally", "gain", "gains", "strong", "outperform", "buyback", "dividend", "expansion"}
_NEG_WORDS = {"miss", "misses", "plunge", "plunges", "recall", "probe", "fraud", "downgrade",
              "loss", "losses", "weak", "underperform", "fine", "penalty", "lawsuit", "layoffs", "crash"}


def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lexicon_sentiment(text: str) -> dict:
    """Deterministic keyword-based fallback — used only if HF_API_TOKEN
    isn't set, so the app is runnable end-to-end with zero API keys."""
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.5, "pos_score": 0.33,
                "neg_score": 0.33, "neu_score": 0.34, "sentiment_score": 0.0, "source": "lexicon_fallback"}
    pos_score = pos / total
    neg_score = neg / total
    label = "positive" if pos_score > neg_score else "negative" if neg_score > pos_score else "neutral"
    conf = max(pos_score, neg_score, 0.34)
    return {"sentiment": label, "confidence": round(conf, 4), "pos_score": round(pos_score, 4),
            "neg_score": round(neg_score, 4), "neu_score": round(1 - pos_score - neg_score, 4),
            "sentiment_score": round(pos_score - neg_score, 4), "source": "lexicon_fallback"}


def _parse_hf_response(item) -> dict:
    """HF's text-classification response for one input is a list of
    {label, score} across all 3 FinBERT classes (positive/negative/neutral)."""
    scores = {d["label"].lower(): float(d["score"]) for d in item}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    label = max(scores, key=scores.get)
    return {
        "sentiment": label,
        "confidence": round(scores[label], 4),
        "pos_score": round(pos, 4),
        "neg_score": round(neg, 4),
        "neu_score": round(neu, 4),
        "sentiment_score": round(pos - neg, 4),
        "source": "finbert_hf_api",
    }


def _call_hf_api(texts: list[str]) -> list[dict]:
    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {"inputs": texts, "parameters": {"top_k": None}, "options": {"wait_for_model": True}}

    for attempt in range(config.HF_MAX_RETRIES):
        try:
            r = requests.post(config.HF_API_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException:
            time.sleep(config.HF_RETRY_WAIT_SEC)
            continue

        if r.status_code == 200:
            data = r.json()
            # batch of N texts -> list of N lists of {label, score}
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [_parse_hf_response(item) for item in data]
            # single text -> one list of {label, score}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return [_parse_hf_response(data)]
            raise ValueError(f"Unexpected HF API response shape: {data}")

        if r.status_code == 503:
            wait = r.json().get("estimated_time", config.HF_RETRY_WAIT_SEC)
            time.sleep(min(wait, 20))
            continue

        r.raise_for_status()

    raise RuntimeError("Hugging Face Inference API unavailable after retries — falling back to lexicon scorer.")


def run_finbert_batch(texts: list[str]) -> list[dict]:
    if not config.HF_API_TOKEN:
        return [_lexicon_sentiment(t) for t in texts]

    results: list[dict] = []
    try:
        for i in range(0, len(texts), config.HF_BATCH_SIZE):
            batch = texts[i: i + config.HF_BATCH_SIZE]
            batch = [t if isinstance(t, str) and len(t) > 0 else "neutral" for t in batch]
            results.extend(_call_hf_api(batch))
    except Exception:
        # HF API down / rate-limited mid-run — finish the batch with the
        # lexicon fallback rather than failing the whole analysis.
        remaining = len(texts) - len(results)
        if remaining > 0:
            results.extend(_lexicon_sentiment(t) for t in texts[len(results):])
    return results


def compute_impact_score(row: pd.Series) -> float:
    weight = config.SEVERITY_WEIGHTS.get(row.get("severity_kw", "Low"), 0.5)
    return round(float(row["sentiment_score"]) * weight, 4)


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["ticker", "trigger_date"]).agg(
        daily_sentiment_index=("sentiment_score", "mean"),
        severity_weighted_score=("impact_score", "sum"),
        article_count=("sentiment_score", "count"),
        pos_count=("sentiment", lambda s: (s == "positive").sum()),
        neg_count=("sentiment", lambda s: (s == "negative").sum()),
        avg_confidence=("confidence", "mean"),
    ).reset_index()

    grouped["pos_neg_ratio"] = grouped["pos_count"] / (grouped["neg_count"] + 1)
    grouped["major_event_flag"] = (grouped["severity_weighted_score"].abs() > 1.0).astype(int)
    grouped["high_news_volume"] = (grouped["article_count"] > grouped["article_count"].median()).astype(int)
    grouped.rename(columns={"trigger_date": "price_impact_day"}, inplace=True)
    return grouped


def add_sentiment_momentum_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df
    daily_df = daily_df.sort_values("price_impact_day").copy()
    daily_df["sentiment_ma3"] = daily_df["daily_sentiment_index"].rolling(3, min_periods=1).mean()
    daily_df["sentiment_ma7"] = daily_df["daily_sentiment_index"].rolling(7, min_periods=1).mean()
    daily_df["sentiment_lag1"] = daily_df["daily_sentiment_index"].shift(1).fillna(0)
    daily_df["sentiment_lag2"] = daily_df["daily_sentiment_index"].shift(2).fillna(0)
    daily_df["sentiment_momentum"] = daily_df["daily_sentiment_index"].diff().fillna(0)
    daily_df["sentiment_trend_dir"] = np.sign(daily_df["sentiment_momentum"])
    return daily_df


def run_model1(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (article-level df with sentiment, daily aggregated df)."""
    if news_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    news_df = news_df.copy()
    news_df["text_clean"] = news_df["text"].apply(preprocess_text)

    sentiment_results = run_finbert_batch(news_df["text_clean"].tolist())
    sentiment_df = pd.DataFrame(sentiment_results)

    df = pd.concat([news_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
    df = df[df["confidence"] >= config.CONFIDENCE_THRESHOLD * 0.7]  # soft filter for live/sparse data
    if df.empty:
        return df, pd.DataFrame()

    df["severity_kw"] = df.get("severity_kw", "Low")
    df["impact_score"] = df.apply(compute_impact_score, axis=1)

    daily_df = aggregate_daily_sentiment(df)
    daily_df = add_sentiment_momentum_features(daily_df)

    return df, daily_df
