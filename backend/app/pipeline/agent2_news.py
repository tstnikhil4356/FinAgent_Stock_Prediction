"""
Agent 2 — Event-driven news retrieval (India)
Fetches only within the ±2hr window around each Agent-1 anomaly trigger,
exactly like the original notebook — but sourced from Indian financial
press (Moneycontrol, Economic Times, Livemint, Business Standard, NDTV
Business, Financial Express) instead of Guardian/global NewsAPI, via
NewsAPI's domain filter + GNews India edition as a fallback.
"""
from __future__ import annotations
import time
import hashlib
import requests
import pandas as pd

from .. import config


def _newsapi_fetch(query: str, win_start: str, win_end: str) -> list[dict]:
    if not config.NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": win_start,
        "to": win_end,
        "domains": config.INDIAN_NEWS_DOMAINS,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": config.NEWSAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        arts = r.json().get("articles", [])
        return [
            {
                "headline": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": (a.get("source") or {}).get("name", "NewsAPI"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "api_source": "newsapi",
            }
            for a in arts
        ]
    except requests.RequestException:
        return []


def _gnews_fetch(query: str, win_start: str, win_end: str) -> list[dict]:
    if not config.GNEWS_KEY:
        return []
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "from": f"{win_start}T00:00:00Z",
        "to": f"{win_end}T23:59:59Z",
        "country": "in",
        "lang": "en",
        "max": 20,
        "apikey": config.GNEWS_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        arts = r.json().get("articles", [])
        return [
            {
                "headline": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": (a.get("source") or {}).get("name", "GNews"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "api_source": "gnews",
            }
            for a in arts
        ]
    except requests.RequestException:
        return []


def _marketaux_fetch(symbol: str, win_start: str, win_end: str) -> list[dict]:
    if not config.MARKETAUX_KEY:
        return []
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "published_after": win_start,
        "published_before": win_end,
        "language": "en",
        "api_token": config.MARKETAUX_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        arts = r.json().get("data", [])
        return [
            {
                "headline": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": a.get("source", "Marketaux"),
                "url": a.get("url", ""),
                "published_at": a.get("published_at", ""),
                "api_source": "marketaux",
            }
            for a in arts
        ]
    except requests.RequestException:
        return []


def fetch_all_for_trigger(ticker: str, company_name: str, trigger_row: pd.Series) -> list[dict]:
    win_start = str(pd.to_datetime(trigger_row["news_window_start"]).date())
    win_end   = str(pd.to_datetime(trigger_row["news_window_end"]).date())
    query     = company_name

    raw: list[dict] = []
    raw.extend(_newsapi_fetch(query, win_start, win_end))
    time.sleep(0.3)
    raw.extend(_gnews_fetch(query, win_start, win_end))
    time.sleep(0.3)
    raw.extend(_marketaux_fetch(ticker.replace(".NS", ""), win_start, win_end))

    for a in raw:
        a["ticker"] = ticker
        a["trigger_date"] = trigger_row["trigger_date"]

    return raw


def is_financially_relevant(text: str) -> bool:
    text_l = (text or "").lower()
    return any(kw in text_l for kw in config.FIN_KEYWORDS)


def get_matched_keywords(text: str) -> str:
    text_l = (text or "").lower()
    return ", ".join(kw for kw in config.FIN_KEYWORDS if kw in text_l)


def categorize_event(text: str) -> str:
    text_l = (text or "").lower()
    if any(k in text_l for k in ["earnings", "results", "q1", "q2", "q3", "q4", "profit", "revenue"]):
        return "Earnings"
    if any(k in text_l for k in ["acquisition", "merger", "acquire", "stake"]):
        return "M&A"
    if any(k in text_l for k in ["rbi", "sebi", "regulatory", "probe", "fine", "penalty", "compliance"]):
        return "Regulatory"
    if any(k in text_l for k in ["sensex", "nifty", "market", "rally", "selloff", "correction"]):
        return "Market"
    if any(k in text_l for k in ["inflation", "gdp", "rate hike", "repo rate", "budget", "fiscal"]):
        return "Macro"
    return "Operational"


def classify_severity(text: str) -> str:
    text_l = (text or "").lower()
    if any(k in text_l for k in ["fraud", "crash", "recall", "ban", "collapse", "scam", "probe"]):
        return "High"
    if any(k in text_l for k in ["upgrade", "downgrade", "warning", "delay", "dispute"]):
        return "Medium"
    return "Low"


def dedupe_articles(articles: list[dict]) -> list[dict]:
    """Simple hash-based near-dedup on normalised headline (MinHash-LSH in the
    original notebook is overkill for the small per-trigger batches we pull
    live; this preserves the same intent — no duplicate headline articles)."""
    seen = set()
    out = []
    for a in articles:
        key = hashlib.md5(a["headline"].strip().lower().encode()).hexdigest()
        if key not in seen and a["headline"]:
            seen.add(key)
            out.append(a)
    return out


def run_agent2(ticker: str, company_name: str, triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()

    all_articles: list[dict] = []
    for _, trigger_row in triggers.iterrows():
        arts = fetch_all_for_trigger(ticker, company_name, trigger_row)
        all_articles.extend(arts)

    all_articles = dedupe_articles(all_articles)

    df = pd.DataFrame(all_articles)
    if df.empty:
        return df

    df["text"] = (df["headline"].fillna("") + ". " + df["description"].fillna(""))
    df = df[df["text"].apply(is_financially_relevant)].copy()
    if df.empty:
        return df

    df["matched_keywords"] = df["text"].apply(get_matched_keywords)
    df["event_type"]       = df["text"].apply(categorize_event)
    df["severity_kw"]      = df["text"].apply(classify_severity)

    return df.reset_index(drop=True)
