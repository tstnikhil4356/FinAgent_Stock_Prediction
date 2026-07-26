"""Tiny SQLite cache so repeat visits don't re-run the full live pipeline."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timedelta

from . import config

_conn = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                ticker TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def get_cached(ticker: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT payload, created_at FROM results WHERE ticker = ?", (ticker,)).fetchone()
    if not row:
        return None
    payload, created_at = row
    if datetime.fromisoformat(created_at) < datetime.utcnow() - timedelta(hours=config.CACHE_TTL_HOURS):
        return None
    data = json.loads(payload)
    data["_cached"] = True
    data["_cached_at"] = created_at
    return data


def set_cached(ticker: str, payload: dict) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO results (ticker, payload, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(ticker) DO UPDATE SET payload = excluded.payload, created_at = excluded.created_at",
        (ticker, json.dumps(payload, default=str), datetime.utcnow().isoformat()),
    )
    conn.commit()


def list_cached_summaries() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT ticker, payload, created_at FROM results").fetchall()
    out = []
    for ticker, payload, created_at in rows:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        out.append({
            "ticker": ticker,
            "company": data.get("company"),
            "verdict": (data.get("verdict") or {}).get("verdict"),
            "conviction": (data.get("verdict") or {}).get("conviction"),
            "created_at": created_at,
        })
    return out
