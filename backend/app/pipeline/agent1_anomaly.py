"""
Agent 1 — Anomaly Detection
Ported directly from the original Fin-Agent notebook (01_Agent1.ipynb).
Logic is untouched: rolling Z-score on returns + volume, RSI/Bollinger
context, volatility-adjusted dynamic threshold, 2hr news window trigger.
Only the ticker universe changed (NSE via yfinance .NS suffix).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf

from .. import config


def fetch_stock_data(ticker: str, period: str = config.PRICE_PERIOD) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def add_technical_indicators(df: pd.DataFrame, window: int = config.ANOMALY_WINDOW) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["daily_return"] = close.pct_change()
    df["rolling_mean"] = close.rolling(window).mean()
    df["rolling_std"]  = close.rolling(window).std()
    df["volume_mean"]  = volume.rolling(window).mean()

    df["price_zscore"] = (
        df["daily_return"] - df["daily_return"].rolling(window).mean()
    ) / df["daily_return"].rolling(window).std()

    df["volume_zscore"] = (volume - df["volume_mean"]) / volume.rolling(window).std()
    df["volatility"] = df["rolling_std"] / df["rolling_mean"]

    # RSI(14) — used downstream as a hybrid-model feature too
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger %B
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std) + 1e-9)

    df.dropna(inplace=True)
    return df


def detect_anomalies(
    df: pd.DataFrame,
    price_z_thresh: float = config.PRICE_Z_THRESH,
    volume_z_thresh: float = config.VOLUME_Z_THRESH,
) -> pd.DataFrame:
    df = df.copy()
    df["price_anomaly"]  = df["price_zscore"].abs() > price_z_thresh
    df["volume_anomaly"] = df["volume_zscore"] > volume_z_thresh
    df["anomaly_flag"]   = df["price_anomaly"] | df["volume_anomaly"]

    def classify(row):
        if row["price_anomaly"] and row["volume_anomaly"]:
            return "both"
        elif row["price_anomaly"]:
            return "price"
        elif row["volume_anomaly"]:
            return "volume"
        return None

    df["anomaly_type"] = df.apply(classify, axis=1)
    return df


def volatility_adjusted_detection(df: pd.DataFrame, base_thresh: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    vol_mean = df["volatility"].mean()
    vol_std  = df["volatility"].std()
    df["dynamic_thresh"]   = base_thresh + ((df["volatility"] - vol_mean) / (vol_std + 1e-9)).clip(-1, 1)
    df["adj_anomaly_flag"] = df["price_zscore"].abs() > df["dynamic_thresh"]
    return df


def generate_triggers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """NSE trades 09:15–15:30 IST — trigger time anchored to market close."""
    anomalies = df[df["anomaly_flag"] == True].copy()  # noqa: E712
    if anomalies.empty:
        return anomalies

    anomalies["ticker"] = ticker
    anomalies["trigger_date"] = anomalies.index.date
    anomalies["trigger_time"] = pd.to_datetime(
        anomalies["trigger_date"].astype(str) + " 15:30:00"
    )
    anomalies["news_window_start"] = anomalies["trigger_time"] - pd.Timedelta(hours=2)
    anomalies["news_window_end"]   = anomalies["trigger_time"] + pd.Timedelta(hours=2)
    anomalies["severity"] = anomalies["price_zscore"].abs().apply(
        lambda z: "High" if z > 3 else "Medium"
    )

    cols = [
        "ticker", "trigger_date", "trigger_time", "anomaly_type",
        "price_zscore", "volume_zscore", "daily_return",
        "news_window_start", "news_window_end", "severity",
        "rsi_14", "bb_pct",
    ]
    return anomalies[cols].reset_index(drop=True)


def run_agent1(ticker: str) -> dict:
    raw_df      = fetch_stock_data(ticker)
    enriched_df = add_technical_indicators(raw_df)
    detected_df = detect_anomalies(enriched_df)
    detected_df = volatility_adjusted_detection(detected_df)
    triggers    = generate_triggers(detected_df, ticker)

    return {
        "raw_df": raw_df,
        "enriched_df": enriched_df,
        "detected_df": detected_df,
        "triggers": triggers,
        "summary": {
            "rows_downloaded": len(raw_df),
            "anomalies_detected": len(triggers),
            "high_severity": int((triggers["severity"] == "High").sum()) if not triggers.empty else 0,
            "anomaly_rate_pct": round(100 * len(triggers) / max(len(enriched_df), 1), 2),
        },
    }
