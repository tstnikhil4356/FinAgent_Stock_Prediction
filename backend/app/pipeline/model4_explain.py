"""
Model 4 — Explainability (SHAP + Granger causality + NLG reasoning)
Ported from 06_ML4_Explanation.ipynb. SHAP runs on a small surrogate
RandomForest trained on the 4 verdict-driving features (predicted_return,
forecast_confidence, daily_sentiment_index, conviction) — same surrogate
approach as the original notebook, which is a known/documented limitation
(see LIMITATIONS in README) since Model 3 itself is rule-based, not a
learned classifier.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

from .. import config

FEATURE_FRIENDLY_NAMES = {
    "predicted_return": "predicted return",
    "forecast_confidence": "model confidence",
    "daily_sentiment_index": "news sentiment",
    "conviction": "conviction score",
}


def run_granger_test(sentiment_series: pd.Series, return_series: pd.Series, max_lag: int = config.GRANGER_MAX_LAG) -> dict:
    data = pd.DataFrame({"returns": return_series.values, "sentiment": sentiment_series.values}).dropna()
    data = data[data["sentiment"] != 0]

    if len(data) < 10:
        return {
            "note": "insufficient_data",
            "causality_confirmed": False,
            "lags": {},
            "explanation": (
                "Not enough non-zero sentiment days yet to run Granger causality reliably "
                "(needs 10+). This is expected on a freshly-triggered ticker with sparse "
                "news coverage — the test becomes reliable as more anomaly-triggered "
                "articles accumulate for this stock."
            ),
        }

    lags = {}
    try:
        max_lag_actual = max(1, min(max_lag, len(data) // 5))
        test_result = grangercausalitytests(data[["returns", "sentiment"]], maxlag=max_lag_actual, verbose=False)
        for lag in range(1, max_lag_actual + 1):
            p_val = test_result[lag][0]["ssr_ftest"][1]
            lags[f"lag_{lag}"] = {"p_value": round(float(p_val), 4), "significant": bool(p_val < 0.05)}
    except Exception as e:
        return {"note": "error", "causality_confirmed": False, "lags": {}, "error": str(e)}

    confirmed = any(v["significant"] for v in lags.values())
    return {
        "note": "ok",
        "causality_confirmed": confirmed,
        "lags": lags,
        "explanation": (
            "Sentiment statistically precedes price movement for this stock (p < 0.05) — "
            "the news reaction is a leading, not coincidental, signal."
            if confirmed else
            "No statistically significant causal link detected between sentiment and price "
            "for this stock in the current window — treat the sentiment feature as "
            "corroborating context rather than a leading indicator here."
        ),
    }


def compute_shap_attribution(verdict_row: dict) -> dict:
    """Trains a tiny surrogate RF on synthetic neighbourhood points around the
    current feature vector so SHAP has something to attribute against, then
    reports per-feature contribution direction/magnitude for this verdict."""
    feats = {k: float(verdict_row.get(k, 0.0)) for k in config.SHAP_FEATURE_COLS}
    X_point = np.array([[feats[k] for k in config.SHAP_FEATURE_COLS]])

    rng = np.random.default_rng(config.SEED)
    noise = rng.normal(0, 0.15, size=(200, len(config.SHAP_FEATURE_COLS)))
    X_synth = X_point + noise
    X_synth = np.vstack([X_synth, X_point])

    # crude but consistent proxy label: BUY-leaning vs SELL-leaning
    y_synth = (X_synth[:, config.SHAP_FEATURE_COLS.index("predicted_return")] > 0).astype(int)
    if y_synth.sum() in (0, len(y_synth)):
        y_synth[0] = 1 - y_synth[0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_synth)

    model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=config.SEED, n_jobs=1)
    model.fit(X_scaled, y_synth)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled[-1:])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.array(shap_vals).flatten()

    contributions = [
        {"feature": f, "friendly_name": FEATURE_FRIENDLY_NAMES.get(f, f),
         "value": round(feats[f], 4), "shap_value": round(float(v), 4)}
        for f, v in zip(config.SHAP_FEATURE_COLS, shap_vals)
    ]
    contributions.sort(key=lambda c: -abs(c["shap_value"]))
    top = contributions[0]

    return {"contributions": contributions, "top_driver": top}


def generate_reasoning(verdict: dict, forecast: dict, sentiment_row: dict | None,
                        shap_result: dict, granger_result: dict, articles: list[dict]) -> str:
    v = verdict["verdict"]
    conviction = verdict["conviction"]
    top = shap_result["top_driver"]
    conf_pct = round(forecast.get("forecast_confidence", 0.5) * 100, 1)

    example_headline = articles[0]["headline"] if articles else None

    parts = [
        f"[{v}] Primary driver: {top['friendly_name']} "
        f"(value {top['value']}, SHAP contribution {top['shap_value']:+.3f})."
    ]
    parts.append(f"Model confidence: {conf_pct}%. Conviction score: {conviction}.")

    if sentiment_row and sentiment_row.get("article_count"):
        parts.append(
            f"Based on {int(sentiment_row['article_count'])} article(s) around the anomaly window"
            + (f", e.g. '{example_headline}'." if example_headline else ".")
        )

    if verdict["conflict"]:
        parts.append("Note: the forecasting model and news sentiment disagree on direction — "
                      "verdict was dampened to HOLD as a precaution.")

    parts.append(granger_result.get("explanation", ""))
    parts.append(
        "This is a reasoned, confidence-scored signal, not a guaranteed prediction — "
        "use it as one input alongside your own research."
    )
    return " ".join(p for p in parts if p)


def run_model4(verdict: dict, forecast: dict, sentiment_row: dict | None,
               featured_df: pd.DataFrame, articles: list[dict]) -> dict:
    row = {
        "predicted_return": forecast.get("predicted_return", 0),
        "forecast_confidence": forecast.get("forecast_confidence", 0.5),
        "daily_sentiment_index": (sentiment_row or {}).get("daily_sentiment_index", 0),
        "conviction": verdict.get("conviction", 0.5),
    }
    shap_result = compute_shap_attribution(row)

    if "daily_sentiment_index" in featured_df.columns:
        granger_result = run_granger_test(featured_df["daily_sentiment_index"], featured_df["daily_return"])
    else:
        granger_result = {"note": "no_sentiment_column", "causality_confirmed": False, "lags": {},
                           "explanation": "No sentiment history available yet for this ticker."}

    reasoning = generate_reasoning(verdict, forecast, sentiment_row, shap_result, granger_result, articles)

    return {"shap": shap_result, "granger": granger_result, "reasoning": reasoning}
