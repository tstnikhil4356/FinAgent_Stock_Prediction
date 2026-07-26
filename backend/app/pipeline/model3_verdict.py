"""
Model 3 — Decision verdict (BUY / HOLD / SELL)
Ported verbatim from 05_Model_3.ipynb — same 4-layer threshold logic:
  1. Risk filter (high volatility + low confidence -> HOLD)
  2. Conflict detection (model direction vs strong opposing sentiment -> HOLD)
  3. Core threshold logic (direction + confidence -> BUY/SELL)
  4. Weighted conviction score (confidence x sentiment boost)
"""
from __future__ import annotations
from .. import config


def compute_verdict(row: dict) -> dict:
    pred_return = float(row.get("predicted_return", 0))
    confidence  = float(row.get("forecast_confidence", 0.5))
    sentiment   = float(row.get("daily_sentiment_index", 0))
    direction   = int(row.get("predicted_direction", 0))
    volatility  = abs(float(row.get("daily_return", 0)))

    # Layer 1 — risk filter
    if volatility > config.VOLATILITY_HIGH and confidence < 0.55:
        return {"verdict": "HOLD", "reason": "high_volatility_risk", "conviction": 0.3, "conflict": False}

    # Layer 2 — conflict detection
    conflict = False
    if abs(sentiment) > 0.01:
        if direction == 1 and sentiment < -0.15:
            conflict = True
        if direction == 0 and sentiment > 0.15:
            conflict = True
    if conflict:
        return {"verdict": "HOLD", "reason": "conflicting_signals", "conviction": 0.4, "conflict": True}

    # Layer 3 — core threshold logic (sentiment = conviction booster, not a gate)
    sentiment_boost = 1 + abs(sentiment) * 0.5

    if direction == 1 and confidence > config.CONFIDENCE_MIN:
        conviction = round(min(confidence * sentiment_boost, 1.0), 3)
        return {"verdict": "BUY", "reason": "threshold_met", "conviction": conviction, "conflict": False}
    elif direction == 0 and confidence > config.CONFIDENCE_MIN:
        conviction = round(min(confidence * sentiment_boost, 1.0), 3)
        return {"verdict": "SELL", "reason": "threshold_met", "conviction": conviction, "conflict": False}
    else:
        return {"verdict": "HOLD", "reason": "neutral_band", "conviction": 0.5, "conflict": False}


def run_model3(forecast_result: dict, latest_sentiment_row: dict | None) -> dict:
    row = dict(forecast_result.get("latest_row", {}))
    row["predicted_direction"] = forecast_result.get("predicted_direction")
    row["predicted_return"] = forecast_result.get("predicted_return")
    row["forecast_confidence"] = forecast_result.get("forecast_confidence")
    if latest_sentiment_row:
        row["daily_sentiment_index"] = latest_sentiment_row.get("daily_sentiment_index", row.get("daily_sentiment_index", 0))

    return compute_verdict(row)
