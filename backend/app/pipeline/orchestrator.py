"""
Orchestrator — runs the full 6-stage Fin-Agent pipeline for one ticker:
Agent1 (anomaly) -> Agent2 (news) -> Model1 (FinBERT sentiment) ->
Model2 (hybrid forecast) -> Model3 (verdict) -> Model4 (SHAP + Granger + NLG)
"""
from __future__ import annotations
import time
import pandas as pd

from .. import config
from . import agent1_anomaly, agent2_news, model1_sentiment, model2_forecast, model3_verdict, model4_explain


def run_full_pipeline(ticker: str) -> dict:
    t0 = time.time()
    company_name = config.TICKERS.get(ticker, ticker)

    # Agent 1
    a1 = agent1_anomaly.run_agent1(ticker)
    triggers = a1["triggers"]

    # Agent 2
    news_df = agent2_news.run_agent2(ticker, company_name, triggers)

    # Model 1
    article_df, daily_sentiment = model1_sentiment.run_model1(news_df)

    # Model 2
    forecast = forecast = model2_forecast.run_model2(a1["enriched_df"], daily_sentiment)
    if "error" in forecast:
        return {
            "ticker": ticker, "company": company_name, "error": forecast["error"],
            "message": "Not enough price history to generate a forecast for this ticker right now.",
        }

    latest_sentiment_row = None
    if not daily_sentiment.empty:
        latest_sentiment_row = daily_sentiment.sort_values("price_impact_day").iloc[-1].to_dict()

    # Model 3
    verdict = model3_verdict.run_model3(forecast, latest_sentiment_row)

    # Model 4
    recent_articles = article_df.sort_values("published_at", ascending=False).head(5).to_dict("records") if not article_df.empty else []
    explain = model4_explain.run_model4(verdict, forecast, latest_sentiment_row, forecast["featured_df"], recent_articles)

    elapsed = round(time.time() - t0, 2)

    return {
        "ticker": ticker,
        "company": company_name,
        "generated_in_sec": elapsed,
        "anomaly_summary": a1["summary"],
        "news_summary": {
            "articles_found": len(news_df) if news_df is not None else 0,
            "articles_used": len(article_df) if not article_df.empty else 0,
            "avg_finbert_confidence": round(float(article_df["confidence"].mean()), 3) if not article_df.empty else None,
        },
        "forecast": {
            "best_model": forecast["best_model"],
            "predicted_direction": "UP" if forecast["predicted_direction"] == 1 else "DOWN",
            "predicted_return": forecast["predicted_return"],
            "forecast_confidence": forecast["forecast_confidence"],
            "baseline_f1": forecast["baseline_f1"],
            "hybrid_f1": forecast["hybrid_f1"],
            "f1_lift": forecast["f1_lift"],
            "model_comparison": forecast["model_comparison"],
            "feature_importance": forecast["feature_importance"],
        },
        "sentiment": {
            "daily_sentiment_index": (latest_sentiment_row or {}).get("daily_sentiment_index"),
            "article_count": (latest_sentiment_row or {}).get("article_count"),
        },
        "verdict": verdict,
        "explainability": explain,
        "recent_headlines": [
            {"headline": a.get("headline"), "source": a.get("source"), "sentiment": a.get("sentiment"),
             "url": a.get("url"), "event_type": a.get("event_type")}
            for a in recent_articles
        ],
    }
