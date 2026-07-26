"""
Model 2 — Hybrid forecasting (price-only baseline vs price+sentiment hybrid)
Ported from 04_model2_Forecasting.ipynb. Trains RandomForest / GradientBoosting
/ LogisticRegression / XGBoost with TimeSeriesSplit CV, exactly as in the
original notebook, then reports the same F1 lift metric quoted on the resume
("+0.089 F1 improvement over price-only baseline").
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from .. import config


def merge_price_sentiment(price_df: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.copy()
    price_df["price_impact_day"] = price_df.index.date
    price_df["price_impact_day"] = pd.to_datetime(price_df["price_impact_day"])

    if daily_sentiment.empty:
        merged = price_df.copy()
        for col in config.SENTIMENT_FEATURES:
            merged[col] = 0.0
        return merged

    daily_sentiment = daily_sentiment.copy()
    daily_sentiment["price_impact_day"] = pd.to_datetime(daily_sentiment["price_impact_day"])
    merged = pd.merge(price_df, daily_sentiment, on="price_impact_day", how="left")

    for col in config.SENTIMENT_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["log_return"] = np.log(close / close.shift(1))
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma_cross_5_20"] = (df["ma5"] > df["ma20"]).astype(int)
    df["price_above_ma20"] = (close > df["ma20"]).astype(int)
    df["rolling_std_10"] = df["daily_return"].rolling(10).std()
    df["rolling_std_20"] = df["daily_return"].rolling(20).std()
    df["price_momentum_5"] = close.pct_change(5)
    df["price_momentum_10"] = close.pct_change(10)
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ratio"] = df["Volume"] / (df["Volume"].rolling(10).mean() + 1e-9)
    df["high_low_range"] = (df["High"] - df["Low"]) / close
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / close
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / close
    df["return_lag1"] = df["daily_return"].shift(1)
    df["return_lag2"] = df["daily_return"].shift(2)
    df["return_lag3"] = df["daily_return"].shift(3)
    df["rolling_mean_5r"] = df["daily_return"].rolling(5).mean()

    df["sentiment_price_cross"] = (
        (df.get("daily_sentiment_index", 0) > 0) & (df["daily_return"] > 0)
    ).astype(int)

    # target: next-day direction (1 = up, 0 = down/flat)
    df["direction"] = (close.shift(-1) > close).astype(int)
    df["price_impact_day"] = df["price_impact_day"]

    df.dropna(subset=["rsi_14", "rolling_std_20"], inplace=True)
    return df


def _get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=config.SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=config.SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=config.SEED),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, eval_metric="logloss", random_state=config.SEED, verbosity=0, n_jobs=1),
    }


def run_timeseries_cv(X, y, model, n_splits=config.CV_SPLITS) -> dict:
    n_splits = min(n_splits, max(2, len(X) // 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], preds))
        f1s.append(f1_score(y.iloc[test_idx], preds, zero_division=0))
    return {
        "accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        "f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
    }


def train_and_forecast(featured_df: pd.DataFrame) -> dict:
    price_feats = [f for f in config.PRICE_FEATURES if f in featured_df.columns]
    sent_feats  = [f for f in config.SENTIMENT_FEATURES if f in featured_df.columns]
    hybrid_feats = price_feats + sent_feats

    X_price  = featured_df[price_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_hybrid = featured_df[hybrid_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = featured_df[config.TARGET]

    if len(featured_df) < 15 or y.nunique() < 2:
        # not enough history for CV — fall back to a simple heuristic
        last = featured_df.iloc[-1]
        pred_dir = 1 if last.get("daily_return", 0) >= 0 else 0
        return {
            "model_comparison": [],
            "best_model": "Random Forest (heuristic fallback — insufficient history)",
            "predicted_direction": pred_dir,
            "predicted_return": float(last.get("daily_return", 0)),
            "forecast_confidence": 0.55,
            "baseline_f1": 0.0,
            "hybrid_f1": 0.0,
            "f1_lift": 0.0,
            "feature_importance": [],
        }

    models = _get_models()
    comparison = []
    best_f1 = -1
    best_name = None
    for name, model in models.items():
        base = run_timeseries_cv(X_price, y, model)
        hyb  = run_timeseries_cv(X_hybrid, y, model)
        comparison.append({"model": name, "baseline_f1": base["f1"], "hybrid_f1": hyb["f1"],
                            "baseline_acc": base["accuracy"], "hybrid_acc": hyb["accuracy"]})
        if hyb["f1"] > best_f1:
            best_f1 = hyb["f1"]
            best_name = name

    best_model = models[best_name]
    best_model.fit(X_hybrid, y)

    last_row = X_hybrid.iloc[[-1]]
    pred_dir = int(best_model.predict(last_row)[0])
    proba = best_model.predict_proba(last_row)[0]
    confidence = float(proba[pred_dir])

    predicted_return = float(featured_df["daily_return"].tail(10).mean() * (1 if pred_dir == 1 else -1))

    feat_importance = []
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        feat_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(hybrid_feats, imp)],
            key=lambda x: -x["importance"],
        )[:8]

    baseline_f1 = next((c["baseline_f1"] for c in comparison if c["model"] == best_name), 0.0)

    return {
        "model_comparison": comparison,
        "best_model": best_name,
        "predicted_direction": pred_dir,
        "predicted_return": round(predicted_return, 5),
        "forecast_confidence": round(confidence, 4),
        "baseline_f1": baseline_f1,
        "hybrid_f1": best_f1,
        "f1_lift": round(best_f1 - baseline_f1, 4),
        "feature_importance": feat_importance,
    }


def run_model2(price_df: pd.DataFrame, daily_sentiment: pd.DataFrame) -> dict:
    merged = merge_price_sentiment(price_df, daily_sentiment)
    featured = engineer_features(merged)
    if featured.empty:
        return {"error": "insufficient_data"}
    result = train_and_forecast(featured)
    result["latest_row"] = featured.iloc[-1].to_dict()
    result["featured_df"] = featured
    return result
