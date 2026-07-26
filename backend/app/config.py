"""
Fin-Agent India — configuration
All thresholds here are carried over unchanged from the original
Fin-Agent research notebooks (Group 03, NMIMS) — only the universe
of tickers and news sources has been localised to India.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Universe — NSE large-caps (yfinance requires the ".NS" suffix for NSE)
# ---------------------------------------------------------------------------
TICKERS = {
    "RELIANCE.NS":  "Reliance Industries Ltd",
    "TCS.NS":       "Tata Consultancy Services",
    "INFY.NS":      "Infosys Ltd",
    "HDFCBANK.NS":  "HDFC Bank Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "SBIN.NS":      "State Bank of India",
    "TATAMOTORS.NS":"Tata Motors Ltd",
    "BHARTIARTL.NS":"Bharti Airtel Ltd",
    "WIPRO.NS":     "Wipro Ltd",
    "ADANIENT.NS":  "Adani Enterprises Ltd",
}

# ---------------------------------------------------------------------------
# Agent 1 — Anomaly detection (unchanged from original notebook)
# ---------------------------------------------------------------------------
ANOMALY_WINDOW          = 10
PRICE_Z_THRESH          = 1.5
VOLUME_Z_THRESH         = 1.5
PRICE_PERIOD            = "1y"

# ---------------------------------------------------------------------------
# Agent 2 — News (Indian sources, replacing Guardian/global NewsAPI)
# ---------------------------------------------------------------------------
NEWSAPI_KEY   = os.getenv("NEWSAPI_KEY", "")
GNEWS_KEY     = os.getenv("GNEWS_KEY", "")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")

# Domains used to restrict NewsAPI to Indian financial press
INDIAN_NEWS_DOMAINS = ",".join([
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "ndtv.com",
    "financialexpress.com",
])

FIN_KEYWORDS = [
    "earnings", "revenue", "profit", "loss", "stock", "share", "nse", "bse",
    "sensex", "nifty", "rbi", "sebi", "ipo", "acquisition", "merger", "results",
    "guidance", "dividend", "buyback", "rating", "upgrade", "downgrade",
    "q1", "q2", "q3", "q4", "quarterly", "regulatory", "fraud", "probe",
]

# ---------------------------------------------------------------------------
# Model 1 — FinBERT sentiment, via Hugging Face's hosted Inference API
# (no local torch/transformers, no model in server RAM — free to host)
# ---------------------------------------------------------------------------
FINBERT_MODEL       = "ProsusAI/finbert"
HF_API_URL          = f"https://api-inference.huggingface.co/models/{FINBERT_MODEL}"
HF_API_TOKEN        = os.getenv("HF_API_TOKEN", "")   # https://huggingface.co/settings/tokens (free)
HF_BATCH_SIZE       = 16          # texts per HF API call
HF_MAX_RETRIES      = 4           # retries while the (free, shared) model cold-starts
HF_RETRY_WAIT_SEC   = 8
CONFIDENCE_THRESHOLD= 0.60

SEVERITY_WEIGHTS = {"High": 1.5, "Medium": 1.0, "Low": 0.5}

# ---------------------------------------------------------------------------
# Model 2 — Hybrid forecasting
# ---------------------------------------------------------------------------
CV_SPLITS  = 5
SEED       = 42

PRICE_FEATURES = [
    "daily_return", "log_return", "ma_cross_5_20", "price_above_ma20",
    "rolling_std_10", "rolling_std_20", "price_momentum_5", "price_momentum_10",
    "volume_change", "volume_ratio", "high_low_range", "upper_shadow",
    "lower_shadow", "rsi_14", "return_lag1", "return_lag2", "return_lag3",
    "rolling_mean_5r",
]

SENTIMENT_FEATURES = [
    "daily_sentiment_index", "sentiment_ma3", "sentiment_ma7",
    "sentiment_lag1", "sentiment_lag2", "sentiment_momentum",
    "sentiment_trend_dir", "severity_weighted_score", "major_event_flag",
    "pos_neg_ratio", "sentiment_price_cross", "high_news_volume",
]

HYBRID_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES
TARGET = "direction"

# ---------------------------------------------------------------------------
# Model 3 — Verdict thresholds (unchanged from original notebook)
# ---------------------------------------------------------------------------
ALPHA           = 0.001
CONFIDENCE_MIN  = 0.51
SENTIMENT_POS   = 0.0
SENTIMENT_NEG   = 0.0
VOLATILITY_HIGH = 0.06
MIN_HOLD_PERIODS= 2

# ---------------------------------------------------------------------------
# Model 4 — Explainability
# ---------------------------------------------------------------------------
SHAP_FEATURE_COLS = [
    "predicted_return",
    "forecast_confidence",
    "daily_sentiment_index",
    "conviction",
]
GRANGER_MAX_LAG = 3

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DB_PATH        = os.getenv("DB_PATH", "finagent_india.db")
CACHE_TTL_HOURS = 24
