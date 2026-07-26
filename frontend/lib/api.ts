const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type TickerInfo = { ticker: string; company: string };

export type AnalysisResult = {
  ticker: string;
  company: string;
  generated_in_sec?: number;
  error?: string;
  message?: string;
  anomaly_summary?: {
    rows_downloaded: number;
    anomalies_detected: number;
    high_severity: number;
    anomaly_rate_pct: number;
  };
  news_summary?: {
    articles_found: number;
    articles_used: number;
    avg_finbert_confidence: number | null;
  };
  forecast?: {
    best_model: string;
    predicted_direction: "UP" | "DOWN";
    predicted_return: number;
    forecast_confidence: number;
    baseline_f1: number;
    hybrid_f1: number;
    f1_lift: number;
    model_comparison: { model: string; baseline_f1: number; hybrid_f1: number }[];
    feature_importance: { feature: string; importance: number }[];
  };
  sentiment?: { daily_sentiment_index: number | null; article_count: number | null };
  verdict?: { verdict: "BUY" | "HOLD" | "SELL"; reason: string; conviction: number; conflict: boolean };
  explainability?: {
    shap: {
      contributions: { feature: string; friendly_name: string; value: number; shap_value: number }[];
      top_driver: { feature: string; friendly_name: string; value: number; shap_value: number };
    };
    granger: {
      note: string;
      causality_confirmed: boolean;
      lags: Record<string, { p_value: number; significant: boolean }>;
      explanation: string;
    };
    reasoning: string;
  };
  recent_headlines?: { headline: string; source: string; sentiment: string; url: string; event_type: string }[];
  _cached?: boolean;
  _cached_at?: string;
};

export async function fetchTickers(): Promise<TickerInfo[]> {
  const res = await fetch(`${API_URL}/api/tickers`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch tickers");
  return res.json();
}

export async function fetchDashboard(): Promise<
  { ticker: string; company: string; verdict: string; conviction: number; created_at: string }[]
> {
  const res = await fetch(`${API_URL}/api/dashboard`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch dashboard");
  return res.json();
}

export async function analyzeTicker(ticker: string, force = false): Promise<AnalysisResult> {
  const res = await fetch(`${API_URL}/api/analyze/${ticker}?force=${force}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Analysis failed");
  }
  return res.json();
}
