"use client";

import { useEffect, useState } from "react";
import { analyzeTicker, fetchTickers, AnalysisResult, TickerInfo } from "@/lib/api";
import { VerdictBadge, SentimentTag, GrangerBadge } from "@/components/Badges";
import { F1Chart } from "@/components/F1Chart";
import { ConvictionGauge } from "@/components/ConvictionGauge";

export default function Home() {
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [selected, setSelected] = useState<string>("RELIANCE.NS");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTickers()
      .then(setTickers)
      .catch(() => setError("Couldn't reach the API. Is the backend running / NEXT_PUBLIC_API_URL set?"));
  }, []);

  async function runAnalysis(ticker: string, force = false) {
    setLoading(true);
    setError(null);
    setSelected(ticker);
    try {
      const data = await analyzeTicker(ticker, force);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-6 py-10 max-w-7xl mx-auto">
      <Header />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6 mt-10">
        {/* Ticker list */}
        <aside className="card p-4 h-fit lg:sticky lg:top-8">
          <h2 className="text-[11px] uppercase tracking-[0.15em] text-slate-500 mb-3 font-semibold px-1">
            NSE Watchlist
          </h2>
          <div className="flex flex-col gap-1">
            {(tickers.length ? tickers : DEFAULT_TICKERS).map((t) => (
              <button
                key={t.ticker}
                onClick={() => runAnalysis(t.ticker)}
                className={`text-left px-3 py-2.5 rounded-xl transition-all border ${
                  selected === t.ticker
                    ? "bg-accent/12 border-accent/40 text-white"
                    : "border-transparent hover:bg-white/[0.04] text-slate-300"
                }`}
              >
                <div className="text-sm font-display font-semibold tracking-tight">
                  {t.ticker.replace(".NS", "")}
                </div>
                <div className="text-[11px] text-slate-500 truncate">{t.company}</div>
              </button>
            ))}
          </div>
        </aside>

        {/* Results */}
        <section className="min-h-[400px]">
          {error && <div className="card p-4 border-sell/30 text-sell text-sm mb-4">{error}</div>}

          {loading && <LoadingState />}

          {!loading && !result && !error && (
            <div className="card p-14 text-center text-slate-400">
              <div className="text-3xl mb-3 opacity-40">◐</div>
              Select a stock from the watchlist to run the live 6-agent analysis.
            </div>
          )}

          {!loading && result && !result.error && (
            <ResultView result={result} onRefresh={() => runAnalysis(result.ticker, true)} />
          )}

          {!loading && result?.error && (
            <div className="card p-8 text-center text-slate-300">
              <p className="font-display font-semibold mb-1">Couldn&apos;t generate a signal yet</p>
              <p className="text-sm text-slate-400">{result.message}</p>
            </div>
          )}
        </section>
      </div>

      <Footer />
    </main>
  );
}

function Header() {
  return (
    <header className="flex flex-col md:flex-row md:items-center justify-between gap-6">
      <div>
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-accent2 font-semibold mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-accent2 animate-pulse" />
          NSE · Live Multi-Agent Pipeline
        </div>
        <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight">
          Fin-<span className="text-accent2">Agent</span> India
        </h1>
        <p className="text-slate-400 text-sm mt-3 max-w-xl leading-relaxed">
          Anomaly detection + FinBERT sentiment + hybrid ML forecasting, explained with SHAP and validated
          with Granger causality. Confidence-scored reasoning — not a guaranteed prediction.
        </p>
      </div>
      <div className="flex flex-wrap gap-2 text-[11px] text-slate-400">
        <Pill>6-agent pipeline</Pill>
        <Pill>FinBERT NLP</Pill>
        <Pill>SHAP + Granger</Pill>
      </div>
    </header>
  );
}

function Pill({ children }: { children: React.ReactNode }) {
  return <span className="card px-3 py-1.5 whitespace-nowrap font-medium">{children}</span>;
}

function LoadingState() {
  return (
    <div className="card p-14 text-center">
      <div className="inline-flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
        <p className="text-sm text-slate-300 font-medium font-display">Running live pipeline…</p>
        <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
          Fetching NSE price data → detecting anomalies → pulling Indian financial news →
          scoring sentiment with FinBERT → forecasting → computing SHAP + Granger causality.
          This can take 20–40 seconds.
        </p>
      </div>
    </div>
  );
}

function ResultView({ result, onRefresh }: { result: AnalysisResult; onRefresh: () => void }) {
  const { verdict, forecast, sentiment, explainability, anomaly_summary, news_summary, recent_headlines } = result;

  return (
    <div className="flex flex-col gap-5 animate-fade-slide-up">
      {/* Verdict header */}
      <div className="card card-glow p-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          <div className="flex-1">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <div className="text-xs text-slate-500 mb-1">{result.company}</div>
                <div className="flex items-center gap-3">
                  <h2 className="text-3xl font-display font-bold tracking-tight">{result.ticker.replace(".NS", "")}</h2>
                  {verdict && <VerdictBadge verdict={verdict.verdict} />}
                </div>
                {result._cached && (
                  <span className="text-[10px] text-slate-500 mono mt-1 inline-block">
                    cached · {result._cached_at?.slice(0, 16)} UTC
                  </span>
                )}
              </div>
              <button
                onClick={onRefresh}
                className="text-xs px-3 py-2 rounded-lg border border-accent/30 text-accent2 hover:bg-accent/10 transition-colors shrink-0"
              >
                ↻ Re-run live analysis
              </button>
            </div>

            {explainability?.reasoning && (
              <p className="text-sm text-slate-300 mt-5 leading-relaxed border-t border-white/10 pt-4">
                {explainability.reasoning}
              </p>
            )}

            {explainability?.granger && (
              <div className="mt-3">
                <GrangerBadge confirmed={explainability.granger.causality_confirmed} />
              </div>
            )}
          </div>

          {verdict && (
            <div className="shrink-0 self-center md:self-start md:pt-2">
              <ConvictionGauge value={verdict.conviction} verdict={verdict.verdict} />
            </div>
          )}
        </div>
      </div>

      {/* Stat strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Anomalies flagged" value={anomaly_summary?.anomalies_detected ?? "—"} sub={`${anomaly_summary?.anomaly_rate_pct ?? "—"}% of days`} accent="accent2" />
        <Stat label="News articles used" value={news_summary?.articles_used ?? "—"} sub={`of ${news_summary?.articles_found ?? 0} found`} accent="accent2" />
        <Stat label="Forecast direction" value={forecast?.predicted_direction ?? "—"} sub={`${((forecast?.forecast_confidence ?? 0) * 100).toFixed(1)}% confidence`} accent={forecast?.predicted_direction === "UP" ? "buy" : "sell"} />
        <Stat label="Sentiment index" value={sentiment?.daily_sentiment_index?.toFixed(3) ?? "—"} sub={`${sentiment?.article_count ?? 0} articles`} accent="hold" />
      </div>

      {/* Model comparison */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
          <h3 className="text-sm font-display font-semibold text-slate-200">Hybrid vs price-only baseline (F1, TimeSeriesSplit CV)</h3>
          {forecast && (
            <span className="text-xs text-accent2 mono">+{forecast.f1_lift.toFixed(3)} F1 lift · best: {forecast.best_model}</span>
          )}
        </div>
        <F1Chart data={forecast?.model_comparison || []} />
      </div>

      {/* SHAP drivers */}
      {explainability?.shap && (
        <div className="card p-5">
          <h3 className="text-sm font-display font-semibold text-slate-200 mb-4">SHAP feature attribution for this verdict</h3>
          <div className="flex flex-col gap-3">
            {explainability.shap.contributions.map((c) => (
              <div key={c.feature} className="flex items-center gap-3">
                <div className="w-40 text-xs text-slate-400 shrink-0">{c.friendly_name}</div>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden relative">
                  <div
                    className={`absolute top-0 h-full rounded-full ${c.shap_value >= 0 ? "bg-buy left-1/2" : "bg-sell right-1/2"}`}
                    style={{ width: `${Math.min(Math.abs(c.shap_value) * 120, 50)}%` }}
                  />
                  <div className="absolute left-1/2 top-0 w-px h-full bg-white/20" />
                </div>
                <div className="w-16 text-right text-xs mono text-slate-300">{c.shap_value.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Headlines */}
      {recent_headlines && recent_headlines.length > 0 && (
        <div className="card p-5">
          <h3 className="text-sm font-display font-semibold text-slate-200 mb-3">Supporting headlines</h3>
          <div className="flex flex-col divide-y divide-white/5">
            {recent_headlines.map((h, i) => (
              <a
                key={i}
                href={h.url}
                target="_blank"
                rel="noreferrer"
                className="py-3 flex items-start justify-between gap-4 hover:bg-white/[0.04] -mx-2 px-2 rounded-lg transition-colors"
              >
                <div>
                  <div className="text-sm text-slate-200">{h.headline}</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">{h.source} · {h.event_type}</div>
                </div>
                <SentimentTag sentiment={h.sentiment} />
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, sub, accent = "accent2" }: { label: string; value: string | number; sub?: string; accent?: string }) {
  const borderColor: Record<string, string> = {
    accent2: "before:bg-accent2",
    buy: "before:bg-buy",
    sell: "before:bg-sell",
    hold: "before:bg-hold",
  };
  return (
    <div className={`card p-4 relative overflow-hidden before:absolute before:left-0 before:top-0 before:h-full before:w-[3px] ${borderColor[accent] || borderColor.accent2}`}>
      <div className="text-[11px] text-slate-500 mb-1">{label}</div>
      <div className="text-xl font-display font-bold mono">{value}</div>
      {sub && <div className="text-[11px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-14 pb-6 text-center text-[11px] text-slate-500 leading-relaxed">
      Fin-Agent India generates confidence-scored, explainable signals for research and educational purposes —
      it is not investment advice, and past reasoning is not a guarantee of future accuracy.
      <br />
      Built on a 6-agent pipeline: anomaly detection · Indian news retrieval · FinBERT sentiment ·
      hybrid ML forecasting · rule-based verdict · SHAP + Granger explainability.
    </footer>
  );
}

const DEFAULT_TICKERS: TickerInfo[] = [
  { ticker: "RELIANCE.NS", company: "Reliance Industries Ltd" },
  { ticker: "TCS.NS", company: "Tata Consultancy Services" },
  { ticker: "INFY.NS", company: "Infosys Ltd" },
  { ticker: "HDFCBANK.NS", company: "HDFC Bank Ltd" },
  { ticker: "ICICIBANK.NS", company: "ICICI Bank Ltd" },
  { ticker: "SBIN.NS", company: "State Bank of India" },
  { ticker: "TATAMOTORS.NS", company: "Tata Motors Ltd" },
  { ticker: "BHARTIARTL.NS", company: "Bharti Airtel Ltd" },
  { ticker: "WIPRO.NS", company: "Wipro Ltd" },
  { ticker: "ADANIENT.NS", company: "Adani Enterprises Ltd" },
];
