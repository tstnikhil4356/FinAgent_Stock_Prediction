# Fin-Agent India

A hosted, explainable BUY/HOLD/SELL signal engine for NSE large-caps — the
web-app version of the Fin-Agent research project (NMIMS, M.Sc. Data
Science, Group 03), localised to Indian stocks and Indian financial news.

> **This is a research/portfolio tool, not investment advice.** It does not
> and cannot predict prices with certainty — it gives a confidence-scored,
> explainable *reason* (SHAP-attributed feature drivers + Granger-tested
> causal link between news sentiment and price) behind each signal, exactly
> like the original notebook pipeline did.

---

## What it does — mapped to the resume bullet

> *"Built a 4-agent AI pipeline combining technical anomaly detection
> (Z-score, RSI, Bollinger Bands, EMA) with FinBERT NLP sentiment analysis
> across 6 news APIs for 5 stocks; hybrid model achieved +0.089 F1-score
> improvement over price-only baseline with anomaly days showing 3.5×
> higher absolute returns. Integrated SHAP & LIME explainability with
> Granger causality validation to generate transparent BUY/HOLD/SELL
> signals with confidence thresholds and supporting news headlines."*

| Resume claim | Where it lives in this repo |
|---|---|
| Z-score / RSI / Bollinger / EMA anomaly detection | `backend/app/pipeline/agent1_anomaly.py` |
| FinBERT sentiment across news APIs | `backend/app/pipeline/model1_sentiment.py` (FinBERT) + `agent2_news.py` (NewsAPI, GNews, Marketaux — Indian sources) |
| Hybrid model vs price-only baseline, F1 lift | `backend/app/pipeline/model2_forecast.py` — reports `f1_lift` live for whichever stock you analyze, same TimeSeriesSplit CV methodology as the notebook |
| Anomaly days → higher absolute returns | `backend/app/pipeline/agent1_anomaly.py` `summary` block + this is the same Mann-Whitney-testable claim as your H1 hypothesis in the slide deck |
| SHAP explainability | `backend/app/pipeline/model4_explain.py` (`compute_shap_attribution`) |
| Granger causality validation | `model4_explain.py` (`run_granger_test`) |
| Transparent BUY/HOLD/SELL with confidence thresholds | `backend/app/pipeline/model3_verdict.py` (verbatim threshold logic from your notebook: α=0.001, confidence_min=0.51, volatility_high=0.06) |
| Supporting news headlines | Returned in every `/api/analyze/{ticker}` response as `recent_headlines`, shown in the UI under each verdict |

**Note on LIME:** the resume bullet mentions LIME as well as SHAP. Your
original notebook computed SHAP for global/verdict-level attribution;
LIME wasn't in the final `06_ML4_Explanation.ipynb` code I found. I've
implemented SHAP faithfully. If you want LIME added too (e.g. per-article
local explanations on the FinBERT sentiment calls), say so and I'll wire
it in — happy to keep the resume line 100% accurate either way, or you can
soften the bullet to "SHAP explainability" if you'd rather not add it.

---

## Architecture

```
Next.js (Vercel)  ──HTTP──►  FastAPI (Render, free tier)
     UI dashboard              │
                                ├─ Agent 1: yfinance OHLCV → Z-score/RSI/Bollinger/EMA anomaly flags
                                ├─ Agent 2: NewsAPI + GNews + Marketaux (Indian sources) in ±2hr trigger window
                                ├─ Model 1: FinBERT sentiment via Hugging Face Inference API (no local model)
                                ├─ Model 2: RF / GBM / LogReg / XGBoost, TimeSeriesSplit CV, hybrid vs baseline F1
                                ├─ Model 3: 4-layer threshold verdict logic (BUY/HOLD/SELL + conviction)
                                └─ Model 4: SHAP attribution + Granger causality + NLG reasoning
                                          │
                                    SQLite cache (24h TTL per ticker)
```

**FinBERT hosting note:** Model 1 no longer loads FinBERT locally. It
calls Hugging Face's hosted Inference API (`api-inference.huggingface.co`)
over HTTP, so the backend has zero ML weights in memory and no
torch/transformers dependency — `pip install` is fast and the whole thing
fits comfortably on Render's **free** tier. Trade-off: the HF free-tier
model is shared infrastructure, so occasionally the first call after a
period of inactivity gets a ~10-20s "model is loading" cold start (handled
automatically via `wait_for_model: true` + retries in
`model1_sentiment.py`). If `HF_API_TOKEN` isn't set at all, sentiment
scoring falls back to a small keyword lexicon so the app still runs
end-to-end with zero configured keys — every response is tagged with
`source: "finbert_hf_api"` or `source: "lexicon_fallback"` so you always
know which one produced a given score.

Every `/api/analyze/{ticker}` call runs the **real, live pipeline** —
nothing is mocked. Results are cached per ticker for 24h so the dashboard
loads instantly on repeat visits; a "↻ Re-run live analysis" button forces
a fresh run (~20–40s, since it's re-downloading price data, hitting 3 news
APIs, and running FinBERT inference).

---

## Stock & news universe (India)

10 NSE large-caps: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN,
TATAMOTORS, BHARTIARTL, WIPRO, ADANIENT (`backend/app/config.py` —
add/remove tickers there, must be valid yfinance NSE symbols, i.e. end in
`.NS`).

News sources: NewsAPI (restricted to moneycontrol.com,
economictimes.indiatimes.com, livemint.com, business-standard.com,
ndtv.com, financialexpress.com via the `domains` filter), GNews
(`country=in`), and Marketaux as a symbol-based fallback.

---

## Local development

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # fill in NEWSAPI_KEY / GNEWS_KEY / MARKETAUX_KEY
uvicorn app.main:app --reload --port 8000
```
First request will take longer (FinBERT downloads on first boot, ~440MB).

Get free API keys:
- NewsAPI: https://newsapi.org/register (free tier: 100 req/day, last 30 days)
- GNews: https://gnews.io/register (free tier: 100 req/day)
- Marketaux: https://www.marketaux.com/register (optional, free tier available)
- Hugging Face: https://huggingface.co/settings/tokens (free account, "Read" token — powers FinBERT sentiment)

### Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```
Open http://localhost:3000.

---

## Deploying (same stack as your InterviewAI India project)

### Backend → Render
1. Push this repo to GitHub.
2. New → Web Service on Render, point at `/backend`.
3. Render will read `render.yaml` (Blueprint) or you can set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables: `NEWSAPI_KEY`, `GNEWS_KEY`, `MARKETAUX_KEY`, `HF_API_TOKEN`.
5. That's it — **Render's free tier is enough.** FinBERT runs via Hugging
   Face's hosted Inference API rather than locally, so the backend has no
   ML weights in memory (see "FinBERT hosting note" above). The only
   caveat: Render's free tier spins the service down after ~15 min of
   inactivity, so the first request after a quiet period takes ~30-50s to
   wake up (cold start) on top of the pipeline's normal ~20-40s run time.
   If that first-load delay bothers you for a live demo/interview, the
   $7/mo Starter tier removes the spin-down — otherwise free is genuinely
   fine for a portfolio project.
6. (Optional) Add a Render Cron Job that hits `POST /api/refresh-all` once
   a day so the dashboard always has fresh cached data even before anyone
   visits, and doubles as a keep-alive ping against the free-tier spin-down.

### Frontend → Vercel
1. Import the repo, set root directory to `/frontend`.
2. Environment variable: `NEXT_PUBLIC_API_URL=https://<your-render-app>.onrender.com`
3. Deploy. Vercel auto-detects Next.js.

### Total hosting cost: $0/month
Render free tier (backend) + Vercel free tier (frontend) + Hugging Face
free Inference API (FinBERT) + NewsAPI/GNews/Marketaux free tiers — the
whole stack runs on free plans. The only reason to ever pay is to remove
Render's cold-start delay before a live interview demo.

---

## Known limitations (carried over honestly from your original slide deck)

- **Free-tier news APIs** cover ~30 days back — sparse coverage for
  anomalies older than that. Same mitigation as your original: multiple
  API fallbacks.
- **Thin per-ticker history** on a freshly-added stock means Model 2's CV
  and Model 4's Granger test may report "insufficient data" until enough
  anomaly-triggered news accumulates — the UI states this plainly instead
  of faking a number.
- **SHAP is on a surrogate model**, since Model 3 is rule-based threshold
  logic, not a learned classifier — same caveat as your original notebook.
- **Not real-time execution** — this is a decision-support signal, not a
  trading bot. No brokerage integration.

---

## Repo structure
```
finagent-india/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + endpoints
│   │   ├── config.py          # tickers, thresholds, API keys
│   │   ├── db.py               # SQLite cache
│   │   └── pipeline/
│   │       ├── agent1_anomaly.py
│   │       ├── agent2_news.py
│   │       ├── model1_sentiment.py
│   │       ├── model2_forecast.py
│   │       ├── model3_verdict.py
│   │       ├── model4_explain.py
│   │       └── orchestrator.py
│   ├── requirements.txt
│   ├── render.yaml
│   └── Procfile
└── frontend/
    ├── app/                    # Next.js App Router
    ├── components/
    ├── lib/api.ts
    └── package.json
```
