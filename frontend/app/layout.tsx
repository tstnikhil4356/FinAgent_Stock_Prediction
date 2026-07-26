import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fin-Agent India | Explainable AI Trading Signals",
  description:
    "Multi-agent financial forecasting for NSE stocks — anomaly detection, FinBERT sentiment, hybrid ML forecasting, and SHAP + Granger explainability.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
