export function VerdictBadge({ verdict }: { verdict: string }) {
  const styles: Record<string, string> = {
    BUY: "bg-buy/15 text-buy border-buy/30",
    SELL: "bg-sell/15 text-sell border-sell/30",
    HOLD: "bg-hold/15 text-hold border-hold/30",
  };
  const icon: Record<string, string> = { BUY: "▲", SELL: "▼", HOLD: "■" };
  const cls = styles[verdict] || styles.HOLD;

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-display font-semibold tracking-wide ${cls}`}>
      <span className="text-[10px]">{icon[verdict] || "■"}</span>
      {verdict}
    </span>
  );
}

export function SentimentTag({ sentiment }: { sentiment?: string }) {
  const s = (sentiment || "NEUTRAL").toUpperCase();
  const styles: Record<string, string> = {
    POSITIVE: "bg-buy/15 text-buy",
    NEGATIVE: "bg-sell/15 text-sell",
    NEUTRAL: "bg-white/5 text-slate-400",
  };
  return (
    <span className={`shrink-0 text-[10px] font-medium px-2 py-1 rounded-md ${styles[s] || styles.NEUTRAL}`}>
      {s}
    </span>
  );
}

export function GrangerBadge({ confirmed }: { confirmed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-slate-400">
      <span className={`w-1.5 h-1.5 rounded-full ${confirmed ? "bg-buy" : "bg-slate-500"}`} />
      Granger {confirmed ? "confirmed" : "not confirmed"}
    </span>
  );
}
