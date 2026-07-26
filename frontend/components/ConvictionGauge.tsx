"use client";

export function ConvictionGauge({ value, verdict }: { value: number; verdict?: string }) {
  // value is 0..1. Map to a -90deg..90deg needle sweep across a semicircle.
  const pct = Math.max(0, Math.min(1, value));
  const angle = -90 + pct * 180;

  const zoneColor =
    verdict === "BUY" ? "#22c55e" : verdict === "SELL" ? "#f0554d" : "#f0b429";

  const r = 70;
  const cx = 90;
  const cy = 90;
  const arcPath = (startDeg: number, endDeg: number) => {
    const toXY = (deg: number) => {
      const rad = (deg * Math.PI) / 180;
      return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
    };
    const [x1, y1] = toXY(startDeg);
    const [x2, y2] = toXY(endDeg);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg width="180" height="105" viewBox="0 0 180 105" className="overflow-visible">
        {/* track */}
        <path d={arcPath(180, 360)} stroke="rgba(255,255,255,0.08)" strokeWidth="10" fill="none" strokeLinecap="round" />
        {/* zone segments */}
        <path d={arcPath(180, 240)} stroke="#f0554d" strokeOpacity="0.35" strokeWidth="10" fill="none" strokeLinecap="round" />
        <path d={arcPath(240, 300)} stroke="#f0b429" strokeOpacity="0.35" strokeWidth="10" fill="none" strokeLinecap="round" />
        <path d={arcPath(300, 360)} stroke="#22c55e" strokeOpacity="0.35" strokeWidth="10" fill="none" strokeLinecap="round" />

        {/* needle */}
        <g style={{ transform: `rotate(${angle}deg)`, transformOrigin: `${cx}px ${cy}px`, transition: "transform 0.7s cubic-bezier(0.16,1,0.3,1)" }}>
          <line x1={cx} y1={cy} x2={cx} y2={cy - r + 14} stroke={zoneColor} strokeWidth="3" strokeLinecap="round" />
        </g>
        <circle cx={cx} cy={cy} r="5" fill={zoneColor} />
      </svg>

      <div className="-mt-2 text-center">
        <div className="text-2xl font-display font-bold mono" style={{ color: zoneColor }}>
          {(pct * 100).toFixed(0)}%
        </div>
        <div className="text-[10px] uppercase tracking-wider text-slate-400 mt-0.5">Conviction</div>
      </div>
    </div>
  );
}
