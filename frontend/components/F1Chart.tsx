"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

type ModelRow = {
  model: string;
  baseline_f1: number;
  hybrid_f1: number;
  baseline_acc: number;
  hybrid_acc: number;
};

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="card p-3 text-xs">
      <div className="font-display font-semibold text-slate-200 mb-1.5">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2 mono">
          <span className="w-2 h-2 rounded-full" style={{ background: p.fill }} />
          <span className="text-slate-400">{p.name}:</span>
          <span className="text-slate-100">{p.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

export function F1Chart({ data }: { data: ModelRow[] }) {
  if (!data || data.length === 0) {
    return <div className="h-64 flex items-center justify-center text-sm text-slate-500">No model comparison data yet.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={6}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="model" tick={{ fill: "#8890b5", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} tickLine={false} />
        <YAxis domain={[0, 1]} tick={{ fill: "#8890b5", fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
        <Bar dataKey="baseline_f1" name="Price-only baseline" fill="#3a4166" radius={[4, 4, 0, 0]} maxBarSize={36} />
        <Bar dataKey="hybrid_f1" name="Hybrid (price + sentiment)" fill="#5b6ef5" radius={[4, 4, 0, 0]} maxBarSize={36} />
      </BarChart>
    </ResponsiveContainer>
  );
}
