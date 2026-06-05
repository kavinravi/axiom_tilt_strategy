"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { EquityPoint } from "@/lib/types";

export function EquityChart({ points }: { points: EquityPoint[] }) {
  const base = points[0];
  const data = points.map((p) => ({
    date: p.date,
    Strategy: base && base.nav ? p.nav / base.nav - 1 : 0,
    SPY: base?.spy_close && p.spy_close ? p.spy_close / base.spy_close - 1 : null,
  }));
  return (
    <div className="h-56 w-full rounded-lg bg-neutral-900 p-3 ring-1 ring-neutral-800">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="date" tick={{ fill: "#a3a3a3", fontSize: 11 }} minTickGap={24} />
          <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: "#a3a3a3", fontSize: 11 }} width={36} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
            contentStyle={{ background: "#171717", border: "1px solid #404040" }} />
          <Line type="monotone" dataKey="Strategy" stroke="#34d399" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="SPY" stroke="#a3a3a3" dot={false} strokeWidth={1.5} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
