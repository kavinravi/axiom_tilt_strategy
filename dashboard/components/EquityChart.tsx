"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { twrReturns } from "@/lib/twr";
import type { EquityPoint } from "@/lib/types";

export function EquityChart({ points }: { points: EquityPoint[] }) {
  // Time-weighted: growth of invested capital, not of account balance —
  // deposits/withdrawals (flow column) contribute zero.
  const strategy = twrReturns(points);
  // SPY baselines on the first trading day's close: point 0 is the pre-trading
  // cost-basis anchor (capital frozen Friday, deployed Monday), and crediting
  // SPY with the Fri→Mon move we sat out in cash would skew the comparison.
  const spyBaseIdx = points.findIndex((p, i) => i > 0 && p.spy_close != null);
  const spyBase = spyBaseIdx > 0 ? points[spyBaseIdx].spy_close : null;
  const data = points.map((p, i) => ({
    date: p.date,
    Strategy: strategy[i],
    SPY: spyBase && p.spy_close && i >= spyBaseIdx ? p.spy_close / spyBase - 1 : null,
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
