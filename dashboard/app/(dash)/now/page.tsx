"use client";
import { Suspense, useState } from "react";
import { usePolling } from "@/lib/usePolling";
import type { EquityPoint, Snapshot } from "@/lib/types";
import { fmtMoney, fmtPct, fmtSignedPct } from "@/lib/format";
import { useSearchParams } from "next/navigation";
import { StatCard } from "@/components/StatCard";
import { EquityChart } from "@/components/EquityChart";
import { RiskCard } from "@/components/RiskCard";
import { RegimeBar } from "@/components/RegimeBar";
import { Empty } from "@/components/Empty";
import { AsOf } from "@/components/AsOf";

const RANGES = ["1W", "1M", "3M", "All"] as const;
type Range = (typeof RANGES)[number];

function sliceRange(curve: EquityPoint[], range: Range): EquityPoint[] {
  if (range === "All" || curve.length === 0) return curve;
  const days = range === "1W" ? 7 : range === "1M" ? 31 : 93;
  const last = new Date(curve[curve.length - 1].date).getTime();
  const cutoff = last - days * 86_400_000;
  const windowed = curve.filter((p) => new Date(p.date).getTime() >= cutoff);
  return windowed.length > 1 ? windowed : curve; // never collapse to <2 points
}

type Payload = { snapshot: Snapshot | null; equityCurve: EquityPoint[] };

function NowInner() {
  const scenario = useSearchParams().get("scenario");
  const url = scenario ? `/api/now?scenario=${scenario}` : "/api/now";
  const { data, error, loading } = usePolling<Payload>(url);
  const [range, setRange] = useState<Range>("All");

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  if (error && !data)
    return (
      <div className="rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
        Couldn't load live data — {error}
      </div>
    );
  const s = data?.snapshot ?? null;
  const curve = data?.equityCurve ?? [];

  if (!s) {
    return (
      <div className="space-y-4">
        <Empty title="Performance builds forward from go-live"
               hint="No live snapshot yet. The target portfolio is visible under History." />
        <RegimeBar kProbs={null} features={null} />
      </div>
    );
  }

  const dayTone = (s.day_pnl ?? 0) > 0 ? "up" : (s.day_pnl ?? 0) < 0 ? "down" : "flat";
  const totTone: "up" | "down" | undefined =
    s.total_return != null && s.spy_return != null
      ? (s.total_return >= s.spy_return ? "up" : "down")
      : undefined;

  return (
    <div className="space-y-4">
      <div className="flex justify-end"><AsOf iso={s.asof} /></div>
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Portfolio Value" value={fmtMoney(s.nav)} sub={`${s.n_positions ?? 0} positions`} />
        <StatCard label="Today" value={fmtMoney(s.day_pnl)} sub={fmtSignedPct(s.day_pnl_pct)} tone={dayTone} />
        <StatCard label="Total Return" value={fmtPct(s.total_return)} sub={`SPY ${fmtPct(s.spy_return)}`} tone={totTone} />
        <StatCard label="Invested" value={fmtPct(s.invested_pct)} />
      </div>
      {curve.length > 1 ? (
        <div className="space-y-2">
          <div className="flex justify-end gap-1">
            {RANGES.map((r) => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className={`rounded px-2 py-0.5 text-xs ${
                  range === r ? "bg-neutral-100 text-neutral-900" : "bg-neutral-800 text-neutral-300"
                }`}
              >
                {r}
              </button>
            ))}
          </div>
          <EquityChart points={sliceRange(curve, range)} />
        </div>
      ) : (
        <Empty title="Equity curve builds forward from go-live" />
      )}
      <RiskCard risk={s.risk} />
      <RegimeBar kProbs={s.k_probs} features={s.regime_features} />
    </div>
  );
}

export default function NowPage() {
  return (
    <Suspense fallback={<p className="text-sm text-neutral-400">Loading…</p>}>
      <NowInner />
    </Suspense>
  );
}
