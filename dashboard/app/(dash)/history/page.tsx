"use client";
import { Suspense, useState } from "react";
import { usePolling } from "@/lib/usePolling";
import { useSearchParams } from "next/navigation";
import type { Execution, Turnover, WeeklyRow } from "@/lib/types";
import { fmtPct } from "@/lib/format";
import { WeekPicker } from "@/components/WeekPicker";
import { PersistenceHeatmap } from "@/components/PersistenceHeatmap";
import { TurnoverCard } from "@/components/TurnoverCard";
import { ExecQualityTable } from "@/components/ExecQualityTable";
import { Empty } from "@/components/Empty";

type Payload = {
  fridays: string[]; selected: string | null;
  weekly: WeeklyRow[]; allWeekly: WeeklyRow[]; executions: Execution[]; turnover: Turnover | null;
};

function HistoryInner() {
  const scenario = useSearchParams().get("scenario");
  const [friday, setFriday] = useState<string | null>(null);
  const q = new URLSearchParams();
  if (scenario) q.set("scenario", scenario);
  if (friday) q.set("friday", friday);
  const url = `/api/history${q.toString() ? `?${q}` : ""}`;
  const { data, error, loading } = usePolling<Payload>(url);

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  if (error && !data)
    return (
      <div className="rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
        Couldn't load live data — {error}
      </div>
    );
  if (!data || data.fridays.length === 0) return <Empty title="No weekly portfolios yet" />;

  return (
    <div className="space-y-4">
      <WeekPicker fridays={data.fridays} selected={friday ?? data.selected} onPick={setFriday} />
      <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
        <p className="mb-2 text-sm font-medium">Target portfolio — {data.selected}</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-neutral-500">
              <th>Ticker</th><th>Company</th><th>Sector</th><th className="text-right">Weight</th>
            </tr>
          </thead>
          <tbody>
            {data.weekly.map((r) => (
              <tr key={r.ticker} className="border-t border-neutral-800">
                <td className="font-medium">{r.ticker}</td>
                <td className="text-neutral-300">{r.company_name ?? "—"}</td>
                <td className="text-neutral-400">{r.sector ?? "—"}</td>
                <td className="text-right tabular-nums">{fmtPct(r.target_weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <TurnoverCard turnover={data.turnover} />
      <PersistenceHeatmap all={data.allWeekly} />
      <ExecQualityTable executions={data.executions} />
    </div>
  );
}

export default function HistoryPage() {
  return (
    <Suspense fallback={<p className="text-sm text-neutral-400">Loading…</p>}>
      <HistoryInner />
    </Suspense>
  );
}
