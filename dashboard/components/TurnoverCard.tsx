import type { Turnover } from "@/lib/types";
import { fmtPct } from "@/lib/format";

export function TurnoverCard({ turnover }: { turnover: Turnover | null }) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Turnover vs prior week</p>
      {turnover ? (
        <div className="space-y-1 text-sm">
          <p>One-way turnover: <span className="tabular-nums">{fmtPct(turnover.turnover_frac)}</span></p>
          <p className="text-emerald-400">Added: {turnover.added.join(", ") || "—"}</p>
          <p className="text-red-400">Dropped: {turnover.dropped.join(", ") || "—"}</p>
        </div>
      ) : <p className="text-sm text-neutral-400">Turnover needs at least two weeks of history.</p>}
    </div>
  );
}
