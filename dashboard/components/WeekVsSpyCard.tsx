import type { WeekVsSpy } from "@/lib/types";
import { fmtSignedPct } from "@/lib/format";

// Trading-week-to-date scoreboard: strategy vs SPY since the prior week's
// last close (Wednesday shows Mon-Wed, Friday the full week).
export function WeekVsSpyCard({ week }: { week: WeekVsSpy | null | undefined }) {
  if (!week || week.portfolio_return === null) return null;
  const excess = week.excess_return;
  const tone =
    excess == null ? "text-neutral-100" : excess >= 0 ? "text-emerald-400" : "text-red-400";
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <div className="flex items-baseline justify-between">
        <p className="text-xs uppercase tracking-wide text-neutral-400">This Week vs SPY</p>
        <p className="text-[11px] text-neutral-500">since {week.baseline_date} close</p>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-lg font-semibold tabular-nums text-neutral-100">
            {fmtSignedPct(week.portfolio_return)}
          </p>
          <p className="text-[11px] text-neutral-400">Strategy</p>
        </div>
        <div>
          <p className="text-lg font-semibold tabular-nums text-neutral-300">
            {fmtSignedPct(week.spy_return)}
          </p>
          <p className="text-[11px] text-neutral-400">SPY</p>
        </div>
        <div>
          <p className={`text-lg font-semibold tabular-nums ${tone}`}>
            {fmtSignedPct(excess)}
          </p>
          <p className="text-[11px] text-neutral-400">Excess</p>
        </div>
      </div>
    </div>
  );
}
