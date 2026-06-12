import type { Holding } from "@/lib/types";
import { fmtMoney, fmtPct } from "@/lib/format";
import { HoldingRow } from "./HoldingRow";

// Weights and cap-bars are measured against the INVESTED book, not total
// account value: deposits sit in cash until the Monday rebalance, and dividing
// by NAV would silently shrink every position (a 10%-cap holding read ~5.6%
// the week a $75k deposit landed). The Cash row keeps the account summing to
// 100% so the dilution is visible instead of hidden.
export function ConcentrationBars({
  holdings,
  nav = null,
  investedPct = null,
}: {
  holdings: Holding[];
  nav?: number | null;
  investedPct?: number | null;
}) {
  const investedFrac = investedPct && investedPct > 0 ? investedPct : 1;
  const cash = nav != null && investedPct != null ? nav * (1 - investedPct) : null;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 px-3 text-xs text-neutral-500">
        <span className="w-16">Ticker</span>
        <span className="flex-1 text-right">Position size</span>
        <span className="w-20 text-right">Today&apos;s P&L</span>
        <span className="w-14 text-right">Weight</span>
      </div>
      <p className="px-3 text-[11px] text-neutral-500">
        Weight = each position&apos;s share of <span className="text-neutral-300">invested money</span>;
        bars fill toward the <span className="text-amber-300">10% per-stock cap</span> (amber = drifted
        above it; trimmed at the next rebalance). Cash is shown as its share of the whole account.
        Tap a row for details.
      </p>
      {holdings.map((h) => <HoldingRow key={h.ticker} h={h} investedFrac={investedFrac} />)}
      {cash != null && cash > 1 && (
        <div className="flex items-center gap-3 rounded-md bg-neutral-900 px-3 py-2 ring-1 ring-neutral-800">
          <span className="flex w-44 min-w-0 flex-col">
            <span className="font-medium">Cash</span>
            <span className="truncate text-[11px] text-neutral-500">awaiting next rebalance</span>
          </span>
          <span className="flex-1 text-right text-xs tabular-nums text-neutral-400">{fmtMoney(cash)}</span>
          <span className="w-20" />
          <span className="w-14 text-right text-sm tabular-nums" title="Share of the whole account">
            {fmtPct(1 - (investedPct ?? 1))}
          </span>
        </div>
      )}
    </div>
  );
}
