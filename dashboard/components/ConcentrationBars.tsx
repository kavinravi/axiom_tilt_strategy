import type { Holding } from "@/lib/types";
import { HoldingRow } from "./HoldingRow";

export function ConcentrationBars({ holdings }: { holdings: Holding[] }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 px-3 text-xs text-neutral-500">
        <span className="w-16">Ticker</span>
        <span className="flex-1 text-right">10% cap →</span>
        <span className="w-14 text-right">Weight</span>
      </div>
      <p className="px-3 text-[11px] text-neutral-500">Bars fill toward the <span className="text-amber-300">10% cap</span>.</p>
      {holdings.map((h) => <HoldingRow key={h.ticker} h={h} />)}
    </div>
  );
}
