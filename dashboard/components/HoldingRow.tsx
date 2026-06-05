"use client";
import { useState } from "react";
import type { Holding } from "@/lib/types";
import { fmtMoney, fmtPct } from "@/lib/format";

const CAP = 0.10;

export function HoldingRow({ h }: { h: Holding }) {
  const [open, setOpen] = useState(false);
  const w = h.weight_actual ?? 0;
  const noQuote = (h.price ?? 0) === 0 && h.shares > 0;
  const pctOfCap = Math.min(w / CAP, 1) * 100;
  return (
    <div className="rounded-md bg-neutral-900 px-3 py-2 ring-1 ring-neutral-800">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 text-left">
        <span className="w-16 font-medium">{h.ticker}</span>
        <span className="relative h-3 flex-1 overflow-hidden rounded bg-neutral-800">
          <span className="absolute inset-y-0 left-0 bg-emerald-500" style={{ width: `${pctOfCap}%` }} />
        </span>
        <span className="w-14 text-right text-sm tabular-nums">{fmtPct(w)}</span>
        {noQuote && <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-300">no quote</span>}
      </button>
      {open && (
        <div className="mt-2 grid grid-cols-2 gap-1 pl-16 text-xs text-neutral-400">
          <span>Target: {fmtPct(h.weight_target ?? null)}</span>
          <span>Shares: {h.shares}</span>
          <span>Price: {fmtMoney(h.price)}</span>
          <span>Value: {fmtMoney(h.market_value)}</span>
        </div>
      )}
    </div>
  );
}
