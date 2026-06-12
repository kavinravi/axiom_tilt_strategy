"use client";
import { useState } from "react";
import type { Holding } from "@/lib/types";
import { fmtMoney, fmtPct, fmtSignedMoney } from "@/lib/format";

const CAP = 0.10;

export function HoldingRow({ h }: { h: Holding }) {
  const [open, setOpen] = useState(false);
  const w = h.weight_actual ?? 0;
  const noQuote = (h.price ?? 0) === 0 && h.shares > 0;
  const pctOfCap = Math.min(w / CAP, 1) * 100;
  const dayPnl = h.daily_pnl ?? null;
  const dayTone =
    dayPnl == null ? "text-neutral-500" : dayPnl >= 0 ? "text-emerald-400" : "text-red-400";
  return (
    <div className="rounded-md bg-neutral-900 px-3 py-2 ring-1 ring-neutral-800">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 text-left">
        <span className="flex w-44 min-w-0 flex-col">
          <span className="font-medium">{h.ticker}</span>
          {h.company_name && <span className="truncate text-[11px] text-neutral-500">{h.company_name}</span>}
        </span>
        <span className="relative h-3 flex-1 overflow-hidden rounded bg-neutral-800">
          <span className="absolute inset-y-0 left-0 bg-emerald-500" style={{ width: `${pctOfCap}%` }} />
        </span>
        <span
          className={`w-20 text-right text-xs tabular-nums ${dayTone}`}
          title="Today's profit/loss on this position"
        >
          {fmtSignedMoney(dayPnl)}
        </span>
        <span className="w-14 text-right text-sm tabular-nums">{fmtPct(w)}</span>
        {noQuote && <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-300">no quote</span>}
      </button>
      {open && (
        <div className="mt-2 grid grid-cols-2 gap-1 pl-3 text-xs text-neutral-400">
          <span>Sector: {h.sector ?? "—"}</span>
          <span>Target: {fmtPct(h.weight_target ?? null)}</span>
          <span>Shares: {h.shares}</span>
          <span>Price: {fmtMoney(h.price)}</span>
          <span>Value: {fmtMoney(h.market_value)}</span>
          <span>Avg cost: {fmtMoney(h.avg_cost ?? null)}</span>
          <span className={dayTone}>Today&apos;s P&L: {fmtSignedMoney(dayPnl)}</span>
          <span className={
            (h.unrealized_pnl ?? 0) >= 0 && h.unrealized_pnl != null
              ? "text-emerald-400" : h.unrealized_pnl != null ? "text-red-400" : ""
          }>
            Gain since purchase: {fmtSignedMoney(h.unrealized_pnl ?? null)}
          </span>
        </div>
      )}
    </div>
  );
}
