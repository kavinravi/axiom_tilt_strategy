import type { WeeklyRow } from "@/lib/types";
import { fmtPct } from "@/lib/format";

// Discrete weight bins instead of a continuous gradient: distinct lightness
// steps AND alternating stripe texture, so every adjacent pair of bins differs
// in two ways and stays tellable apart at a glance.
function stripes(base: string, stripe: string): string {
  return `repeating-linear-gradient(45deg, ${base} 0px, ${base} 4px, ${stripe} 4px, ${stripe} 8px)`;
}

const BINS: { min: number; label: string; bg: string }[] = [
  { min: 0.075, label: "≥7.5%", bg: stripes("#a7f3d0", "#34d399") },
  { min: 0.05, label: "5–7.5%", bg: "#34d399" },
  { min: 0.025, label: "2.5–5%", bg: stripes("#059669", "#065f46") },
  { min: 1e-9, label: "<2.5%", bg: "#065f46" },
  { min: -Infinity, label: "not held", bg: "#171717" },
];

function binOf(w: number) {
  return BINS.find((b) => w >= b.min) ?? BINS[BINS.length - 1];
}

export function PersistenceHeatmap({ all }: { all: WeeklyRow[] }) {
  const weeks = [...new Set(all.map((r) => r.asof_friday))].sort();
  const tickers = [...new Set(all.map((r) => r.ticker))].sort();
  const wOf = new Map(all.map((r) => [`${r.asof_friday}|${r.ticker}`, r.target_weight]));
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium">Persistence</p>
        <div className="flex flex-wrap items-center gap-3 text-[10px] text-neutral-400">
          <span>Target weight:</span>
          {[...BINS].reverse().map((b) => (
            <span key={b.label} className="flex items-center gap-1">
              <span
                className="inline-block h-3 w-4 rounded-sm ring-1 ring-neutral-700"
                style={{ background: b.bg }}
              />
              {b.label}
            </span>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs">
          <thead>
            <tr><th className="text-left text-neutral-500"></th>
              {weeks.map((w) => <th key={w} className="px-1 text-neutral-500">{w.slice(5)}</th>)}</tr>
          </thead>
          <tbody>
            {tickers.map((t) => (
              <tr key={t}>
                <td className="pr-2 text-right text-neutral-300">{t}</td>
                {weeks.map((w) => (
                  <td key={w}>
                    <div className="h-5 w-8 rounded-sm" title={`${t} ${w}: ${fmtPct(wOf.get(`${w}|${t}`) ?? 0)}`} style={{ background: binOf(wOf.get(`${w}|${t}`) ?? 0).bg }} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
