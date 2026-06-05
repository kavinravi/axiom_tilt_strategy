import type { WeeklyRow } from "@/lib/types";
import { fmtPct } from "@/lib/format";

function color(w: number): string {
  if (w <= 0) return "#171717";
  const a = Math.min(w / 0.10, 1);
  return `rgba(52, 211, 153, ${0.2 + a * 0.8})`;
}

export function PersistenceHeatmap({ all }: { all: WeeklyRow[] }) {
  const weeks = [...new Set(all.map((r) => r.asof_friday))].sort();
  const tickers = [...new Set(all.map((r) => r.ticker))].sort();
  const wOf = new Map(all.map((r) => [`${r.asof_friday}|${r.ticker}`, r.target_weight]));
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Persistence</p>
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
                    <div className="h-5 w-8 rounded-sm" title={`${t} ${w}: ${fmtPct(wOf.get(`${w}|${t}`) ?? 0)}`} style={{ background: color(wOf.get(`${w}|${t}`) ?? 0) }} />
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
