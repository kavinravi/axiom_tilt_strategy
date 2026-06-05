import type { Risk } from "@/lib/types";
import { fmtPct } from "@/lib/format";

export function RiskCard({ risk }: { risk: Risk | null }) {
  const rows: [string, string][] = [
    ["Current drawdown", fmtPct(risk?.current_drawdown ?? null)],
    ["Max drawdown", fmtPct(risk?.max_drawdown ?? null)],
    ["Sharpe (to date)", risk?.sharpe != null ? risk.sharpe.toFixed(2) : "—"],
    ["Annualized vol", fmtPct(risk?.ann_vol ?? null)],
  ];
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Risk</p>
      <dl className="grid grid-cols-2 gap-2 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-neutral-400">{k}</dt><dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
