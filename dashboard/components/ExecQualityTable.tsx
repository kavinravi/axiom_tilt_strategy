import type { Execution } from "@/lib/types";
import { fmtBps, fmtMoney } from "@/lib/format";

export function ExecQualityTable({ executions }: { executions: Execution[] }) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Execution quality</p>
      {executions.length ? (
        <table className="w-full text-sm">
          <thead><tr className="text-left text-neutral-500">
            <th>Ticker</th><th>Side</th><th className="text-right">Fill</th><th className="text-right">Mid</th><th className="text-right">Slippage</th>
          </tr></thead>
          <tbody>
            {executions.map((e) => (
              <tr key={`${e.ticker}-${e.side}`} className="border-t border-neutral-800">
                <td>{e.ticker}</td><td>{e.side}</td>
                <td className="text-right">{fmtMoney(e.realized_price)}</td>
                <td className="text-right">{fmtMoney(e.midpoint)}</td>
                <td className={`text-right ${(e.slippage_bps ?? 0) > 0 ? "text-red-400" : "text-emerald-400"}`}>{fmtBps(e.slippage_bps)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="text-sm text-neutral-400">No fills for this week yet.</p>}
    </div>
  );
}
