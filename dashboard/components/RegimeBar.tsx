import { fmtPct } from "@/lib/format";

const SLEEVES = ["10", "20", "30", "50"] as const;
const COLORS: Record<string, string> = { "10": "#60a5fa", "20": "#34d399", "30": "#fbbf24", "50": "#f87171" };

export function RegimeBar({ kProbs, features }: {
  kProbs: Record<string, number> | null; features: Record<string, number> | null;
}) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Regime Call</p>
      {kProbs ? (
        <>
          <div className="flex h-4 w-full overflow-hidden rounded">
            {SLEEVES.map((k) => (
              <div key={k} style={{ width: `${(kProbs[k] ?? 0) * 100}%`, background: COLORS[k] }} title={`k=${k}: ${fmtPct(kProbs[k] ?? 0)}`} />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-400">
            {SLEEVES.map((k) => <span key={k}>k={k}: {fmtPct(kProbs[k] ?? 0)}</span>)}
          </div>
        </>
      ) : <p className="text-sm text-neutral-400">No regime call yet.</p>}
      {features && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
          {Object.entries(features).map(([k, v]) => <span key={k}>{k}: {v}</span>)}
        </div>
      )}
    </div>
  );
}
