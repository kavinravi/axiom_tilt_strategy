export function StatCard({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: "up" | "down" | "flat";
}) {
  const toneClass = tone === "up" ? "text-emerald-400" : tone === "down" ? "text-red-400" : "text-neutral-100";
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="text-xs uppercase tracking-wide text-neutral-400">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${toneClass}`}>{value}</p>
      {sub && <p className="mt-0.5 text-sm text-neutral-400">{sub}</p>}
    </div>
  );
}
