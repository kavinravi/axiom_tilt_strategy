// Approximate static S&P 500 GICS-ish sector weights (~2026); reference baseline only, not updated daily.
const SP500_RAW: Record<string, number> = {
  "Technology": 0.31,
  "Financial Services": 0.13,
  "Healthcare": 0.11,
  "Consumer Cyclical": 0.10,
  "Communication Services": 0.09,
  "Industrials": 0.085,
  "Consumer Defensive": 0.06,
  "Energy": 0.035,
  "Utilities": 0.025,
  "Real Estate": 0.025,
  "Basic Materials": 0.02,
};

const SP500_TOTAL = Object.values(SP500_RAW).reduce((a, b) => a + b, 0);

export const SP500_SECTORS: { sector: string; weight: number }[] = Object.entries(SP500_RAW)
  .map(([sector, w]) => ({ sector, weight: w / SP500_TOTAL }))
  .sort((a, b) => b.weight - a.weight);

/** Dark-theme-friendly palette; the same sector gets the same color in both pies. */
export const SECTOR_COLORS: Record<string, string> = {
  "Technology": "#34d399",
  "Financial Services": "#60a5fa",
  "Healthcare": "#f472b6",
  "Consumer Cyclical": "#fb923c",
  "Communication Services": "#a78bfa",
  "Industrials": "#facc15",
  "Consumer Defensive": "#4ade80",
  "Energy": "#f87171",
  "Utilities": "#22d3ee",
  "Real Estate": "#fb7185",
  "Basic Materials": "#a3e635",
  "Unknown": "#737373",
};

/**
 * Aggregate a list of holdings or weekly rows into per-sector weights.
 * - Null / empty sector → bucketed as "Unknown".
 * - Zero or negative weights are skipped.
 * - Result is sorted by weight descending.
 */
export function aggregateSectors(
  items: Array<{ sector?: string | null }>,
  weightKey: string,
): { sector: string; weight: number }[] {
  const map = new Map<string, number>();
  for (const item of items) {
    const w = (item as Record<string, unknown>)[weightKey];
    if (typeof w !== "number" || w <= 0) continue;
    const sector = (item.sector?.trim()) || "Unknown";
    map.set(sector, (map.get(sector) ?? 0) + w);
  }
  return Array.from(map.entries())
    .map(([sector, weight]) => ({ sector, weight }))
    .sort((a, b) => b.weight - a.weight);
}
