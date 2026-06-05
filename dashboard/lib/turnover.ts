import type { Turnover, WeeklyRow } from "./types";

export function computeTurnover(
  all: WeeklyRow[], fridaysDesc: string[], selected: string | null,
): Turnover | null {
  if (!selected) return null;
  const idx = fridaysDesc.indexOf(selected);
  if (idx === -1) return null;
  const prior = fridaysDesc[idx + 1];
  if (!prior) return null;
  const wOf = (f: string) =>
    new Map(all.filter((r) => r.asof_friday === f).map((r) => [r.ticker, r.target_weight]));
  const cur = wOf(selected);
  const prev = wOf(prior);
  const tickers = new Set([...cur.keys(), ...prev.keys()]);
  let turnover = 0;
  for (const t of tickers) turnover += Math.abs((cur.get(t) ?? 0) - (prev.get(t) ?? 0));
  return {
    added: [...cur.keys()].filter((t) => !prev.has(t)).sort(),
    dropped: [...prev.keys()].filter((t) => !cur.has(t)).sort(),
    turnover_frac: 0.5 * turnover,
  };
}
