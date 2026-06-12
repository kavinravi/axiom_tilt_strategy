import type { EquityPoint } from "@/lib/types";

// Cumulative time-weighted return per curve point (0 at the first point).
// Day return = (nav_t - flow_t) / nav_{t-1} - 1, so external cash (deposits,
// withdrawals) compounds at exactly zero — mirrors trading/publish/metrics.py
// twr_index. A non-positive prior NAV makes the day unmeasurable: carry flat.
export function twrReturns(points: EquityPoint[]): number[] {
  const out: number[] = [];
  let index = 1;
  for (let i = 0; i < points.length; i++) {
    if (i > 0) {
      const prevNav = points[i - 1].nav;
      if (prevNav > 0) index *= (points[i].nav - (points[i].flow ?? 0)) / prevNav;
    }
    out.push(index - 1);
  }
  return out;
}
