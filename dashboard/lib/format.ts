const DASH = "—";

export function fmtMoney(n: number | null | undefined): string {
  if (n === null || n === undefined) return DASH;
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" });
}
export function fmtSignedMoney(n: number | null | undefined): string {
  if (n === null || n === undefined) return DASH;
  const s = Math.abs(n).toLocaleString("en-US", { style: "currency", currency: "USD" });
  return `${n >= 0 ? "+" : "-"}${s}`;
}
export function fmtPct(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined) return DASH;
  return `${(n * 100).toFixed(digits)}%`;
}
export function fmtSignedPct(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined) return DASH;
  const s = (n * 100).toFixed(digits);
  return `${n >= 0 ? "+" : ""}${s}%`;
}
export function fmtBps(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined) return DASH;
  return `${n.toFixed(digits)} bps`;
}
export function asOfET(iso: string | null | undefined): string {
  if (!iso) return "no data yet";
  const t = new Date(iso).toLocaleString("en-US", {
    timeZone: "America/New_York", hour: "numeric", minute: "2-digit", hour12: true,
  });
  return `as of ${t} ET`;
}
