import { asOfET } from "@/lib/format";
export function AsOf({ iso }: { iso: string | null }) {
  return <span className="text-xs text-neutral-400">{asOfET(iso)}</span>;
}
