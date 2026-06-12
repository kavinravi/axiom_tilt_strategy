"use client";
import { Suspense } from "react";
import { usePolling } from "@/lib/usePolling";
import { useSearchParams } from "next/navigation";
import type { Holding } from "@/lib/types";
import { ConcentrationBars } from "@/components/ConcentrationBars";
import { Empty } from "@/components/Empty";
import { AsOf } from "@/components/AsOf";
import { SectorComparison } from "@/components/SectorComparison";
import { aggregateSectors } from "@/lib/sectors";

type Payload = {
  holdings: Holding[];
  asof: string | null;
  nav: number | null;
  invested_pct: number | null;
};

function HoldingsInner() {
  const scenario = useSearchParams().get("scenario");
  const url = scenario ? `/api/holdings?scenario=${scenario}` : "/api/holdings";
  const { data, error, loading } = usePolling<Payload>(url);

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  if (error && !data)
    return (
      <div className="rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
        Couldn't load live data — {error}
      </div>
    );
  const holdings = data?.holdings ?? [];

  if (holdings.length === 0) {
    return <Empty title="No live positions yet"
                  hint="The target portfolio is visible under History until the strategy trades." />;
  }
  return (
    <div className="space-y-3">
      <div className="flex justify-end"><AsOf iso={data?.asof ?? null} /></div>
      <ConcentrationBars
        holdings={holdings}
        nav={data?.nav ?? null}
        investedPct={data?.invested_pct ?? null}
      />
      <SectorComparison
        portfolio={aggregateSectors(holdings, "weight_actual")}
        title="Sector Allocation"
      />
    </div>
  );
}

export default function HoldingsPage() {
  return (
    <Suspense fallback={<p className="text-sm text-neutral-400">Loading…</p>}>
      <HoldingsInner />
    </Suspense>
  );
}
