import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";

export async function GET(req: NextRequest) {
  const scenario = req.nextUrl.searchParams.get("scenario") ?? undefined;
  const ds = await getDataSource(scenario);
  const [holdings, snapshot] = await Promise.all([ds.getHoldings(), ds.getSnapshot()]);
  return NextResponse.json({ holdings, asof: snapshot?.asof ?? null });
}
