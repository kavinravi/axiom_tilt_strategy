import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";

export async function GET(req: NextRequest) {
  const scenario = req.nextUrl.searchParams.get("scenario") ?? undefined;
  const ds = await getDataSource(scenario);
  const [snapshot, equityCurve] = await Promise.all([ds.getSnapshot(), ds.getEquityCurve()]);
  return NextResponse.json({ snapshot, equityCurve });
}
