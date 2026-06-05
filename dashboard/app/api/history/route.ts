import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";
import { computeTurnover } from "@/lib/turnover";

export async function GET(req: NextRequest) {
  const ds = await getDataSource(req.nextUrl.searchParams.get("scenario") ?? undefined);
  const fridays = await ds.getWeeklyFridays();
  const selected = req.nextUrl.searchParams.get("friday") ?? fridays[0] ?? null;
  const [weekly, allWeekly, executions] = await Promise.all([
    selected ? ds.getWeeklyPortfolio(selected) : Promise.resolve([]),
    ds.getAllWeekly(),
    selected ? ds.getExecutions(selected) : Promise.resolve([]),
  ]);
  const turnover = computeTurnover(allWeekly, fridays, selected);
  return NextResponse.json({ fridays, selected, weekly, allWeekly, executions, turnover });
}
