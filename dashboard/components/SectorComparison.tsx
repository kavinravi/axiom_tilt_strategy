"use client";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { SP500_SECTORS, SECTOR_COLORS } from "@/lib/sectors";

interface SectorWeight {
  sector: string;
  weight: number;
}

interface Props {
  portfolio: SectorWeight[];
  title?: string;
}

function sectorColor(sector: string): string {
  return SECTOR_COLORS[sector] ?? "#737373";
}

function PieSide({
  data,
  label,
  caption,
}: {
  data: SectorWeight[];
  label: string;
  caption?: string;
}) {
  return (
    <div className="flex flex-col items-center">
      <p className="mb-1 text-xs text-neutral-400">{label}</p>
      {data.length === 0 ? (
        <div className="flex h-52 w-full items-center justify-center text-xs text-neutral-600">
          No data
        </div>
      ) : (
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="weight"
                nameKey="sector"
                cx="50%"
                cy="42%"
                outerRadius={60}
              >
                {data.map((entry) => (
                  <Cell key={entry.sector} fill={sectorColor(entry.sector)} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                contentStyle={{ background: "#171717", border: "1px solid #404040" }}
                itemStyle={{ color: "#a3a3a3" }}
              />
              <Legend
                wrapperStyle={{ fontSize: 10, color: "#a3a3a3", paddingTop: 4 }}
                iconSize={8}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
      {caption && (
        <p className="mt-1 text-center text-[10px] text-neutral-600">{caption}</p>
      )}
    </div>
  );
}

export function SectorComparison({ portfolio, title = "Sector Allocation" }: Props) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-4 text-sm font-medium">{title}</p>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <PieSide data={portfolio} label="Your Portfolio" />
        <PieSide
          data={SP500_SECTORS}
          label="S&P 500 (approx)"
          caption="approximate static reference"
        />
      </div>
    </div>
  );
}
