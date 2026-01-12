"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TestPredictionRow } from "@/lib/types";

function formatShortDate(value: string) {
  return value.slice(2);
}

export default function PriceChart({
  rows,
  showSma50 = false,
  showSma200 = false,
}: {
  rows: TestPredictionRow[];
  showSma50?: boolean;
  showSma200?: boolean;
}) {
  const chartData = rows.map((r) => ({
    date: r.date,
    close: r.close,
    buy: r.pred === 1 ? r.close : null,
    sma50: r.sma50,
    sma200: r.sma200,
  }));

  return (
    <div className="h-96 w-full rounded-xl border border-zinc-800 bg-zinc-950 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={formatShortDate}
            minTickGap={24}
            stroke="#a1a1aa"
          />
          <YAxis stroke="#a1a1aa" domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#09090b", border: "1px solid #27272a" }}
            labelStyle={{ color: "#e4e4e7" }}
          />
          {/* Main Price Line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#60a5fa" // blue-400
            dot={false}
            strokeWidth={2}
            name="Close Price"
          />

          {/* AI Buy Signal Scatter */}
          <Scatter dataKey="buy" fill="#22c55e" name="Buy Signal" />

          {/* Indicators */}
          {showSma50 && (
            <Line
              type="monotone"
              dataKey="sma50"
              stroke="#f97316" // orange-500
              dot={false}
              strokeWidth={1.5}
              name="SMA 50"
            />
          )}
          {showSma200 && (
            <Line
              type="monotone"
              dataKey="sma200"
              stroke="#a855f7" // purple-500
              dot={false}
              strokeWidth={1.5}
              name="SMA 200"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
