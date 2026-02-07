import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";

type Egx30DayRow = {
  date: string;
  open?: number | string | null;
  high?: number | string | null;
  low?: number | string | null;
  close?: number | string | null;
};

function toNum(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : typeof v === "number" ? v : null;
  if (n === null) return null;
  if (!Number.isFinite(n)) return null;
  return n;
}

function monthKey(dateStr: string) {
  // expects YYYY-MM-DD
  return dateStr.slice(0, 7);
}

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "..", "symbols_data", "EGX30-INDEX.json");
    const raw = await fs.readFile(filePath, "utf8");
    const rows = JSON.parse(raw) as Egx30DayRow[];

    const grouped = new Map<
      string,
      {
        start_date: string;
        end_date: string;
        open: number;
        high: number;
        low: number;
        close: number;
      }
    >();

    const sorted = (Array.isArray(rows) ? rows : [])
      .filter((r) => typeof r?.date === "string" && r.date.length >= 10)
      .slice()
      .sort((a, b) => (a.date > b.date ? 1 : -1));

    for (const r of sorted) {
      const o = toNum(r.open) ?? toNum(r.close);
      const h = toNum(r.high) ?? toNum(r.close);
      const l = toNum(r.low) ?? toNum(r.close);
      const c = toNum(r.close);
      if (o === null || h === null || l === null || c === null) continue;

      const key = monthKey(r.date);
      const existing = grouped.get(key);
      if (!existing) {
        grouped.set(key, {
          start_date: r.date,
          end_date: r.date,
          open: o,
          high: h,
          low: l,
          close: c,
        });
        continue;
      }

      if (r.date < existing.start_date) {
        existing.start_date = r.date;
        existing.open = o;
      }
      if (r.date > existing.end_date) {
        existing.end_date = r.date;
        existing.close = c;
      }
      existing.high = Math.max(existing.high, h);
      existing.low = Math.min(existing.low, l);
    }

    const months = Array.from(grouped.entries())
      .sort(([a], [b]) => (a > b ? 1 : -1))
      .map(([month, v]) => ({
        month,
        ...v,
      }));

    return NextResponse.json({ months });
  } catch (e: any) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to load EGX30 monthly candles" },
      { status: 500 }
    );
  }
}

