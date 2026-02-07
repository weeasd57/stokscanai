import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";

type Egx30Row = {
  date: string;
  open?: number | string | null;
  close?: number | string | null;
};

function toNum(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : typeof v === "number" ? v : null;
  if (n === null || !Number.isFinite(n)) return null;
  return n;
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const start = url.searchParams.get("start");
    const end = url.searchParams.get("end");
    if (!start || !end) {
      return NextResponse.json({ error: "Missing start/end" }, { status: 400 });
    }

    const filePath = path.join(process.cwd(), "..", "symbols_data", "EGX30-INDEX.json");
    const raw = await fs.readFile(filePath, "utf8");
    const rows = JSON.parse(raw) as Egx30Row[];

    const startDt = new Date(start);
    const endDt = new Date(end);

    const period = (Array.isArray(rows) ? rows : [])
      .filter((r) => typeof r?.date === "string")
      .filter((r) => {
        const d = new Date(r.date);
        return d >= startDt && d <= endDt;
      })
      .sort((a, b) => (a.date > b.date ? 1 : -1));

    if (period.length < 2) {
      return NextResponse.json({ return_pct: null });
    }

    const first = period[0];
    const last = period[period.length - 1];
    const startPrice = toNum(first.open) ?? toNum(first.close);
    const endPrice = toNum(last.close);
    if (startPrice === null || endPrice === null) {
      return NextResponse.json({ return_pct: null });
    }

    const pct = ((endPrice - startPrice) / startPrice) * 100;
    return NextResponse.json({ return_pct: Math.round(pct * 100) / 100 });
  } catch (e: any) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to compute EGX30 return" },
      { status: 500 }
    );
  }
}

