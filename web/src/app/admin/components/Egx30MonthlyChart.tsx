"use client";

import React from "react";
import { createChart, ColorType, CrosshairMode, type IChartApi } from "lightweight-charts";
import { RefreshCw, ZoomIn, ZoomOut, Move } from "lucide-react";

type MonthCandle = {
  month: string; // YYYY-MM
  start_date: string; // YYYY-MM-DD
  end_date: string; // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
};

export default function Egx30MonthlyChart({
  onSelectMonth,
}: {
  onSelectMonth: (range: { startDate: string; endDate: string; month: string }) => void;
}) {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const monthByTimeRef = React.useRef<Map<string, MonthCandle>>(new Map());
  const fitContentRef = React.useRef<(() => void) | null>(null);

  const [months, setMonths] = React.useState<MonthCandle[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const zoom = React.useCallback((direction: "in" | "out") => {
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const range = ts.getVisibleLogicalRange();
    if (!range) return;

    const from = range.from as number;
    const to = range.to as number;
    const span = Math.max(5, to - from);
    const center = from + span / 2;
    const nextSpan = direction === "in" ? span * 0.8 : span / 0.8;
    ts.setVisibleLogicalRange({ from: center - nextSpan / 2, to: center + nextSpan / 2 });
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/egx30/monthly");
        const json = await res.json();
        if (!res.ok) throw new Error(json?.error || "Failed to load EGX30 candles");
        const data = (json?.months ?? []) as MonthCandle[];
        if (!cancelled) setMonths(Array.isArray(data) ? data : []);
      } catch (e: any) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load EGX30 candles");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    if (!containerRef.current) return;
    if (loading || error) return;
    if (!months.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }
    fitContentRef.current = null;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 220,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#a1a1aa",
        fontSize: 11,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif',
      },
      grid: {
        vertLines: { color: "rgba(39,39,42,0.6)" },
        horzLines: { color: "rgba(39,39,42,0.6)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: "rgba(39,39,42,0.6)",
      },
      timeScale: {
        borderColor: "rgba(39,39,42,0.6)",
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      // TradingView-like controls:
      // - Zoom: mouse wheel / pinch / axis drag
      // - Pan: drag / mouse wheel scroll
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    });
    chartRef.current = chart;
    fitContentRef.current = () => chart.timeScale().fitContent();

    const series = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      borderVisible: false,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });

    const items = months.map((m) => {
      const time = `${m.month}-01`;
      monthByTimeRef.current.set(time, m);
      return {
        time,
        open: m.open,
        high: m.high,
        low: m.low,
        close: m.close,
      };
    });
    series.setData(items as any);
    chart.timeScale().fitContent();

    const onResize = () => {
      if (!containerRef.current || !chartRef.current) return;
      chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(containerRef.current);

    chart.subscribeClick((param) => {
      const t = (param as any)?.time as string | undefined;
      if (!t) return;
      const m = monthByTimeRef.current.get(t);
      if (!m) return;
      onSelectMonth({ startDate: m.start_date, endDate: m.end_date, month: m.month });
    });

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      fitContentRef.current = null;
    };
  }, [months, loading, error, onSelectMonth]);

  return (
    <div className="rounded-2xl border border-white/10 bg-zinc-950/40 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-zinc-900/40">
        <div className="flex items-center gap-3">
          <div className="text-[10px] font-black uppercase tracking-widest text-zinc-400">EGX30 Monthly Candles</div>
          <div className="hidden sm:flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-zinc-600">
            <Move className="h-3.5 w-3.5" />
            Drag to move • Wheel/pinch to zoom • Click candle to set dates
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => zoom("in")}
            className="p-2 rounded-xl border border-white/10 bg-white/5 text-zinc-300 hover:text-white hover:bg-white/10 transition-all"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => zoom("out")}
            className="p-2 rounded-xl border border-white/10 bg-white/5 text-zinc-300 hover:text-white hover:bg-white/10 transition-all"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => fitContentRef.current?.()}
            className="p-2 rounded-xl border border-white/10 bg-white/5 text-zinc-300 hover:text-white hover:bg-white/10 transition-all"
            title="Reset view"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div className="p-4">
        {loading ? (
          <div className="p-8 rounded-xl border border-white/10 bg-zinc-950/40 text-center text-zinc-400 text-sm">
            Loading EGX30…
          </div>
        ) : error ? (
          <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/10 text-red-300 text-sm">
            {error}
          </div>
        ) : !months.length ? (
          <div className="p-8 rounded-xl border border-white/10 bg-zinc-950/40 text-center text-zinc-400 text-sm">
            No EGX30 data.
          </div>
        ) : (
          <div ref={containerRef} />
        )}
      </div>
    </div>
  );
}
