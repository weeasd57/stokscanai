"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, IChartApi, CrosshairMode, ISeriesApi } from "lightweight-charts";
import { Loader2, X, Maximize2, Minimize2 } from "lucide-react";

interface Candle {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface Marker {
    time: number;
    position: "aboveBar" | "belowBar" | "inBar";
    color: string;
    shape: "arrowUp" | "arrowDown" | "circle" | "square";
    text: string;
    price: number;
}

interface LiveCandleChartProps {
    symbol: string;
    botId?: string;
    height?: number;
    onClose?: () => void;
    showControls?: boolean;
    autoRefresh?: boolean;
}

export default function LiveCandleChart({
    symbol,
    botId = "primary",
    height = 300,
    onClose,
    showControls = true,
    autoRefresh = true
}: LiveCandleChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isMaximized, setIsMaximized] = useState(false);

    const fetchData = async () => {
        try {
            const res = await fetch(`/api/ai_bot/candles?symbol=${encodeURIComponent(symbol)}&bot_id=${botId}&limit=150`);
            if (!res.ok) {
                if (res.status === 404) throw new Error("No data found");
                throw new Error(`Fetch failed: ${res.status}`);
            }
            const data = await res.json();

            if (data.count === 0) {
                setError("No candle data available. Check 'Save to Supabase' setting.");
            } else {
                setError(null);
            }

            if (chartRef.current && candleSeriesRef.current) {
                // Update candle series
                candleSeriesRef.current.setData(data.candles);

                // Update volume series if it exists
                if (volumeSeriesRef.current) {
                    const volumeData = data.candles.map((c: Candle) => ({
                        time: c.time,
                        value: c.volume,
                        color: c.close >= c.open ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)"
                    }));
                    volumeSeriesRef.current.setData(volumeData);
                }

                // Update markers with snapping to candle times
                if (data.markers && data.markers.length > 0 && data.candles.length > 0) {
                    const candleTimes = data.candles.map((c: any) => c.time);
                    const validMarkers = data.markers.map((m: any) => {
                        // Find the nearest preceding candle time
                        const nearestTime = candleTimes.reduce((prev: number, curr: number) => {
                            return (curr <= m.time) ? curr : prev;
                        }, candleTimes[0]);

                        return {
                            ...m,
                            time: nearestTime
                        };
                    }).sort((a: any, b: any) => a.time - b.time);

                    candleSeriesRef.current.setMarkers(validMarkers);
                } else if (candleSeriesRef.current) {
                    candleSeriesRef.current.setMarkers([]);
                }
            }
            setError(null);
        } catch (err: any) {
            console.error("Chart fetch error:", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: "transparent" },
                textColor: "#A1A1AA",
                fontSize: 10,
                fontFamily: "JetBrains Mono, monospace",
            },
            grid: {
                vertLines: { color: "rgba(39, 39, 42, 0.5)" },
                horzLines: { color: "rgba(39, 39, 42, 0.5)" },
            },
            crosshair: {
                mode: CrosshairMode.Magnet,
                vertLine: { labelVisible: true, color: "#6366f1" },
                horzLine: { labelVisible: true, color: "#6366f1" },
            },
            timeScale: {
                borderColor: "rgba(39, 39, 42, 0.5)",
                timeVisible: true,
                secondsVisible: false,
                barSpacing: 6,
            },
            rightPriceScale: {
                borderColor: "rgba(39, 39, 42, 0.5)",
                autoScale: true,
                alignLabels: true,
            },
            handleScroll: { mouseWheel: true, pressedMouseMove: true },
            handleScale: { mouseWheel: true, pinch: true },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: "#22c55e",
            downColor: "#ef4444",
            borderVisible: false,
            wickUpColor: "#22c55e",
            wickDownColor: "#ef4444",
            priceFormat: { type: "price", precision: symbol.includes("SHIB") ? 6 : (symbol.includes("USD") ? 2 : 4) },
        });

        const volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });

        volumeSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        volumeSeriesRef.current = volumeSeries;

        fetchData();

        let interval: NodeJS.Timeout;
        if (autoRefresh) {
            interval = setInterval(() => {
                if (chartRef.current) fetchData();
            }, 10000);
        }

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener("resize", handleResize);
        const resizeTimeout = setTimeout(handleResize, 100);

        return () => {
            window.removeEventListener("resize", handleResize);
            clearTimeout(resizeTimeout);
            if (interval) clearInterval(interval);

            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
                candleSeriesRef.current = null;
                volumeSeriesRef.current = null;
            }
        };
    }, [symbol, botId, autoRefresh, isMaximized]);

    return (
        <div className={`relative bg-zinc-900/40 border border-zinc-800 rounded-2xl flex flex-col group transition-all ${isMaximized ? 'fixed inset-4 z-50 bg-black/95 shadow-2xl' : ''}`}
            style={{ height: isMaximized ? 'calc(100vh - 32px)' : height }}>

            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/50 border-b border-zinc-800 rounded-t-2xl">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-[10px] font-black tracking-widest text-zinc-300 uppercase">{symbol}</span>
                </div>

                {showControls && (
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                            onClick={() => setIsMaximized(!isMaximized)}
                            className="p-1.5 hover:bg-white/5 rounded-lg text-zinc-500 hover:text-white transition-colors"
                        >
                            {isMaximized ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                        </button>
                        {onClose && (
                            <button
                                onClick={onClose}
                                className="p-1.5 hover:bg-red-500/10 rounded-lg text-zinc-500 hover:text-red-400 transition-colors"
                            >
                                <X size={12} />
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* Chart Area */}
            <div className="flex-1 relative min-h-0">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/10 backdrop-blur-[1px]">
                        <Loader2 className="w-5 h-5 text-indigo-500 animate-spin" />
                    </div>
                )}

                {error && (
                    <div className="absolute inset-0 flex items-center justify-center z-10 text-[10px] text-red-400 font-mono text-center px-4">
                        {error}
                    </div>
                )}

                <div ref={chartContainerRef} className="w-full h-full" />
            </div>

            {/* Legend / Overlay */}
            {!loading && !error && (
                <div className="absolute top-12 left-4 pointer-events-none flex flex-col gap-0.5">
                    <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-tighter">Live Monitor</span>
                </div>
            )}
        </div>
    );
}
