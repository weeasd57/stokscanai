"use client";

import { useEffect, useRef, useMemo } from "react";
import { createChart, ColorType, IChartApi, CrosshairMode, SeriesMarker, Time } from "lightweight-charts";
import type { TestPredictionRow } from "@/lib/types";

interface PriceChartProps {
    rows: TestPredictionRow[];
    showBuySignals?: boolean;
    showSellSignals?: boolean;
    showSma50?: boolean;
    showSma200?: boolean;
    showEma50?: boolean;
    showEma200?: boolean;
    showBB?: boolean;
    showRsi?: boolean;
    showMacd?: boolean;
    showVolume?: boolean;
    height?: number;
}

export default function PriceChart({
    rows,
    showBuySignals = true,
    showSellSignals = true,
    showSma50 = false,
    showSma200 = false,
    showEma50 = false,
    showEma200 = false,
    showBB = false,
    showRsi = false,
    showMacd = false,
    showVolume = true,
    height = 400,
}: PriceChartProps) {
    const mainChartContainerRef = useRef<HTMLDivElement>(null);
    const rsiChartContainerRef = useRef<HTMLDivElement>(null);
    const macdChartContainerRef = useRef<HTMLDivElement>(null);

    const mainChartRef = useRef<IChartApi | null>(null);
    const rsiChartRef = useRef<IChartApi | null>(null);
    const macdChartRef = useRef<IChartApi | null>(null);

    const sortedRows = useMemo(() => {
        return [...rows].sort((a, b) => (a.date > b.date ? 1 : -1));
    }, [rows]);

    useEffect(() => {
        if (!mainChartContainerRef.current) return;

        // Cleanup previous charts
        if (mainChartRef.current) {
            mainChartRef.current.remove();
            mainChartRef.current = null;
        }
        if (rsiChartRef.current) {
            rsiChartRef.current.remove();
            rsiChartRef.current = null;
        }
        if (macdChartRef.current) {
            macdChartRef.current.remove();
            macdChartRef.current = null;
        }

        const chartOptions = {
            layout: {
                background: { type: ColorType.Solid, color: "#09090b" },
                textColor: "#a1a1aa",
            },
            grid: {
                vertLines: { color: "#27272a" },
                horzLines: { color: "#27272a" },
            },
            timeScale: {
                borderColor: "#27272a",
                timeVisible: true,
                secondsVisible: false,
            },
            crosshair: { mode: CrosshairMode.Magnet },
            rightPriceScale: {
                borderColor: "#27272a",
            },
        };

        // --- MAIN CHART ---
        const mainChart = createChart(mainChartContainerRef.current, {
            ...chartOptions,
            width: mainChartContainerRef.current.clientWidth,
            height: height,
        });
        mainChartRef.current = mainChart;

        // Candlestick series
        const candlestickData = sortedRows.map((r) => ({
            time: r.date as Time,
            open: r.open ?? r.close,
            high: r.high ?? r.close,
            low: r.low ?? r.close,
            close: r.close,
        }));

        const candleSeries = mainChart.addCandlestickSeries({
            upColor: "#22c55e",
            downColor: "#ef4444",
            borderVisible: false,
            wickUpColor: "#22c55e",
            wickDownColor: "#ef4444",
        });
        candleSeries.setData(candlestickData);

        // Volume histogram
        if (showVolume) {
            const volumeData = sortedRows.map((r) => ({
                time: r.date as Time,
                value: r.volume ?? 0,
                color: r.close >= (r.open ?? r.close) ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)",
            }));
            const volumeSeries = mainChart.addHistogramSeries({
                priceFormat: { type: "volume" },
                priceScaleId: "volume",
            });
            volumeSeries.priceScale().applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
            });
            volumeSeries.setData(volumeData);
        }

        // SMA 50
        if (showSma50) {
            const sma50Data = sortedRows
                .filter((r) => r.sma50 != null && r.sma50 !== 0)
                .map((r) => ({ time: r.date as Time, value: r.sma50! }));
            if (sma50Data.length > 0) {
                const sma50Series = mainChart.addLineSeries({
                    color: "#f59e0b",
                    lineWidth: 2,
                    title: "SMA 50",
                });
                sma50Series.setData(sma50Data);
            }
        }

        // SMA 200
        if (showSma200) {
            const sma200Data = sortedRows
                .filter((r) => r.sma200 != null && r.sma200 !== 0)
                .map((r) => ({ time: r.date as Time, value: r.sma200! }));
            if (sma200Data.length > 0) {
                const sma200Series = mainChart.addLineSeries({
                    color: "#06b6d4",
                    lineWidth: 2,
                    title: "SMA 200",
                });
                sma200Series.setData(sma200Data);
            }
        }

        // EMA 50
        if (showEma50) {
            const ema50Data = sortedRows
                .filter((r) => r.ema50 != null && r.ema50 !== 0)
                .map((r) => ({ time: r.date as Time, value: r.ema50! }));
            if (ema50Data.length > 0) {
                const ema50Series = mainChart.addLineSeries({
                    color: "#f97316",
                    lineWidth: 2,
                    lineStyle: 2,
                    title: "EMA 50",
                });
                ema50Series.setData(ema50Data);
            }
        }

        // EMA 200
        if (showEma200) {
            const ema200Data = sortedRows
                .filter((r) => r.ema200 != null && r.ema200 !== 0)
                .map((r) => ({ time: r.date as Time, value: r.ema200! }));
            if (ema200Data.length > 0) {
                const ema200Series = mainChart.addLineSeries({
                    color: "#0ea5e9",
                    lineWidth: 2,
                    lineStyle: 2,
                    title: "EMA 200",
                });
                ema200Series.setData(ema200Data);
            }
        }

        // Bollinger Bands
        if (showBB) {
            const bbUpperData = sortedRows
                .filter((r) => r.bb_upper != null && r.bb_upper !== 0)
                .map((r) => ({ time: r.date as Time, value: r.bb_upper! }));
            const bbLowerData = sortedRows
                .filter((r) => r.bb_lower != null && r.bb_lower !== 0)
                .map((r) => ({ time: r.date as Time, value: r.bb_lower! }));

            if (bbUpperData.length > 0 && bbLowerData.length > 0) {
                const bbUpperSeries = mainChart.addLineSeries({
                    color: "rgba(168, 85, 247, 0.6)",
                    lineWidth: 1,
                    lineStyle: 2,
                    title: "BB Upper",
                });
                bbUpperSeries.setData(bbUpperData);

                const bbLowerSeries = mainChart.addLineSeries({
                    color: "rgba(168, 85, 247, 0.6)",
                    lineWidth: 1,
                    lineStyle: 2,
                    title: "BB Lower",
                });
                bbLowerSeries.setData(bbLowerData);
            }
        }

        // Signal Markers
        const markers: SeriesMarker<Time>[] = [];

        if (showBuySignals) {
            sortedRows
                .filter((r) => r.pred === 1)
                .forEach((r) => {
                    markers.push({
                        time: r.date as Time,
                        position: "belowBar",
                        color: "#22c55e",
                        shape: "circle",
                        text: "BUY",
                    });
                });
        }

        if (showSellSignals) {
            sortedRows
                .filter((r) => r.pred === 0)
                .forEach((r) => {
                    markers.push({
                        time: r.date as Time,
                        position: "aboveBar",
                        color: "#ef4444",
                        shape: "arrowDown",
                        text: "SELL",
                    });
                });
        }

        // Sort markers by time
        markers.sort((a, b) => (a.time > b.time ? 1 : -1));
        candleSeries.setMarkers(markers);

        // --- RSI CHART ---
        let rsiChart: IChartApi | null = null;
        if (showRsi && rsiChartContainerRef.current) {
            rsiChart = createChart(rsiChartContainerRef.current, {
                ...chartOptions,
                width: rsiChartContainerRef.current.clientWidth,
                height: 120,
            });
            rsiChartRef.current = rsiChart;

            const rsiData = sortedRows
                .filter((r) => r.rsi != null)
                .map((r) => ({ time: r.date as Time, value: r.rsi ?? 50 }));

            const rsiSeries = rsiChart.addLineSeries({
                color: "#a855f7",
                lineWidth: 2,
                title: "RSI",
            });
            rsiSeries.setData(rsiData);

            // Overbought/Oversold lines
            const overboughtData = sortedRows.map((r) => ({ time: r.date as Time, value: 70 }));
            const oversoldData = sortedRows.map((r) => ({ time: r.date as Time, value: 30 }));

            const obLine = rsiChart.addLineSeries({
                color: "rgba(239, 68, 68, 0.5)",
                lineWidth: 1,
                lineStyle: 2,
                title: "Ob",
            });
            obLine.setData(overboughtData);

            const osLine = rsiChart.addLineSeries({
                color: "rgba(34, 197, 94, 0.5)",
                lineWidth: 1,
                lineStyle: 2,
                title: "Os",
            });
            osLine.setData(oversoldData);
        }

        // --- MACD CHART ---
        let macdChart: IChartApi | null = null;
        if (showMacd && macdChartContainerRef.current) {
            macdChart = createChart(macdChartContainerRef.current, {
                ...chartOptions,
                width: macdChartContainerRef.current.clientWidth,
                height: 140,
            });
            macdChartRef.current = macdChart;

            const macdData = sortedRows
                .filter((r) => r.macd != null)
                .map((r) => ({ time: r.date as Time, value: r.macd ?? 0 }));

            const macdSeries = macdChart.addLineSeries({
                color: "#2962FF",
                lineWidth: 1,
                title: "MACD",
            });
            macdSeries.setData(macdData);

            const signalData = sortedRows
                .filter((r) => r.macd_signal != null)
                .map((r) => ({ time: r.date as Time, value: r.macd_signal ?? 0 }));

            const signalSeries = macdChart.addLineSeries({
                color: "#FF6D00",
                lineWidth: 1,
                title: "Signal",
            });
            signalSeries.setData(signalData);

            // Histogram
            const histData = sortedRows
                .filter((r) => r.macd != null && r.macd_signal != null)
                .map((r) => {
                    const diff = (r.macd ?? 0) - (r.macd_signal ?? 0);
                    return {
                        time: r.date as Time,
                        value: diff,
                        color: diff >= 0 ? "#26a69a" : "#ef5350",
                    };
                });

            const histSeries = macdChart.addHistogramSeries({
                title: "Hist",
            });
            histSeries.setData(histData);
        }

        // Sync time scales
        const allCharts = [mainChart, rsiChart, macdChart].filter(Boolean) as IChartApi[];
        allCharts.forEach((chart, idx) => {
            chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                if (range) {
                    allCharts.forEach((other, otherIdx) => {
                        if (idx !== otherIdx) {
                            other.timeScale().setVisibleLogicalRange(range);
                        }
                    });
                }
            });
        });

        mainChart.timeScale().fitContent();

        // Resize handler
        const handleResize = () => {
            if (mainChartContainerRef.current && mainChartRef.current) {
                mainChartRef.current.applyOptions({ width: mainChartContainerRef.current.clientWidth });
            }
            if (rsiChartContainerRef.current && rsiChartRef.current) {
                rsiChartRef.current.applyOptions({ width: rsiChartContainerRef.current.clientWidth });
            }
            if (macdChartContainerRef.current && macdChartRef.current) {
                macdChartRef.current.applyOptions({ width: macdChartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            if (mainChartRef.current) {
                mainChartRef.current.remove();
                mainChartRef.current = null;
            }
            if (rsiChartRef.current) {
                rsiChartRef.current.remove();
                rsiChartRef.current = null;
            }
            if (macdChartRef.current) {
                macdChartRef.current.remove();
                macdChartRef.current = null;
            }
        };
    }, [
        sortedRows,
        showBuySignals,
        showSellSignals,
        showSma50,
        showSma200,
        showEma50,
        showEma200,
        showBB,
        showRsi,
        showMacd,
        showVolume,
        height
    ]);

    return (
        <div className="w-full rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-1">
            {/* Main Chart */}
            <div ref={mainChartContainerRef} className="w-full" />

            {/* RSI Chart */}
            {showRsi && (
                <>
                    <div className="text-xs text-zinc-500 px-2 mt-3 font-bold uppercase tracking-wider">
                        RSI (14)
                    </div>
                    <div ref={rsiChartContainerRef} className="w-full border-t border-zinc-800" />
                </>
            )}

            {/* MACD Chart */}
            {showMacd && (
                <>
                    <div className="text-xs text-zinc-500 px-2 mt-3 font-bold uppercase tracking-wider">
                        MACD (12, 26, 9)
                    </div>
                    <div ref={macdChartContainerRef} className="w-full border-t border-zinc-800" />
                </>
            )}
        </div>
    );
}
