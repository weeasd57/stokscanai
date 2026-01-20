"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, CrosshairMode } from "lightweight-charts";

import type { TestPredictionRow } from "@/lib/types";

interface CandleChartProps {
    rows: TestPredictionRow[];
    showEma50?: boolean;
    showEma200?: boolean;
    showBB?: boolean;
    showRsi?: boolean;
    showVolume?: boolean;
    chartType?: "candle" | "area";
    targetPrice?: number;
    stopPrice?: number;
    savedDate?: string;
    height?: number;
}

export default function CandleChart({
    rows,
    showEma50 = false,
    showEma200 = false,
    showBB = false,
    showRsi = false,
    showVolume = false,
    chartType = "candle",
    targetPrice,
    stopPrice,
    savedDate,
    height = 450,
}: CandleChartProps) {
    const mainChartContainerRef = useRef<HTMLDivElement>(null);
    const macdChartContainerRef = useRef<HTMLDivElement>(null);
    const rsiChartContainerRef = useRef<HTMLDivElement>(null);

    const mainChartRef = useRef<IChartApi | null>(null);
    const macdChartRef = useRef<IChartApi | null>(null);
    const rsiChartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!mainChartContainerRef.current || !macdChartContainerRef.current) return;

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
        };

        // --- MAIN CHART ---
        const mainChart = createChart(mainChartContainerRef.current, {
            ...chartOptions,
            width: mainChartContainerRef.current.clientWidth,
            height: height,
        });

        // --- MACD CHART ---
        const macdChart = createChart(macdChartContainerRef.current, {
            ...chartOptions,
            width: macdChartContainerRef.current.clientWidth,
            height: 160,
        });

        // --- RSI CHART (conditional) ---
        let rsiChart: IChartApi | null = null;
        if (showRsi && rsiChartContainerRef.current) {
            rsiChart = createChart(rsiChartContainerRef.current, {
                ...chartOptions,
                width: rsiChartContainerRef.current.clientWidth,
                height: 140,
            });
        }

        // --- DATA PREP ---
        const sortedRows = [...rows].sort((a, b) => (a.date > b.date ? 1 : -1));

        // --- MAIN SERIES ---
        let mainSeries: any;

        if (chartType === "area") {
            const areaData = sortedRows.map((r) => ({
                time: r.date,
                value: r.close,
            }));
            mainSeries = mainChart.addAreaSeries({
                topColor: 'rgba(33, 150, 243, 0.56)',
                bottomColor: 'rgba(33, 150, 243, 0.04)',
                lineColor: 'rgba(33, 150, 243, 1)',
                lineWidth: 2,
            });
            mainSeries.setData(areaData);
        } else {
            const candleData = sortedRows.map((r) => ({
                time: r.date,
                open: r.open ?? r.close,
                high: r.high ?? r.close,
                low: r.low ?? r.close,
                close: r.close,
            }));
            mainSeries = mainChart.addCandlestickSeries({
                upColor: "#22c55e",
                downColor: "#ef4444",
                borderVisible: false,
                wickUpColor: "#22c55e",
                wickDownColor: "#ef4444",
            });
            mainSeries.setData(candleData);
        }

        // Volume (on main chart as histogram)
        if (showVolume) {
            const volumeData = sortedRows.map((r) => ({
                time: r.date,
                value: r.volume ?? 0,
                color: (r.close >= (r.open ?? r.close)) ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'
            }));
            const volumeSeries = mainChart.addHistogramSeries({
                priceFormat: { type: 'volume' },
                priceScaleId: 'volume',
            });
            volumeSeries.priceScale().applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
            });
            volumeSeries.setData(volumeData);
        }

        // EMA 50
        if (showEma50) {
            const ema50Data = sortedRows
                .map((r) => ({ time: r.date, value: r.ema50 ?? 0 }))
                .filter(d => d.value !== 0);
            if (ema50Data.length > 0) {
                const s = mainChart.addLineSeries({ color: "#f97316", lineWidth: 2, title: "EMA 50" });
                s.setData(ema50Data);
            }
        }

        // EMA 200
        if (showEma200) {
            const ema200Data = sortedRows
                .map((r) => ({ time: r.date, value: r.ema200 ?? 0 }))
                .filter(d => d.value !== 0);
            if (ema200Data.length > 0) {
                const s = mainChart.addLineSeries({ color: "#06b6d4", lineWidth: 2, title: "EMA 200" });
                s.setData(ema200Data);
            }
        }

        // Bollinger Bands
        if (showBB) {
            const bbUpperData = sortedRows.map(r => ({ time: r.date, value: r.bb_upper ?? 0 })).filter(d => d.value !== 0);
            const bbLowerData = sortedRows.map(r => ({ time: r.date, value: r.bb_lower ?? 0 })).filter(d => d.value !== 0);

            if (bbUpperData.length > 0 && bbLowerData.length > 0) {
                const u = mainChart.addLineSeries({ color: 'rgba(168, 85, 247, 0.6)', lineWidth: 1, lineStyle: 2, title: 'BB Upper' });
                u.setData(bbUpperData);
                const l = mainChart.addLineSeries({ color: 'rgba(168, 85, 247, 0.6)', lineWidth: 1, lineStyle: 2, title: 'BB Lower' });
                l.setData(bbLowerData);
            }
        }

        // Markers (Buy Signals + Saved Date)
        const markers: any[] = sortedRows.filter((r) => r.pred === 1).map((r) => ({
            time: r.date,
            position: "belowBar",
            color: "#22c55e",
            shape: "circle",
            text: "BUY",
        }));

        if (savedDate) {
            // Find closest date
            const exact = sortedRows.find(r => r.date === savedDate);
            // If exact match not found, could search close, but strict for now
            if (exact) {
                markers.push({
                    time: savedDate,
                    position: "aboveBar",
                    color: "#facc15",
                    shape: "arrowDown",
                    text: "SAVED",
                    size: 2
                });
            }
        }

        // @ts-ignore
        mainSeries.setMarkers(markers);

        // Target / Stop Lines
        if (targetPrice) {
            const targetLine = mainSeries.createPriceLine({
                price: targetPrice,
                color: '#22c55e',
                lineWidth: 2,
                lineStyle: 1, // Dotted
                axisLabelVisible: true,
                title: 'TARGET',
            });
        }
        if (stopPrice) {
            const stopLine = mainSeries.createPriceLine({
                price: stopPrice,
                color: '#ef4444',
                lineWidth: 2,
                lineStyle: 1, // Dotted
                axisLabelVisible: true,
                title: 'STOP',
            });
        }

        // --- MACD SERIES ---
        const macdData = sortedRows.map(r => ({ time: r.date, value: r.macd ?? 0 }));
        const macdSeries = macdChart.addLineSeries({ color: '#2962FF', lineWidth: 1, title: 'MACD' });
        macdSeries.setData(macdData);

        const signalData = sortedRows.map(r => ({ time: r.date, value: r.macd_signal ?? 0 }));
        const signalSeries = macdChart.addLineSeries({ color: '#FF6D00', lineWidth: 1, title: 'Signal' });
        signalSeries.setData(signalData);

        const histData = sortedRows.map(r => ({
            time: r.date,
            value: (r.macd ?? 0) - (r.macd_signal ?? 0),
            color: ((r.macd ?? 0) - (r.macd_signal ?? 0)) >= 0 ? '#26a69a' : '#ef5350'
        }));
        const histSeries = macdChart.addHistogramSeries({ title: 'Hist' });
        histSeries.setData(histData);

        // --- RSI SERIES ---
        if (rsiChart) {
            const rsiData = sortedRows.map(r => ({ time: r.date, value: r.rsi ?? 50 }));
            const rsiSeries = rsiChart.addLineSeries({ color: '#a855f7', lineWidth: 2, title: 'RSI' });
            rsiSeries.setData(rsiData);

            // Add overbought/oversold lines
            const overbought = sortedRows.map(r => ({ time: r.date, value: 70 }));
            const oversold = sortedRows.map(r => ({ time: r.date, value: 30 }));
            const obLine = rsiChart.addLineSeries({ color: 'rgba(239, 68, 68, 0.5)', lineWidth: 1, lineStyle: 2 });
            obLine.setData(overbought);
            const osLine = rsiChart.addLineSeries({ color: 'rgba(34, 197, 94, 0.5)', lineWidth: 1, lineStyle: 2 });
            osLine.setData(oversold);
        }

        // --- SYNC TIMEFRAMES ---
        const syncCharts = [mainChart, macdChart, rsiChart].filter(Boolean) as IChartApi[];

        syncCharts.forEach((chart, idx) => {
            chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) {
                    syncCharts.forEach((other, otherIdx) => {
                        if (idx !== otherIdx) {
                            other.timeScale().setVisibleLogicalRange(range);
                        }
                    });
                }
            });
        });

        mainChart.timeScale().fitContent();

        mainChartRef.current = mainChart;
        macdChartRef.current = macdChart;
        rsiChartRef.current = rsiChart;

        const handleResize = () => {
            if (mainChartContainerRef.current) mainChart.applyOptions({ width: mainChartContainerRef.current.clientWidth });
            if (macdChartContainerRef.current) macdChart.applyOptions({ width: macdChartContainerRef.current.clientWidth });
            if (rsiChartContainerRef.current && rsiChart) rsiChart.applyOptions({ width: rsiChartContainerRef.current.clientWidth });
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            mainChart.remove();
            macdChart.remove();
            if (rsiChart) rsiChart.remove();
        };
    }, [rows, showEma50, showEma200, showBB, showRsi, showVolume, chartType, targetPrice, stopPrice, savedDate]);

    return (
        <div className="w-full rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-1">
            {/* Main Chart */}
            <div ref={mainChartContainerRef} className="w-full" />

            {/* MACD Chart */}
            <div className="text-xs text-zinc-500 px-2 mt-4 font-bold uppercase tracking-wider">MACD (12, 26, 9)</div>
            <div ref={macdChartContainerRef} className="w-full h-[160px] border-t border-zinc-800" />

            {/* RSI Chart (conditional) */}
            {showRsi && (
                <>
                    <div className="text-xs text-zinc-500 px-2 mt-4 font-bold uppercase tracking-wider">RSI (14)</div>
                    <div ref={rsiChartContainerRef} className="w-full h-[140px] border-t border-zinc-800" />
                </>
            )}
        </div>
    );
}
