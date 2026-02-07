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
    showMacd?: boolean;
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
    showMacd = true,
    chartType = "candle",
    targetPrice,
    stopPrice,
    savedDate,
    height = 450,
}: CandleChartProps) {
    const mainChartContainerRef = useRef<HTMLDivElement>(null);
    const macdChartContainerRef = useRef<HTMLDivElement>(null);
    const rsiChartContainerRef = useRef<HTMLDivElement>(null);
    const legendRef = useRef<HTMLDivElement>(null);

    const mainChartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!mainChartContainerRef.current) return;

        const chartOptions: any = {
            layout: {
                background: { type: ColorType.Solid, color: "#09090b" },
                textColor: "#a1a1aa",
                fontSize: 11,
                fontFamily: '-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif',
            },
            grid: {
                vertLines: { color: "#27272a" },
                horzLines: { color: "#27272a" },
            },
            timeScale: {
                borderColor: "#27272a",
                timeVisible: true,
                secondsVisible: false,
                visible: true,
                fixLeftEdge: true,
                fixRightEdge: true,
                barSpacing: 8,
                rightOffset: 5,
            },
            rightPriceScale: {
                borderColor: "#27272a",
                autoScale: true,
                visible: true,
                alignLabels: true,
            },
            crosshair: {
                mode: CrosshairMode.Magnet,
                vertLine: {
                    labelVisible: true,
                    color: "#758696",
                    width: 1,
                    style: 3,
                    labelBackgroundColor: "#758696",
                },
                horzLine: {
                    labelVisible: true,
                    color: "#758696",
                    width: 1,
                    style: 3,
                    labelBackgroundColor: "#758696",
                },
            },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
            },
            handleScale: {
                axisPressedMouseMove: false,
                mouseWheel: false,
                pinch: false,
            },
        };

        // --- MAIN CHART ---
        const mainChart = createChart(mainChartContainerRef.current, {
            ...chartOptions,
            width: mainChartContainerRef.current.clientWidth,
            height: height || 450,
        });
        mainChartRef.current = mainChart;

        // --- MACD CHART (conditional) ---
        let macdChart: IChartApi | null = null;
        if (showMacd && macdChartContainerRef.current) {
            macdChart = createChart(macdChartContainerRef.current, {
                ...chartOptions,
                width: macdChartContainerRef.current.clientWidth,
                height: 160,
                timeScale: { ...chartOptions.timeScale, visible: false }, // Hide time scale on sub-charts
            });
        }

        // --- RSI CHART (conditional) ---
        let rsiChart: IChartApi | null = null;
        if (showRsi && rsiChartContainerRef.current) {
            rsiChart = createChart(rsiChartContainerRef.current, {
                ...chartOptions,
                width: rsiChartContainerRef.current.clientWidth,
                height: 140,
                timeScale: { ...chartOptions.timeScale, visible: true }, // Show time scale on the bottom-most
            });
        }

        // --- DATA PREP ---
        const sortedRows = [...rows]
            .filter(r => r.close > 0)
            .sort((a, b) => (a.date > b.date ? 1 : -1));

        if (sortedRows.length === 0) return;

        // --- MAIN SERIES ---
        let mainSeries: any;

        if (chartType === "area") {
            const areaData = sortedRows.map((r) => ({
                time: r.date,
                value: r.close,
            }));
            mainSeries = mainChart.addAreaSeries({
                topColor: 'rgba(37, 99, 235, 0.4)', // blue-600
                bottomColor: 'rgba(37, 99, 235, 0.0)',
                lineColor: '#3b82f6', // blue-500
                lineWidth: 2,
                priceFormat: { precision: 2, minMove: 0.01 },
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
                priceFormat: {
                    type: 'price',
                    precision: 2,
                    minMove: 0.01,
                },
            });
            mainSeries.setData(candleData);
        }

        // Volume
        if (showVolume) {
            const volumeData = sortedRows.map((r) => ({
                time: r.date,
                value: r.volume ?? 0,
                color: (r.close >= (r.open ?? r.close)) ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'
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

        // Apply scale options for price
        mainChart.priceScale('right').applyOptions({
            scaleMargins: { top: 0.1, bottom: 0.2 },
            autoScale: true,
        });

        // Indicators
        if (showEma50) {
            const data = sortedRows.map(r => ({ time: r.date, value: r.ema50 ?? 0 })).filter(d => d.value !== 0);
            if (data.length > 0) mainChart.addLineSeries({ color: "#f97316", lineWidth: 1, title: "EMA 50" }).setData(data);
        }
        if (showEma200) {
            const data = sortedRows.map(r => ({ time: r.date, value: r.ema200 ?? 0 })).filter(d => d.value !== 0);
            if (data.length > 0) mainChart.addLineSeries({ color: "#06b6d4", lineWidth: 1, title: "EMA 200" }).setData(data);
        }
        if (showBB) {
            const u = sortedRows.map(r => ({ time: r.date, value: r.bb_upper ?? 0 })).filter(d => d.value !== 0);
            const l = sortedRows.map(r => ({ time: r.date, value: r.bb_lower ?? 0 })).filter(d => d.value !== 0);
            if (u.length > 0) mainChart.addLineSeries({ color: 'rgba(168, 85, 247, 0.3)', lineWidth: 1, lineStyle: 2 }).setData(u);
            if (l.length > 0) mainChart.addLineSeries({ color: 'rgba(168, 85, 247, 0.3)', lineWidth: 1, lineStyle: 2 }).setData(l);
        }

        // Markers
        const markers: any[] = [];
        let consec = 0;
        sortedRows.forEach(r => {
            if (r.pred === 1) {
                consec++;
                if (consec === 1 || consec % 5 === 0) {
                    markers.push({ time: r.date, position: "belowBar", color: "#22c55e", shape: "arrowUp", text: consec === 1 ? "BUY" : "", size: 1 });
                }
            } else consec = 0;
        });

        // Add Target/Stop markers at the saved date (origin of the signal)
        if (savedDate) {
            const hasDataAtDate = sortedRows.find(r => r.date === savedDate);
            if (hasDataAtDate) {
                markers.push({ time: savedDate, position: "aboveBar", color: "#facc15", shape: "arrowDown", text: "SIGNAL", size: 2 });

                if (targetPrice) {
                    markers.push({
                        time: savedDate,
                        position: "aboveBar",
                        color: "#10b981",
                        shape: "circle",
                        text: `TARGET @ ${targetPrice.toFixed(2)}`,
                        size: 1
                    });
                }
                if (stopPrice) {
                    markers.push({
                        time: savedDate,
                        position: "belowBar",
                        color: "#ef4444",
                        shape: "circle",
                        text: `STOP @ ${stopPrice.toFixed(2)}`,
                        size: 1
                    });
                }
            }
        }

        mainSeries.setMarkers(markers.sort((a, b) => a.time > b.time ? 1 : -1));

        // Viewport
        setTimeout(() => {
            mainChart.timeScale().setVisibleLogicalRange({ from: sortedRows.length - 100, to: sortedRows.length });
        }, 50);

        // Price Lines for persistent levels
        if (targetPrice) mainSeries.createPriceLine({ price: targetPrice, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'TARGET' });
        if (stopPrice) mainSeries.createPriceLine({ price: stopPrice, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'STOP' });

        // Legend Handler
        mainChart.subscribeCrosshairMove((param) => {
            if (!legendRef.current) return;
            const data = param.seriesData.get(mainSeries) as any;
            if (data) {
                const dateStr = typeof param.time === 'string' ? param.time : '';
                const open = data.open?.toFixed(2) ?? data.value?.toFixed(2) ?? '-';
                const high = data.high?.toFixed(2) ?? data.value?.toFixed(2) ?? '-';
                const low = data.low?.toFixed(2) ?? data.value?.toFixed(2) ?? '-';
                const close = data.close?.toFixed(2) ?? data.value?.toFixed(2) ?? '-';
                const color = (data.close >= data.open) ? 'text-emerald-500' : 'text-red-500';

                legendRef.current.innerHTML = `
                    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] font-bold uppercase tracking-wider">
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">O</span><span class="${color}">${open}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">H</span><span class="${color}">${high}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">L</span><span class="${color}">${low}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">C</span><span class="${color}">${close}</span></div>
                        <div class="hidden sm:flex items-center gap-1.5 ml-2"><span class="text-zinc-400 font-medium">${dateStr}</span></div>
                    </div>
                `;
            } else {
                const last = sortedRows[sortedRows.length - 1];
                if (!last) return;
                const color = last.close >= (last.open ?? last.close) ? 'text-emerald-500' : 'text-red-500';
                legendRef.current.innerHTML = `
                    <div class="flex items-center gap-x-4 text-[10px] font-bold uppercase tracking-wider">
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">O</span><span class="${color}">${(last.open ?? last.close).toFixed(2)}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">H</span><span class="${color}">${(last.high ?? last.close).toFixed(2)}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">L</span><span class="${color}">${(last.low ?? last.close).toFixed(2)}</span></div>
                        <div class="flex items-center gap-1.5"><span class="text-zinc-500">C</span><span class="${color}">${last.close.toFixed(2)}</span></div>
                    </div>
                `;
            }
        });

        // Sub-charts
        if (macdChart) {
            const mData = sortedRows.map(r => ({ time: r.date, value: r.macd ?? 0 }));
            macdChart.addLineSeries({ color: '#6366f1', lineWidth: 1, title: 'MACD' }).setData(mData);
            const sData = sortedRows.map(r => ({ time: r.date, value: r.macd_signal ?? 0 }));
            macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: 'SIGNAL' }).setData(sData);
            const hData = sortedRows.map(r => ({ time: r.date, value: (r.macd ?? 0) - (r.macd_signal ?? 0), color: ((r.macd ?? 0) - (r.macd_signal ?? 0)) >= 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)' }));
            macdChart.addHistogramSeries({ title: 'HIST' }).setData(hData);
        }
        if (rsiChart) {
            rsiChart.addLineSeries({ color: '#8b5cf6', lineWidth: 2, title: 'RSI' }).setData(sortedRows.map(r => ({ time: r.date, value: r.rsi ?? 50 })));
            rsiChart.priceScale('right').applyOptions({ scaleMargins: { top: 0.1, bottom: 0.1 } });
        }

        // Sync and Resize
        const syncs = [mainChart, macdChart, rsiChart].filter(Boolean) as IChartApi[];
        syncs.forEach((c, idx) => {
            c.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) syncs.forEach((o, oIdx) => { if (idx !== oIdx) o.timeScale().setVisibleLogicalRange(range); });
            });
        });

        // Resize Observer
        const resizeObserver = new ResizeObserver((entries) => {
            if (entries.length === 0 || !entries[0].contentRect) return;
            const newRect = entries[0].contentRect;
            mainChart.applyOptions({ width: newRect.width });
            if (macdChart) macdChart.applyOptions({ width: newRect.width });
            if (rsiChart) rsiChart.applyOptions({ width: newRect.width });
        });

        if (mainChartContainerRef.current) {
            resizeObserver.observe(mainChartContainerRef.current);
        }

        return () => {
            resizeObserver.disconnect();
            mainChart.remove();
            if (macdChart) macdChart.remove();
            if (rsiChart) rsiChart.remove();
        };
    }, [rows, showEma50, showEma200, showBB, showRsi, showVolume, showMacd, chartType, targetPrice, stopPrice, savedDate, height]);

    // Zoom Handler
    const handleZoom = (period: string) => {
        if (!mainChartRef.current || rows.length === 0) return;
        const total = rows.length;
        let visibleCount = total;

        switch (period) {
            case "1M": visibleCount = 22; break; // Approx 22 trading days
            case "3M": visibleCount = 66; break;
            case "6M": visibleCount = 132; break;
            case "YTD":
                // Simple approximation for YTD based on current date vs rows
                // Assuming rows are sorted daily, just take from Jan 1st of current year if possible
                // For simplicity, let's just use 252 days or full if less
                visibleCount = 252; // Fallback to 1Y essentially if accurate dates hard to parse quickly
                // Or better: find the index where year starts. 
                // Let's stick to simple fixed counts for robustness first.
                break;
            case "1Y": visibleCount = 252; break;
            case "ALL": visibleCount = total; break;
        }

        const from = Math.max(0, total - visibleCount);
        mainChartRef.current.timeScale().setVisibleLogicalRange({ from, to: total });
    };

    return (
        <div className="flex flex-col gap-2 w-full h-full">
            <div className="flex items-center justify-between px-2">
                <div ref={legendRef} className="font-mono text-[10px] text-zinc-400 font-bold uppercase tracking-wider flex items-center gap-4 min-h-[1.5rem]" />

                <div className="flex items-center gap-1">
                    {["1M", "3M", "6M", "1Y", "ALL"].map((p) => (
                        <button
                            key={p}
                            onClick={() => handleZoom(p)}
                            className="px-2 py-1 rounded bg-white/5 hover:bg-white/10 text-[9px] font-black text-zinc-500 hover:text-white uppercase transition-all"
                        >
                            {p}
                        </button>
                    ))}
                </div>
            </div>

            <div className="relative w-full border border-white/5 rounded-2xl overflow-hidden bg-black/20" style={{ height: height || 450 }}>
                <div ref={mainChartContainerRef} className="w-full h-full" />
            </div>

            {showMacd && (
                <div className="relative w-full border border-white/5 rounded-2xl overflow-hidden bg-black/20 mt-2" style={{ height: 150 }}>
                    <div ref={macdChartContainerRef} className="w-full h-full" />
                </div>
            )}
            {showRsi && (
                <div className="relative w-full border border-white/5 rounded-2xl overflow-hidden bg-black/20 mt-2" style={{ height: 150 }}>
                    <div ref={rsiChartContainerRef} className="w-full h-full" />
                </div>
            )}
        </div>
    );
}
