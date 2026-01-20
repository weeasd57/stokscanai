"use client";

import { X, Loader2, BarChart2, LineChart } from "lucide-react";
import CandleChart from "@/components/CandleChart";
import { useLanguage } from "@/contexts/LanguageContext";
import { useEffect, useState } from "react";
import { predictStock } from "@/lib/api";
import { TestPredictionRow } from "@/lib/types";

interface ChartDialogProps {
    symbol: string | null;
    onClose: () => void;
}

export default function ChartDialog({ symbol, onClose }: ChartDialogProps) {
    const { t } = useLanguage();
    const [rows, setRows] = useState<TestPredictionRow[]>([]);
    const [loading, setLoading] = useState(false);

    // Indicator visibility states
    const [chartType, setChartType] = useState<"candle" | "area">("candle");
    const [showEma50, setShowEma50] = useState(true);
    const [showEma200, setShowEma200] = useState(true);
    const [showBB, setShowBB] = useState(false);
    const [showRsi, setShowRsi] = useState(true);
    const [showVolume, setShowVolume] = useState(true);

    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", handleEsc);
        return () => window.removeEventListener("keydown", handleEsc);
    }, [onClose]);

    useEffect(() => {
        if (!symbol) return;

        const fetchData = async () => {
            setLoading(true);
            try {
                const d = new Date();
                d.setDate(d.getDate() - 250);
                const fromDate = d.toISOString().split('T')[0];

                const data = await predictStock({
                    ticker: symbol,
                    fromDate,
                    rfPreset: "default"
                });
                if (data && data.testPredictions) {
                    setRows(data.testPredictions);
                }
            } catch (err) {
                console.error("Failed to load chart data", err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [symbol]);

    if (!symbol) return null;

    return (
        <div className="fixed inset-0 z-[9999] flex flex-col">
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            <div className="relative w-full h-full bg-zinc-950 flex flex-col">
                {/* Header */}
                <div className="flex-none flex items-center justify-between px-6 py-4 border-b border-white/5 bg-zinc-900/50">
                    <div className="space-y-1">
                        <h2 className="text-xl font-black text-white uppercase tracking-tight">
                            {symbol}
                        </h2>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Chart Type Toggle */}
                        <div className="flex gap-1 p-1 rounded-xl bg-zinc-800/50 border border-white/5">
                            <button
                                onClick={() => setChartType("candle")}
                                className={`p-2 rounded-lg transition-all ${chartType === "candle" ? "bg-indigo-600 text-white shadow-lg" : "text-zinc-500 hover:text-white"}`}
                            >
                                <BarChart2 className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setChartType("area")}
                                className={`p-2 rounded-lg transition-all ${chartType === "area" ? "bg-indigo-600 text-white shadow-lg" : "text-zinc-500 hover:text-white"}`}
                            >
                                <LineChart className="w-4 h-4" />
                            </button>
                        </div>

                        {/* Indicator Toggles */}
                        <div className="flex gap-1.5">
                            {[
                                { id: "ema50", label: "EMA50", color: "bg-orange-500", active: showEma50, toggle: () => setShowEma50(!showEma50) },
                                { id: "ema200", label: "EMA200", color: "bg-cyan-500", active: showEma200, toggle: () => setShowEma200(!showEma200) },
                                { id: "bb", label: "BB", color: "bg-purple-500", active: showBB, toggle: () => setShowBB(!showBB) },
                                { id: "rsi", label: "RSI", color: "bg-pink-500", active: showRsi, toggle: () => setShowRsi(!showRsi) },
                                { id: "vol", label: "VOL", color: "bg-blue-500", active: showVolume, toggle: () => setShowVolume(!showVolume) },
                            ].map((ind) => (
                                <button
                                    key={ind.id}
                                    onClick={ind.toggle}
                                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wide transition-all ${ind.active
                                            ? "bg-white/10 text-white border border-white/20"
                                            : "text-zinc-600 hover:text-zinc-400 border border-transparent"
                                        }`}
                                >
                                    <div className={`w-2 h-2 rounded-full ${ind.active ? ind.color : "bg-zinc-700"}`} />
                                    {ind.label}
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={onClose}
                            className="p-2 rounded-xl text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
                        >
                            <X className="w-6 h-6" />
                        </button>
                    </div>
                </div>

                {/* Content - Full height */}
                <div className="flex-1 p-4 bg-zinc-950 flex flex-col overflow-hidden">
                    {loading ? (
                        <div className="flex-1 flex flex-col items-center justify-center gap-4 text-zinc-500">
                            <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
                            <span className="text-sm font-bold uppercase tracking-widest animate-pulse">Loading market data...</span>
                        </div>
                    ) : rows.length > 0 ? (
                        <CandleChart
                            rows={rows}
                            showEma50={showEma50}
                            showEma200={showEma200}
                            showBB={showBB}
                            showRsi={showRsi}
                            showVolume={showVolume}
                            chartType={chartType}
                            height={500}
                        />
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center gap-4 text-zinc-500">
                            <span className="text-sm font-bold uppercase tracking-widest">No data available</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
