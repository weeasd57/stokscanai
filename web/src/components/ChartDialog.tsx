"use client";

import { X, Loader2 } from "lucide-react";
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
                // We reuse predictStock to get full data including history
                // Calculate approx 200 days ago for fromDate
                const d = new Date();
                d.setDate(d.getDate() - 250); // Fetch a bit more context
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
        <div className="fixed inset-0 z-[300] flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            <div className="relative w-full max-w-[95vw] md:max-w-5xl rounded-[2rem] border border-white/10 bg-zinc-950 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[90vh]">
                {/* Header */}
                <div className="flex-none flex items-center justify-between px-6 py-4 border-b border-white/5 bg-zinc-900/50">
                    <div className="space-y-1">
                        <h2 className="text-xl font-black text-white uppercase tracking-tight">
                            {t("chart.title").replace("{symbol}", symbol)}
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-xl text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 p-1 min-h-[300px] md:h-[600px] bg-zinc-950 flex flex-col items-center justify-center overflow-hidden">
                    {loading ? (
                        <div className="flex flex-col items-center gap-4 text-zinc-500">
                            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                            <span className="text-sm font-medium animate-pulse">Loading market data...</span>
                        </div>
                    ) : (
                        <CandleChart
                            rows={rows}
                            showEma50
                            showEma200
                            showBB
                            showRsi
                            showVolume
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
