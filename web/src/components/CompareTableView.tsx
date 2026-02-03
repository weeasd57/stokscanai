"use client";

import { useMemo } from "react";
import { TrendingUp, TrendingDown, Info, X, Loader2, AlertCircle, Star } from "lucide-react";
import type { PredictResponse } from "@/lib/types";
import { calculateIndicatorStats, type IndicatorStats } from "@/lib/indicators";
import { useLanguage } from "@/contexts/LanguageContext";
import StockLogo from "@/components/StockLogo";

function StatCell({ stats }: { stats: IndicatorStats }) {
    const winRate = parseFloat(stats.buyWinRate);
    const winRateColor = winRate >= 70 ? "text-green-400" : winRate >= 50 ? "text-yellow-400" : "text-red-400";

    return (
        <div className="flex flex-col gap-1 py-1">
            <div className="flex items-center justify-between gap-4 text-[10px] text-zinc-500">
                <span className="flex items-center gap-1">
                    <TrendingUp className="h-2.5 w-2.5 text-green-500" /> {stats.buySignals}
                </span>
                <span className="flex items-center gap-1">
                    <TrendingDown className="h-2.5 w-2.5 text-red-500" /> {stats.sellSignals}
                </span>
            </div>
            <div className={`text-xs font-bold ${winRateColor}`}>
                {stats.buyWinRate}% <span className="text-[10px] font-normal text-zinc-500">WR</span>
            </div>
        </div>
    );
}



type Props = {
    results: Record<string, PredictResponse>;
    loadingSymbols: string[];
    errors: Record<string, string>;
    onRemove: (symbol: string) => void;
    onSave?: (symbol: string) => void;
    onChart?: (symbol: string) => void;
    isSaved?: (symbol: string) => boolean;
};

export default function CompareTableView({ results, loadingSymbols, errors, onRemove, onSave, onChart, isSaved }: Props) {
    const { t } = useLanguage();

    // De-duplicate and prioritize: Success > Loading > Error
    const renderItems = useMemo(() => {
        const items: { type: "success" | "loading" | "error"; symbol: string; data?: PredictResponse; error?: string }[] = [];
        const seen = new Set<string>();

        // 1. Successful results
        Object.keys(results).sort().forEach(s => {
            if (!seen.has(s)) {
                items.push({ type: "success", symbol: s, data: results[s] });
                seen.add(s);
            }
        });

        // 2. Loading symbols (if not already in results)
        loadingSymbols.forEach(s => {
            if (!seen.has(s)) {
                items.push({ type: "loading", symbol: s });
                seen.add(s);
            }
        });

        // 3. Errors (if not already in results or loading)
        Object.entries(errors).forEach(([s, err]) => {
            if (!seen.has(s)) {
                items.push({ type: "error", symbol: s, error: err });
                seen.add(s);
            }
        });

        return items;
    }, [results, loadingSymbols, errors]);


    const statsMap = useMemo(() => {
        const map: Record<string, ReturnType<typeof calculateIndicatorStats>> = {};
        renderItems.forEach(item => {
            if (item.type === "success" && item.data) {
                map[item.symbol] = calculateIndicatorStats(item.data.testPredictions);
            }
        });
        return map;
    }, [renderItems]);

    if (renderItems.length === 0) {
        return (
            <div className="py-24 text-center border-2 border-dashed border-zinc-900 rounded-3xl bg-zinc-950/20 backdrop-blur-sm">
                <div className="mx-auto w-16 h-16 rounded-2xl bg-zinc-900 flex items-center justify-center mb-4 transition-transform hover:scale-110 duration-500">
                    <TrendingUp className="w-8 h-8 text-zinc-700" />
                </div>
                <h3 className="text-zinc-400 font-bold uppercase tracking-widest text-sm mb-2">{t("compare.empty_title") || "No Symbols Selected"}</h3>
                <p className="text-zinc-600 text-xs max-w-sm mx-auto">
                    {t("compare.empty")}
                </p>
            </div>
        );
    }

    return (
        <div className="w-full overflow-hidden rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl shadow-2xl">
            <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left text-sm whitespace-nowrap">
                    <thead className="bg-zinc-950/80 text-[10px] text-zinc-500 border-b border-white/5 font-black uppercase tracking-[0.2em]">
                        <tr>
                            <th className="px-6 py-4 sticky left-0 bg-zinc-950 z-10 w-48">{t("compare.symbol")}</th>
                            <th className="px-4 py-4 text-center">{t("compare.signal")}</th>
                            <th className="px-4 py-4 text-center">{t("compare.precision")}</th>
                            <th className="px-4 py-4 text-center text-indigo-400">Council</th>
                            <th className="px-4 py-4">{t("compare.rsi")}</th>
                            <th className="px-4 py-4">{t("compare.macd")}</th>
                            <th className="px-4 py-4">{t("compare.ema")}</th>
                            <th className="px-4 py-4">{t("compare.bb")}</th>
                            <th className="px-6 py-4 text-right">{t("compare.actions")}</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {renderItems.map(item => {
                            if (item.type === "success") {
                                const data = item.data!;
                                const stats = statsMap[item.symbol];
                                if (!stats) return null;

                                const signal = data.signal || "HOLD";
                                const isBuy = signal.toUpperCase().includes("BUY") || signal.toUpperCase() === "UP";
                                const isSell = signal.toUpperCase().includes("SELL") || signal.toUpperCase() === "DOWN";

                                return (
                                    <tr
                                        key={item.symbol}
                                        className="hover:bg-white/[0.02] transition-colors group cursor-pointer"
                                        onClick={() => onChart?.(item.symbol)}
                                    >
                                        <td className="px-6 py-4 sticky left-0 bg-zinc-950/90 group-hover:bg-zinc-900/90 z-10 backdrop-blur-md">
                                            <div className="flex items-center gap-4">
                                                <StockLogo symbol={item.symbol} logoUrl={data.fundamentals.logoUrl} size="md" />
                                                <div className="flex flex-col">
                                                    <span className="font-mono font-bold text-blue-400 text-sm tracking-tight">{item.symbol}</span>
                                                    <span className="text-[10px] text-zinc-500 truncate max-w-[140px] font-medium uppercase tracking-tighter">{data.fundamentals.name}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4 text-center">
                                            <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border ${isBuy ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : isSell ? "bg-red-500/10 border-red-500/20 text-red-400" : "bg-zinc-500/10 border-zinc-500/20 text-zinc-400"}`}>
                                                {isBuy ? <TrendingUp className="w-3 h-3" /> : isSell ? <TrendingDown className="w-3 h-3" /> : <Info className="w-3 h-3" />}
                                                <span className="text-[10px] font-bold uppercase tracking-wider">{isBuy ? t("signal.up") : isSell ? t("signal.down") : "HOLD"}</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4 text-center">
                                            <div className="flex flex-col items-center gap-1">
                                                <div className="relative flex items-center justify-center">
                                                    <span className={`text-sm font-black ${data.precision > 0.7 ? "text-green-400" : "text-yellow-400"}`}>
                                                        {(data.precision * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <span className="text-[9px] uppercase tracking-widest font-bold text-zinc-600">AI Score</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4 text-center">
                                            <div className="flex flex-col items-center gap-1">
                                                <span className="text-sm font-black text-white italic">{(data as any).councilScore ? `${(data as any).councilScore.toFixed(1)}%` : "N/A"}</span>
                                                <span className="text-[9px] uppercase tracking-widest font-black text-indigo-500/60">Council</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4">
                                            <StatCell stats={stats.rsi} />
                                        </td>
                                        <td className="px-4 py-4">
                                            <StatCell stats={stats.macd} />
                                        </td>
                                        <td className="px-4 py-4">
                                            <StatCell stats={stats.ema} />
                                        </td>
                                        <td className="px-4 py-4">
                                            <StatCell stats={stats.bb} />
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="flex items-center justify-end gap-2" onClick={(e) => e.stopPropagation()}>
                                                {onSave && (
                                                    <button
                                                        onClick={() => onSave(item.symbol)}
                                                        className={`p-2 rounded-xl transition-all ${isSaved?.(item.symbol)
                                                            ? 'text-yellow-400 bg-yellow-500/10'
                                                            : 'text-zinc-600 hover:text-yellow-400 hover:bg-yellow-500/10'
                                                            }`}
                                                        title={isSaved?.(item.symbol) ? t("watchlist.remove") || 'Remove from Watchlist' : t("compare.save")}
                                                    >
                                                        <Star className={`h-4 w-4 ${isSaved?.(item.symbol) ? 'fill-yellow-400' : ''}`} />
                                                    </button>
                                                )}
                                                <button
                                                    onClick={() => onRemove(item.symbol)}
                                                    className="p-2 rounded-xl text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-all"
                                                    title="Remove"
                                                >
                                                    <X className="h-4 w-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                );
                            } else if (item.type === "loading") {
                                return (
                                    <tr key={item.symbol} className="bg-white/[0.01]">
                                        <td className="px-6 py-6 sticky left-0 bg-zinc-950/80 z-10">
                                            <div className="flex items-center gap-3">
                                                <div className="relative">
                                                    <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                                                    <div className="absolute inset-0 blur-sm bg-blue-500/30 animate-pulse rounded-full" />
                                                </div>
                                                <span className="font-mono text-zinc-400 font-bold">{item.symbol}</span>
                                            </div>
                                        </td>
                                        <td colSpan={7} className="px-4 py-6 text-xs text-zinc-500 font-medium italic">
                                            <div className="flex items-center gap-2">
                                                <div className="h-1 w-24 bg-zinc-800 rounded-full overflow-hidden">
                                                    <div className="h-full bg-blue-500 animate-progress" style={{ width: "60%" }} />
                                                </div>
                                                {t("compare.fetching")}
                                            </div>
                                        </td>
                                    </tr>
                                );
                            } else {
                                return (
                                    <tr key={item.symbol} className="bg-red-500/[0.02]">
                                        <td className="px-6 py-5 sticky left-0 bg-zinc-950/90 z-10 border-l-2 border-red-500/30">
                                            <div className="flex items-center gap-2">
                                                <AlertCircle className="w-3.5 h-3.5 text-red-500/50" />
                                                <span className="font-mono text-zinc-400 font-medium">{item.symbol}</span>
                                            </div>
                                        </td>
                                        <td colSpan={6} className="px-4 py-5 text-xs text-red-500/50 font-medium">
                                            <span className="flex items-center gap-2">
                                                {item.error || "Analysis failed for this ticker."}
                                            </span>
                                        </td>
                                        <td className="px-6 py-5 text-right">
                                            <button
                                                onClick={() => onRemove(item.symbol)}
                                                className="p-2 rounded-xl text-zinc-700 hover:text-red-400 hover:bg-red-500/10 transition-all"
                                            >
                                                <X className="h-4 w-4" />
                                            </button>
                                        </td>
                                    </tr>
                                );
                            }
                        })}
                    </tbody>
                </table>
            </div>

            <div className="bg-zinc-950/60 backdrop-blur-md px-6 py-3 text-[9px] text-zinc-500 border-t border-white/5 flex items-center justify-between font-bold uppercase tracking-widest">
                <div className="flex items-center gap-3">
                    <Info className="h-3.5 w-3.5 text-blue-500/50" />
                    <span>{t("compare.winrate_info")}</span>
                </div>
                <div className="flex items-center gap-4 text-zinc-600">
                    <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-green-500" /> {t("signal.up")}</span>
                    <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-red-500" /> {t("signal.down")}</span>
                </div>
            </div>

            <style jsx>{`
                @keyframes progress {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
                .animate-progress {
                    animation: progress 2s infinite linear;
                }
            `}</style>
        </div>
    );
}
