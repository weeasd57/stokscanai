"use client";

import { useMemo } from "react";
import { TrendingUp, TrendingDown, BarChart2 } from "lucide-react";
import type { PredictResponse } from "@/lib/types";
import { calculateIndicatorStats } from "@/lib/indicators";

interface IndicatorStatisticsProps {
    results: Record<string, PredictResponse>;
}

interface AggregatedStats {
    buySignals: number;
    sellSignals: number;
    buyWins: number;
    buyWinRate: string;
}

export default function IndicatorStatistics({ results }: IndicatorStatisticsProps) {
    const aggregatedStats = useMemo(() => {
        const symbols = Object.keys(results);
        if (symbols.length === 0) return null;

        const aggregate = {
            rsi: { buySignals: 0, sellSignals: 0, buyWins: 0 },
            macd: { buySignals: 0, sellSignals: 0, buyWins: 0 },
            ema: { buySignals: 0, sellSignals: 0, buyWins: 0 },
            bb: { buySignals: 0, sellSignals: 0, buyWins: 0 },
        };

        symbols.forEach(symbol => {
            const data = results[symbol];
            if (!data?.testPredictions) return;

            const stats = calculateIndicatorStats(data.testPredictions);

            // RSI
            aggregate.rsi.buySignals += parseInt(stats.rsi.buySignals.toString());
            aggregate.rsi.sellSignals += parseInt(stats.rsi.sellSignals.toString());
            const rsiBuyWinRate = parseFloat(stats.rsi.buyWinRate.toString());
            aggregate.rsi.buyWins += Math.round((parseInt(stats.rsi.buySignals.toString()) * rsiBuyWinRate) / 100);

            // MACD
            aggregate.macd.buySignals += parseInt(stats.macd.buySignals.toString());
            aggregate.macd.sellSignals += parseInt(stats.macd.sellSignals.toString());
            const macdBuyWinRate = parseFloat(stats.macd.buyWinRate.toString());
            aggregate.macd.buyWins += Math.round((parseInt(stats.macd.buySignals.toString()) * macdBuyWinRate) / 100);

            // EMA Cross
            aggregate.ema.buySignals += parseInt(stats.ema.buySignals.toString());
            aggregate.ema.sellSignals += parseInt(stats.ema.sellSignals.toString());
            const emaBuyWinRate = parseFloat(stats.ema.buyWinRate.toString());
            aggregate.ema.buyWins += Math.round((parseInt(stats.ema.buySignals.toString()) * emaBuyWinRate) / 100);

            // Bollinger Bands
            aggregate.bb.buySignals += parseInt(stats.bb.buySignals.toString());
            aggregate.bb.sellSignals += parseInt(stats.bb.sellSignals.toString());
            const bbBuyWinRate = parseFloat(stats.bb.buyWinRate.toString());
            aggregate.bb.buyWins += Math.round((parseInt(stats.bb.buySignals.toString()) * bbBuyWinRate) / 100);
        });

        const calculateWinRate = (wins: number, total: number): string => {
            if (total === 0) return "0.0";
            return ((wins / total) * 100).toFixed(1);
        };

        return {
            rsi: {
                buySignals: aggregate.rsi.buySignals,
                sellSignals: aggregate.rsi.sellSignals,
                buyWins: aggregate.rsi.buyWins,
                buyWinRate: calculateWinRate(aggregate.rsi.buyWins, aggregate.rsi.buySignals),
            },
            macd: {
                buySignals: aggregate.macd.buySignals,
                sellSignals: aggregate.macd.sellSignals,
                buyWins: aggregate.macd.buyWins,
                buyWinRate: calculateWinRate(aggregate.macd.buyWins, aggregate.macd.buySignals),
            },
            ema: {
                buySignals: aggregate.ema.buySignals,
                sellSignals: aggregate.ema.sellSignals,
                buyWins: aggregate.ema.buyWins,
                buyWinRate: calculateWinRate(aggregate.ema.buyWins, aggregate.ema.buySignals),
            },
            bb: {
                buySignals: aggregate.bb.buySignals,
                sellSignals: aggregate.bb.sellSignals,
                buyWins: aggregate.bb.buyWins,
                buyWinRate: calculateWinRate(aggregate.bb.buyWins, aggregate.bb.buySignals),
            },
        };
    }, [results]);

    if (!aggregatedStats) return null;

    const indicators = [
        { name: "RSI", stats: aggregatedStats.rsi, color: "blue" },
        { name: "MACD", stats: aggregatedStats.macd, color: "purple" },
        { name: "EMA Cross", stats: aggregatedStats.ema, color: "orange" },
        { name: "Bollinger", stats: aggregatedStats.bb, color: "pink" },
    ];

    const getWinRateColor = (winRate: string) => {
        const rate = parseFloat(winRate);
        if (rate >= 70) return "text-emerald-400";
        if (rate >= 50) return "text-yellow-400";
        return "text-red-400";
    };

    const getColorClasses = (color: string) => {
        const colors: Record<string, { bg: string; border: string; text: string }> = {
            blue: { bg: "bg-blue-600/10", border: "border-blue-500/20", text: "text-blue-400" },
            purple: { bg: "bg-purple-600/10", border: "border-purple-500/20", text: "text-purple-400" },
            orange: { bg: "bg-orange-600/10", border: "border-orange-500/20", text: "text-orange-400" },
            pink: { bg: "bg-pink-600/10", border: "border-pink-500/20", text: "text-pink-400" },
        };
        return colors[color] || colors.blue;
    };

    return (
        <div className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 p-8 shadow-2xl backdrop-blur-xl overflow-hidden relative">
            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2" />

            <div className="flex items-center gap-4 mb-8 relative z-10">
                <div className="p-3 rounded-2xl bg-blue-600/20 border border-blue-500/30 shadow-xl shadow-blue-600/10">
                    <BarChart2 className="h-6 w-6 text-blue-400" />
                </div>
                <div className="space-y-1">
                    <h2 className="text-2xl font-black text-white uppercase tracking-tight italic">
                        Indicator Statistics
                    </h2>
                    <p className="text-[10px] font-black text-blue-400/60 uppercase tracking-[0.3em]">
                        Aggregated Performance Across All Symbols
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">
                <div className="lg:col-span-4 flex flex-wrap gap-4 mb-4">
                    <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 flex items-center gap-2">
                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Symbols:</span>
                        <span className="text-zinc-200 font-bold">{Object.keys(results).length}</span>
                    </div>
                    <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 flex items-center gap-2">
                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Avg Days:</span>
                        <span className="text-zinc-200 font-bold">
                            {Math.round(Object.values(results).reduce((acc, curr) => acc + (curr.testPredictions?.length || 0), 0) / (Object.keys(results).length || 1))}
                        </span>
                    </div>
                </div>
                {indicators.map((indicator) => {
                    const colors = getColorClasses(indicator.color);
                    return (
                        <div
                            key={indicator.name}
                            className={`rounded-2xl border ${colors.border} ${colors.bg} p-6 space-y-4 transition-all hover:scale-105 duration-300`}
                        >
                            <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                {indicator.name}
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between text-xs">
                                    <span className="flex items-center gap-2 text-zinc-400">
                                        <TrendingUp className="h-3 w-3 text-emerald-500" />
                                        <span className="font-medium">Buy Signals:</span>
                                    </span>
                                    <span className={`font-bold ${colors.text}`}>{indicator.stats.buySignals}</span>
                                </div>

                                <div className="flex items-center justify-between text-xs">
                                    <span className="flex items-center gap-2 text-zinc-400">
                                        <TrendingDown className="h-3 w-3 text-red-500" />
                                        <span className="font-medium">Sell Signals:</span>
                                    </span>
                                    <span className="font-bold text-zinc-300">{indicator.stats.sellSignals}</span>
                                </div>

                                <div className="pt-2 border-t border-white/5">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                            Buy Win Rate
                                        </span>
                                        <span className={`text-lg font-black ${getWinRateColor(indicator.stats.buyWinRate)}`}>
                                            {indicator.stats.buyWinRate}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
