"use client";

import { useState, useMemo } from "react";
import {
    Search, Filter, Calendar, CheckCircle,
    TrendingUp, TrendingDown, Minus, BarChart3, ArrowLeftRight
} from "lucide-react";
import type { TestPredictionRow } from "@/lib/types";
import { useAppState } from "@/contexts/AppStateContext";
import { useLanguage } from "@/contexts/LanguageContext";

import { getIndicatorSignals, calculateIndicatorStats, type IndicatorSignals } from "@/lib/indicators";

function formatNum(v?: number | null) {
    if (v === undefined || v === null) return "-";
    return v.toFixed(2);
}

function SignalBadge({ signal, label }: { signal: "buy" | "sell" | "neutral"; label: string }) {
    if (signal === "buy") {
        return (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-green-500/20 text-green-400 border border-green-500/30">
                <TrendingUp className="h-2.5 w-2.5" />
                {label}
            </span>
        );
    }
    if (signal === "sell") {
        return (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">
                <TrendingDown className="h-2.5 w-2.5" />
                {label}
            </span>
        );
    }
    return (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-zinc-800 text-zinc-500 border border-zinc-700">
            <Minus className="h-2.5 w-2.5" />
            -
        </span>
    );
}

export default function TableView({ rows, ticker }: { rows: TestPredictionRow[], ticker?: string }) {
    const [last60Days, setLast60Days] = useState(true);
    const [showBuyOnly, setShowBuyOnly] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [showStats, setShowStats] = useState(true);
    const { addSymbolToCompare } = useAppState();
    const { t } = useLanguage();

    const filteredRows = useMemo(() => {
        // Sort by date desc
        let data = [...rows].reverse();

        if (last60Days) {
            data = data.slice(0, 60);
        }

        if (showBuyOnly) {
            data = data.filter((r) => r.pred === 1);
        }

        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            data = data.filter((r) => r.date.includes(q));
        }

        return data;
    }, [rows, last60Days, showBuyOnly, searchQuery]);

    // Calculate statistics for indicators
    const stats = useMemo(() => calculateIndicatorStats(rows), [rows]);

    return (
        <div className="w-full space-y-4">
            {/* Statistics Panel */}
            {showStats && (
                <div className="rounded-xl border border-zinc-800 bg-gradient-to-br from-zinc-900 to-zinc-950 p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <BarChart3 className="h-4 w-4 text-blue-400" />
                        <span className="text-sm font-semibold text-zinc-200">Indicator Statistics</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {/* RSI Stats */}
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                            <div className="text-xs font-medium text-purple-400 mb-2">RSI</div>
                            <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-green-400">Buy Signals:</span>
                                    <span className="text-zinc-200">{stats.rsi.buySignals}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-red-400">Sell Signals:</span>
                                    <span className="text-zinc-200">{stats.rsi.sellSignals}</span>
                                </div>
                                <div className="flex justify-between text-zinc-400">
                                    <span>Buy Win Rate:</span>
                                    <span className="text-green-400">{stats.rsi.buyWinRate}%</span>
                                </div>
                            </div>
                        </div>

                        {/* MACD Stats */}
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                            <div className="text-xs font-medium text-blue-400 mb-2">MACD</div>
                            <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-green-400">Buy Signals:</span>
                                    <span className="text-zinc-200">{stats.macd.buySignals}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-red-400">Sell Signals:</span>
                                    <span className="text-zinc-200">{stats.macd.sellSignals}</span>
                                </div>
                                <div className="flex justify-between text-zinc-400">
                                    <span>Buy Win Rate:</span>
                                    <span className="text-green-400">{stats.macd.buyWinRate}%</span>
                                </div>
                            </div>
                        </div>

                        {/* EMA Stats */}
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                            <div className="text-xs font-medium text-orange-400 mb-2">EMA Cross</div>
                            <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-green-400">Buy Signals:</span>
                                    <span className="text-zinc-200">{stats.ema.buySignals}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-red-400">Sell Signals:</span>
                                    <span className="text-zinc-200">{stats.ema.sellSignals}</span>
                                </div>
                                <div className="flex justify-between text-zinc-400">
                                    <span>Buy Win Rate:</span>
                                    <span className="text-green-400">{stats.ema.buyWinRate}%</span>
                                </div>
                            </div>
                        </div>

                        {/* BB Stats */}
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                            <div className="text-xs font-medium text-cyan-400 mb-2">Bollinger</div>
                            <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-green-400">Buy Signals:</span>
                                    <span className="text-zinc-200">{stats.bb.buySignals}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-red-400">Sell Signals:</span>
                                    <span className="text-zinc-200">{stats.bb.sellSignals}</span>
                                </div>
                                <div className="flex justify-between text-zinc-400">
                                    <span>Buy Win Rate:</span>
                                    <span className="text-green-400">{stats.bb.buyWinRate}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Filtering Toolbar */}
            <div className="flex flex-wrap items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900 p-3">
                <div className="flex items-center gap-2">
                    <Filter className="h-4 w-4 text-zinc-400" />
                    <span className="text-sm font-medium text-zinc-300">Filters:</span>
                </div>

                <button
                    onClick={() => setLast60Days(!last60Days)}
                    className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${last60Days ? "bg-blue-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                        }`}
                >
                    <Calendar className="h-3.5 w-3.5" />
                    Last 60 Days
                </button>

                <button
                    onClick={() => setShowBuyOnly(!showBuyOnly)}
                    className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${showBuyOnly ? "bg-green-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                        }`}
                >
                    <CheckCircle className="h-3.5 w-3.5" />
                    Buy Signals Only
                </button>

                <button
                    onClick={() => setShowStats(!showStats)}
                    className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${showStats ? "bg-purple-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                        }`}
                >
                    <BarChart3 className="h-3.5 w-3.5" />
                    Statistics
                </button>

                {ticker && (
                    <button
                        onClick={() => {
                            void addSymbolToCompare(ticker);
                        }}
                        className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                    >
                        <ArrowLeftRight className="h-3.5 w-3.5" />
                        {t("nav.scanner.compare")}
                    </button>
                )}

                <div className="ml-auto flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5">
                    <Search className="h-3.5 w-3.5 text-zinc-500" />
                    <input
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search dates..."
                        className="w-32 bg-transparent text-xs text-zinc-200 outline-none placeholder:text-zinc-600"
                    />
                </div>
            </div>

            {/* Scrollable Table Container */}
            <div className="w-full overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-xl">
                <div className="overflow-x-auto pb-2">
                    <table className="w-full min-w-[1200px] border-collapse text-xs">
                        <thead>
                            {/* Grouped Headers */}
                            <tr className="bg-yellow-400 text-black font-bold border-b border-zinc-500">
                                <th className="border-r border-zinc-500 p-1 text-center" colSpan={5}>MARKET DATA</th>
                                <th className="border-r border-zinc-500 p-1 text-center bg-blue-300" colSpan={4}>AI SIGNAL</th>
                                <th className="border-r border-zinc-500 p-1 text-center bg-green-300" colSpan={4}>INDICATOR SIGNALS</th>
                                <th className="p-1 text-center bg-gray-300" colSpan={5}>VALUES</th>
                            </tr>
                            <tr className="bg-zinc-800 text-zinc-100 font-bold border-b border-zinc-600">
                                <th className="px-3 py-2 border-r border-zinc-600 whitespace-nowrap">Date</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">Open</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">High</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">Low</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right bg-blue-900/30 whitespace-nowrap">Close</th>

                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap">AI</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap">Council</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap">Consensus</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap">Result</th>

                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap bg-purple-900/20">RSI</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap bg-blue-900/20">MACD</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap bg-orange-900/20">EMA</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-center whitespace-nowrap bg-cyan-900/20">BB</th>

                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">RSI</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">EMA50</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">EMA200</th>
                                <th className="px-3 py-2 border-r border-zinc-600 text-right whitespace-nowrap">MACD</th>
                                <th className="px-3 py-2 text-right whitespace-nowrap">Change%</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-700 font-mono">
                            {filteredRows.map((r, i) => {
                                const originalIndex = rows.indexOf(r);
                                const prevRow = originalIndex > 0 ? rows[originalIndex - 1] : undefined;
                                const signals = getIndicatorSignals(r, prevRow);

                                const itemsClass = "px-3 py-1.5 border-r border-zinc-700 whitespace-nowrap";
                                const isBuy = r.pred === 1;
                                const isTargetUp = r.target === 1;

                                // Change calculation
                                let changePct = 0;
                                if (prevRow?.close) {
                                    changePct = (r.close - prevRow.close) / prevRow.close * 100;
                                }

                                return (
                                    <tr key={r.date} className="hover:bg-zinc-800 transition-colors bg-zinc-900/50 even:bg-zinc-900">
                                        <td className={`${itemsClass} text-zinc-300 font-medium`}>{r.date}</td>
                                        <td className={`${itemsClass} text-zinc-400 text-right`}>{formatNum(r.open)}</td>
                                        <td className={`${itemsClass} text-zinc-400 text-right`}>{formatNum(r.high)}</td>
                                        <td className={`${itemsClass} text-zinc-400 text-right`}>{formatNum(r.low)}</td>
                                        <td className={`${itemsClass} text-zinc-100 font-bold text-right bg-blue-950/30`}>{r.close.toFixed(2)}</td>

                                        {/* AI Signal */}
                                        <td className={`${itemsClass} text-center font-bold text-xs ${isBuy ? "bg-green-600 text-white" : "bg-zinc-800 text-zinc-500"}`}>
                                            {isBuy ? "BUY" : "HOLD"}
                                        </td>

                                        <td className={`${itemsClass} text-center font-bold text-white italic whitespace-nowrap`}>
                                            {r.councilScore ? `${r.councilScore.toFixed(1)}%` : '-'}
                                        </td>
                                        <td className={`${itemsClass} text-center font-bold text-indigo-400 uppercase tracking-widest whitespace-nowrap`}>
                                            {r.consensusRatio || '-'}
                                        </td>

                                        {/* Target (Actual Result) */}
                                        <td className={`${itemsClass} text-center font-bold text-xs ${isTargetUp ? "text-green-400" : "text-red-400"}`}>
                                            {isTargetUp ? "UP" : "DOWN"}
                                        </td>

                                        {/* Indicator Signals */}
                                        <td className={`${itemsClass} text-center`}>
                                            <SignalBadge signal={signals.rsiSignal} label="RSI" />
                                        </td>
                                        <td className={`${itemsClass} text-center`}>
                                            <SignalBadge signal={signals.macdSignal} label="MACD" />
                                        </td>
                                        <td className={`${itemsClass} text-center`}>
                                            <SignalBadge signal={signals.emaSignal} label="EMA" />
                                        </td>
                                        <td className={`${itemsClass} text-center`}>
                                            <SignalBadge signal={signals.bbSignal} label="BB" />
                                        </td>

                                        {/* Values */}
                                        <td className={`${itemsClass} text-right ${(r.rsi || 50) > 70 ? "text-red-400 font-bold" : (r.rsi || 50) < 30 ? "text-green-400 font-bold" : "text-zinc-400"}`}>
                                            {formatNum(r.rsi)}
                                        </td>
                                        <td className={`${itemsClass} text-orange-300 text-right`}>{formatNum(r.ema50)}</td>
                                        <td className={`${itemsClass} text-purple-300 text-right`}>{formatNum(r.ema200)}</td>
                                        <td className={`${itemsClass} text-zinc-400 text-right`}>{formatNum(r.macd)}</td>
                                        <td className={`px-3 py-1.5 text-right font-bold ${changePct >= 0 ? "text-green-400" : "text-red-400"}`}>
                                            {changePct > 0 ? "+" : ""}{changePct.toFixed(2)}%
                                        </td>
                                    </tr>
                                );
                            })}
                            {filteredRows.length === 0 && (
                                <tr>
                                    <td colSpan={16} className="py-8 text-center text-zinc-500">
                                        No data matches your filters.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
