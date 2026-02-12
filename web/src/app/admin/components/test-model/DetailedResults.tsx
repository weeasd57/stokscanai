
import { TrendingUp, TrendingDown, LineChart, BarChart3, CircleDot, Gauge, LayoutDashboard } from "lucide-react";
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from "recharts";
import { Button } from "@/components/ui/button";
import TestModelCandleChart from "../TestModelCandleChart";
import { PredictResponse } from "@/lib/types";
import { calculateClassification } from "./utils";

interface DetailedResultsProps {
    testResult: PredictResponse;
    predictionsWithOutcome: any[];
    showBUYSignals: boolean;
    setShowBUYSignals: (v: boolean) => void;
    showSELLSignals: boolean;
    setShowSELLSignals: (v: boolean) => void;
    showVolume: boolean;
    setShowVolume: (v: boolean) => void;
    showSMA50: boolean;
    setShowSMA50: (v: boolean) => void;
    showSMA200: boolean;
    setShowSMA200: (v: boolean) => void;
    showEMA50: boolean;
    setShowEMA50: (v: boolean) => void;
    showEMA200: boolean;
    setShowEMA200: (v: boolean) => void;
    showBB: boolean;
    setShowBB: (v: boolean) => void;
    showRSI: boolean;
    setShowRSI: (v: boolean) => void;
    showMACD: boolean;
    setShowMACD: (v: boolean) => void;
    kpis: any;
}

export default function DetailedResults({
    testResult,
    predictionsWithOutcome,
    showBUYSignals,
    setShowBUYSignals,
    showSELLSignals,
    setShowSELLSignals,
    showVolume,
    setShowVolume,
    showSMA50,
    setShowSMA50,
    showSMA200,
    setShowSMA200,
    showEMA50,
    setShowEMA50,
    showEMA200,
    setShowEMA200,
    showBB,
    setShowBB,
    showRSI,
    setShowRSI,
    showMACD,
    setShowMACD,
    kpis,
}: DetailedResultsProps) {
    const signal = testResult.testPredictions?.[testResult.testPredictions.length - 1]?.pred ?? 1;

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3 sm:gap-4">
                <div className="flex items-center gap-2 text-xs sm:text-sm font-bold uppercase tracking-[0.2em] text-zinc-400">
                    <LineChart className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-400" /> Detailed Results
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                    <div className="flex flex-wrap items-center gap-2 text-[9px] sm:text-[10px] uppercase tracking-[0.2em] text-zinc-600">
                        <div>
                            Precision: <span className="text-emerald-400">{(testResult.precision * 100).toFixed(1)}%</span>
                        </div>
                        <div className="hidden sm:inline">
                            Close: <span className="text-zinc-300">${testResult.lastClose?.toFixed?.(2)}</span>
                        </div>
                    </div>
                    <div
                        className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm tracking-wide ${signal === 1
                            ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                            : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                            }`}
                    >
                        {signal === 1 ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                        {signal === 1 ? "BUY" : "SELL"} ({kpis.winRate.toFixed(1)}%)
                    </div>
                </div>
            </div>

            {/* KPI Grid */}
            <div className="grid gap-3 sm:gap-4 grid-cols-2 md:grid-cols-4 lg:grid-cols-5">
                <div className="bg-zinc-900/50 rounded-lg p-3 border border-white/5">
                    <div className="text-[9px] text-zinc-500 uppercase tracking-wider">Tests</div>
                    <div className="text-lg font-bold text-white mt-1">{kpis.totalTests}</div>
                </div>
                <div className="bg-emerald-500/10 rounded-lg p-3 border border-emerald-500/20">
                    <div className="text-[9px] text-emerald-400 uppercase tracking-wider">Buy</div>
                    <div className="text-lg font-bold text-emerald-400 mt-1">{kpis.buySignals}</div>
                </div>
                <div className="bg-rose-500/10 rounded-lg p-3 border border-rose-500/20">
                    <div className="text-[9px] text-rose-400 uppercase tracking-wider">Sell</div>
                    <div className="text-lg font-bold text-rose-400 mt-1">{kpis.sellSignals}</div>
                </div>
                <div className="bg-indigo-500/10 rounded-lg p-3 border border-indigo-500/20">
                    <div className="text-[9px] text-indigo-400 uppercase tracking-wider">Win Rate</div>
                    <div className="text-lg font-bold text-indigo-400 mt-1">{kpis.winRate.toFixed(1)}%</div>
                </div>
                <div
                    className={`rounded-lg p-3 border ${testResult.earnPercentage != null
                        ? testResult.earnPercentage >= 0
                            ? "bg-emerald-500/10 border-emerald-500/20"
                            : "bg-rose-500/10 border-rose-500/20"
                        : "bg-zinc-900/50 border-white/5"
                        }`}
                >
                    <div
                        className={`text-[9px] uppercase tracking-wider ${testResult.earnPercentage != null
                            ? testResult.earnPercentage >= 0
                                ? "text-emerald-400"
                                : "text-rose-400"
                            : "text-zinc-500"
                            }`}
                    >
                        Earn Rate
                    </div>
                    <div
                        className={`text-lg font-bold mt-1 ${testResult.earnPercentage != null
                            ? testResult.earnPercentage >= 0
                                ? "text-emerald-400"
                                : "text-rose-400"
                            : "text-white"
                            }`}
                    >
                        {testResult.earnPercentage != null
                            ? `${testResult.earnPercentage >= 0 ? "+" : ""}${testResult.earnPercentage.toFixed(1)}%`
                            : "-"}
                    </div>
                </div>
            </div>

            {/* Classification Performance */}
            <div className="bg-zinc-950/40 rounded-xl p-4 sm:p-6 border border-white/5">
                <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-4">Classification Performance</h3>
                {(() => {
                    const cls = calculateClassification(testResult.testPredictions || []);
                    const prData = [
                        { name: "BUY", precision: cls.precisionBuy * 100, recall: cls.recallBuy * 100, f1: cls.f1Buy * 100 },
                        { name: "SELL", precision: cls.precisionSell * 100, recall: cls.recallSell * 100, f1: cls.f1Sell * 100 },
                    ];
                    return (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="h-56">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={prData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                        <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 11 }} />
                                        <YAxis
                                            stroke="rgba(255,255,255,0.4)"
                                            tick={{ fontSize: 11 }}
                                            tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                                        />
                                        <Tooltip
                                            formatter={(value: any) => (typeof value === "number" ? `${value.toFixed(1)}%` : value)}
                                            contentStyle={{
                                                backgroundColor: "rgba(9,9,11,0.98)",
                                                border: "1px solid rgba(99,102,241,0.3)",
                                                borderRadius: 10,
                                                padding: "8px 12px",
                                            }}
                                        />
                                        <Legend wrapperStyle={{ fontSize: 10 }} />
                                        <Bar dataKey="precision" name="Precision" fill="#22c55e" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="recall" name="Recall" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="f1" name="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3">
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-400">TP (Correct BUY)</div>
                                    <div className="mt-1 text-2xl font-black text-emerald-300">{cls.tp}</div>
                                </div>
                                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3">
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-rose-400">FP (False BUY)</div>
                                    <div className="mt-1 text-2xl font-black text-rose-300">{cls.fp}</div>
                                </div>
                                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3">
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-amber-400">FN (Missed BUY)</div>
                                    <div className="mt-1 text-2xl font-black text-amber-300">{cls.fn}</div>
                                </div>
                                <div className="rounded-xl border border-sky-500/30 bg-sky-500/10 p-3">
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-sky-400">TN (Correct SELL)</div>
                                    <div className="mt-1 text-2xl font-black text-sky-300">{cls.tn}</div>
                                </div>
                            </div>
                        </div>
                    );
                })()}
            </div>

            {/* Chart Section */}
            <div className="bg-zinc-950/40 rounded-xl p-4 sm:p-6 border border-white/5">
                <div className="flex flex-col gap-4 mb-6">
                    <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500">Price & Signals Chart</h3>
                    <div className="flex flex-wrap gap-2">
                        <Button
                            variant={showBUYSignals ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowBUYSignals(!showBUYSignals)}
                            className="text-[10px]"
                        >
                            <TrendingUp className="w-3 h-3 text-emerald-400 mr-1" /> BUY
                        </Button>
                        <Button
                            variant={showVolume ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowVolume(!showVolume)}
                            className="text-[10px]"
                        >
                            <BarChart3 className="w-3 h-3 text-blue-400 mr-1" /> Volume
                        </Button>
                        <Button
                            variant={showSMA50 ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowSMA50(!showSMA50)}
                            className="text-[10px]"
                        >
                            SMA50
                        </Button>
                        <Button
                            variant={showSMA200 ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowSMA200(!showSMA200)}
                            className="text-[10px]"
                        >
                            SMA200
                        </Button>
                        <Button variant={showBB ? "default" : "outline"} size="sm" onClick={() => setShowBB(!showBB)} className="text-[10px]">
                            BB
                        </Button>
                        <Button
                            variant={showRSI ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowRSI(!showRSI)}
                            className="text-[10px]"
                        >
                            RSI
                        </Button>
                        <Button
                            variant={showMACD ? "default" : "outline"}
                            size="sm"
                            onClick={() => setShowMACD(!showMACD)}
                            className="text-[10px]"
                        >
                            MACD
                        </Button>
                    </div>
                </div>
                <TestModelCandleChart
                    rows={predictionsWithOutcome}
                    showBuySignals={showBUYSignals}
                    showSellSignals={showSELLSignals}
                    showSMA50={showSMA50}
                    showSMA200={showSMA200}
                    showBB={showBB}
                    showRSI={showRSI}
                    showMACD={showMACD}
                    showVolume={showVolume}
                />
            </div>
        </div>
    );
}
