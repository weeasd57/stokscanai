import { LineChart, ChevronUp, ChevronDown } from "lucide-react";
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from "recharts";
import { ModelSummary, SortConfig } from "./types";
import { calculateClassification } from "./utils";

interface ResultsComparisonProps {
    sortedMultiSummaries: ModelSummary[];
    multiClassificationChart: any[];
    toggleSort: (key: string) => void;
    sortConfig: SortConfig | null;
}

export default function ResultsComparison({
    sortedMultiSummaries,
    multiClassificationChart,
    toggleSort,
    sortConfig,
}: ResultsComparisonProps) {
    const SortIcon = ({ column }: { column: string }) => {
        if (sortConfig?.key !== column) return <div className="w-3 h-3 ml-1 opacity-20" />;
        return sortConfig.direction === "asc" ? (
            <ChevronUp className="w-3 h-3 ml-1" />
        ) : (
            <ChevronDown className="w-3 h-3 ml-1" />
        );
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-xs sm:text-sm font-black uppercase tracking-[0.3em] text-zinc-500">
                    <LineChart className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-400 inline mr-2" /> Results Comparison ({sortedMultiSummaries.length} Models)
                </h3>
            </div>

            {/* Comparative win-rate chart */}
            <div className="bg-zinc-950/40 rounded-xl border border-white/5 p-4 mb-4">
                <h4 className="text-[10px] sm:text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-3">
                    Win Rate by Model
                </h4>
                <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={sortedMultiSummaries.map((item) => ({
                                name: item.modelName.replace(/^model_|\.pkl$/gi, ""),
                                winRate: item.kpis.winRate,
                                buyAcc: item.kpis.buyAccuracy,
                                sellAcc: item.kpis.sellAccuracy,
                            }))}
                            margin={{ top: 10, right: 20, left: 0, bottom: 30 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                            <XAxis
                                dataKey="name"
                                stroke="rgba(255,255,255,0.4)"
                                tick={{ fontSize: 10 }}
                                angle={-20}
                                textAnchor="end"
                                height={40}
                            />
                            <YAxis
                                stroke="rgba(255,255,255,0.4)"
                                tick={{ fontSize: 10 }}
                                tickFormatter={(v) => `${v.toFixed(0)}%`}
                            />
                            <Tooltip
                                formatter={(value: any, key: any) =>
                                    typeof value === "number" ? `${value.toFixed(1)}%` : value
                                }
                                contentStyle={{
                                    backgroundColor: "rgba(9,9,11,0.98)",
                                    border: "1px solid rgba(99,102,241,0.3)",
                                    borderRadius: 10,
                                    padding: "8px 12px",
                                }}
                            />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Bar dataKey="winRate" name="Win Rate" fill="#6366f1" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="buyAcc" name="Buy Acc" fill="#22c55e" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="sellAcc" name="Sell Acc" fill="#f97373" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Detailed Results Table */}
            <div className="bg-zinc-950/40 rounded-xl border border-white/5 overflow-hidden">
                <div className="overflow-x-auto custom-scrollbar">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-white/5">
                                {[
                                    { label: "Model", key: "name" },
                                    { label: "Win Rate", key: "winRate" },
                                    { label: "Buy Acc", key: "buyAcc" },
                                    { label: "Sell Acc", key: "sellAcc" },
                                    { label: "Prec.", key: "precision" },
                                    { label: "Earn", key: "earnRate" },
                                    { label: "Recall", key: "recall" },
                                    { label: "F1", key: "f1" },
                                    { label: "Time", key: "execTime" },
                                ].map((col) => (
                                    <th
                                        key={col.key}
                                        onClick={() => toggleSort(col.key)}
                                        className="px-4 py-3 text-[10px] font-black uppercase tracking-widest text-zinc-500 cursor-pointer hover:bg-white/5 transition-colors"
                                    >
                                        <div className="flex items-center">
                                            {col.label}
                                            <SortIcon column={col.key} />
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {sortedMultiSummaries.map((item) => {
                                const cls = calculateClassification(item.result.testPredictions || []);
                                const execTime = (item.result as any).executionTime;
                                return (
                                    <tr key={item.modelName} className="hover:bg-white/5 transition-colors group">
                                        <td className="px-4 py-3 text-xs font-bold text-zinc-300 truncate max-w-[150px] font-mono group-hover:text-indigo-400">
                                            {item.modelName.replace(/^model_|\.pkl$/gi, "")}
                                        </td>
                                        <td className="px-4 py-3 text-xs font-bold text-indigo-400">
                                            {item.kpis.winRate.toFixed(1)}%
                                        </td>
                                        <td className="px-4 py-3 text-xs font-bold text-emerald-400">
                                            {item.kpis.buyAccuracy.toFixed(1)}%
                                        </td>
                                        <td className="px-4 py-3 text-xs font-bold text-rose-400">
                                            {item.kpis.sellAccuracy.toFixed(1)}%
                                        </td>
                                        <td className="px-4 py-3 text-xs text-zinc-400">
                                            {(cls.precisionBuy * 100).toFixed(1)}%
                                        </td>
                                        <td
                                            className={`px-4 py-3 text-xs font-bold ${item.result.earnPercentage != null
                                                    ? item.result.earnPercentage >= 0
                                                        ? "text-emerald-400"
                                                        : "text-rose-400"
                                                    : "text-zinc-500"
                                                }`}
                                        >
                                            {item.result.earnPercentage != null
                                                ? `${item.result.earnPercentage >= 0 ? "+" : ""}${item.result.earnPercentage.toFixed(1)}%`
                                                : "-"}
                                        </td>
                                        <td className="px-4 py-3 text-xs text-zinc-400">{(cls.recallBuy * 100).toFixed(1)}%</td>
                                        <td className="px-4 py-3 text-xs text-zinc-400">{(cls.f1Buy * 100).toFixed(1)}%</td>
                                        <td className="px-4 py-3 text-xs text-zinc-500 italic">
                                            {execTime ? `${execTime}ms` : "-"}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="bg-zinc-950/40 rounded-xl border border-white/5 p-4 mb-4">
                <h4 className="text-[10px] sm:text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-3">
                    Classification (BUY) by Model
                </h4>
                <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={multiClassificationChart} margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                            <XAxis
                                dataKey="name"
                                stroke="rgba(255,255,255,0.4)"
                                tick={{ fontSize: 10 }}
                                angle={-20}
                                textAnchor="end"
                                height={40}
                            />
                            <YAxis
                                stroke="rgba(255,255,255,0.4)"
                                tick={{ fontSize: 10 }}
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
            </div>
        </div>
    );
}
