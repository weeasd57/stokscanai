import { TrendingUp } from "lucide-react";
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar } from "recharts";
import { ModelStats } from "./types";

interface ModelStatsChartProps {
    selectedModelStats: ModelStats[];
}

export default function ModelStatsChart({ selectedModelStats }: ModelStatsChartProps) {
    return (
        <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
            <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center gap-2 mb-3">
                <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Model Stats
            </div>
            {selectedModelStats.length > 0 ? (
                <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={selectedModelStats} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                            <XAxis
                                dataKey="name"
                                stroke="rgba(255,255,255,0.4)"
                                tick={{ fontSize: 10 }}
                                angle={-20}
                                textAnchor="end"
                                height={40}
                            />
                            <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 10 }} allowDecimals={false} />
                            <Tooltip
                                contentStyle={{ background: "#0a0a0a", border: "1px solid rgba(255,255,255,0.08)", color: "#fff" }}
                            />
                            <Bar dataKey="trainingSamples" fill="#38bdf8" name="Samples" />
                            <Bar dataKey="numFeatures" fill="#a855f7" name="Features" />
                            <Bar dataKey="nEstimators" fill="#34d399" name="Estimators" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <div className="h-56 flex items-center justify-center text-xs text-zinc-500">
                    Select a model to view stats.
                </div>
            )}
        </div>
    );
}
