"use client";

import { useState, useEffect } from "react";
import { Activity, TrendingUp, TrendingDown, RefreshCw, BarChart3, Globe, Loader2, Calendar } from "lucide-react";
import { getIndicatorDashboard } from "@/lib/api";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";

interface DashboardData {
    rsi: { buy_signals: number; sell_signals: number; win_rate: number };
    macd: { buy_signals: number; sell_signals: number; win_rate: number };
    ema: { buy_signals: number; sell_signals: number; win_rate: number };
    bb: { buy_signals: number; sell_signals: number; win_rate: number };
    scanned_count: number;
}

export default function StrategyDashboard() {
    const { t } = useLanguage();
    const { countries } = useAppState();
    const [country, setCountry] = useState("Egypt");
    const [days, setDays] = useState(60);
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await getIndicatorDashboard(country, 20, days);
            setData(result);
            setLastUpdated(new Date());
        } catch (err: any) {
            setError(err.message || "Failed to fetch dashboard data");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void fetchData();
    }, [country, days]);

    if (error) {
        return (
            <div className="p-4 rounded-2xl bg-red-500/5 border border-red-500/20 text-red-400 text-xs flex items-center justify-between">
                <span>{error}</span>
                <button onClick={() => void fetchData()} className="p-2 hover:bg-red-500/10 rounded-lg transition-colors">
                    <RefreshCw className="w-3.5 h-3.5" />
                </button>
            </div>
        );
    }

    const indicators = [
        { id: "rsi", name: "RSI (14)", color: "text-blue-400", bg: "bg-blue-500/10" },
        { id: "macd", name: "MACD", color: "text-purple-400", bg: "bg-purple-500/10" },
        { id: "ema", name: "EMA Cross", color: "text-amber-400", bg: "bg-amber-500/10" },
        { id: "bb", name: "Bollinger", color: "text-emerald-400", bg: "bg-emerald-500/10" },
    ];

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div className="space-y-1">
                    <h2 className="text-xl font-black text-white flex items-center gap-3 uppercase tracking-tight italic">
                        <div className="p-2 rounded-xl bg-indigo-600/20 border border-indigo-500/30">
                            <BarChart3 className="w-5 h-5 text-indigo-400" />
                        </div>
                        {t("dash.title")}
                    </h2>
                    <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-[0.2em]">
                        {t("dash.subtitle").replace("{country}", country)}
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                    {/* Time Window Selector */}
                    <div className="flex p-1 rounded-xl bg-zinc-900 border border-white/5">
                        {[30, 60, 90].map((d) => (
                            <button
                                key={d}
                                onClick={() => setDays(d)}
                                className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${days === d ? "bg-white text-zinc-950 shadow-lg" : "text-zinc-600 hover:text-zinc-400"}`}
                            >
                                {t(`dash.days.${d}`)}
                            </button>
                        ))}
                    </div>

                    <div className="h-6 w-px bg-white/5 mx-2 hidden md:block" />

                    <div className="flex items-center gap-3">
                        <div className="relative group">
                            <Globe className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500 group-hover:text-blue-500 transition-colors" />
                            <select
                                value={country}
                                onChange={(e) => setCountry(e.target.value)}
                                className="h-10 pl-9 pr-8 rounded-xl bg-zinc-900 border border-white/5 text-[11px] font-black uppercase tracking-widest text-zinc-300 outline-none hover:bg-zinc-800 transition-all appearance-none"
                            >
                                {countries.map(c => (
                                    <option key={c} value={c}>{c.toUpperCase()}</option>
                                ))}
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600 text-[8px]">▼</div>
                        </div>

                        <button
                            onClick={() => void fetchData()}
                            disabled={loading}
                            className="h-10 w-10 flex items-center justify-center rounded-xl bg-zinc-900 border border-white/5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-50"
                        >
                            {loading ? <Loader2 className="w-4 h-4 animate-spin text-indigo-500" /> : <RefreshCw className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {indicators.map((ind) => {
                    const stats = data ? (data as any)[ind.id] : null;
                    const wr = stats?.win_rate || 0;

                    return (
                        <div key={ind.id} className="relative overflow-hidden rounded-[2rem] border border-white/5 bg-zinc-950/40 p-6 backdrop-blur-xl group hover:border-indigo-500/30 transition-all duration-500">
                            {loading && <div className="absolute inset-0 bg-zinc-950/20 backdrop-blur-[1px] animate-pulse z-10" />}

                            <div className="flex items-center justify-between mb-6">
                                <div className={`p-2.5 rounded-xl ${ind.bg} ${ind.color} shadow-lg shadow-black/50`}>
                                    <Activity className="w-4.5 h-4.5" />
                                </div>
                                <div className="text-right">
                                    <div className="text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-1">{t("dash.winrate")}</div>
                                    <div className={`text-2xl font-mono font-black tracking-tighter ${wr >= 70 ? "text-emerald-400" : wr >= 50 ? "text-amber-400" : "text-red-400"}`}>
                                        {loading ? "--" : `${wr}%`}
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-5">
                                <div className="h-1 w-full bg-zinc-900 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(255,255,255,0.1)] ${wr >= 70 ? "bg-emerald-500" : wr >= 50 ? "bg-amber-500" : "bg-red-500"}`}
                                        style={{ width: `${loading ? 0 : wr}%` }}
                                    />
                                </div>

                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-black text-white uppercase tracking-tight italic">{ind.name}</span>
                                    <div className="flex items-center gap-3">
                                        <div className="flex flex-col items-end">
                                            <span className="text-[8px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">{t("dash.signals")}</span>
                                            <div className="flex items-center gap-2">
                                                <div className="h-7 px-3 rounded-lg bg-emerald-500/5 border border-emerald-500/10 flex items-center gap-1.5">
                                                    <TrendingUp className="w-2.5 h-2.5 text-emerald-500" />
                                                    <span className="text-[10px] font-black text-emerald-400 font-mono">{stats?.buy_signals || 0}</span>
                                                </div>
                                                <div className="h-7 px-3 rounded-lg bg-red-500/5 border border-red-500/10 flex items-center gap-1.5">
                                                    <TrendingDown className="w-2.5 h-2.5 text-red-500" />
                                                    <span className="text-[10px] font-black text-red-400 font-mono">{stats?.sell_signals || 0}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Decorative Glow */}
                            <div className={`absolute -bottom-10 -right-10 w-24 h-24 blur-3xl rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-1000 ${ind.bg}`} />
                        </div>
                    );
                })}
            </div>

            <div className="flex items-center justify-between px-2 pt-2">
                <div className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em] flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-indigo-500 animate-ping" />
                    {t("dash.scanned").replace("{count}", data?.scanned_count.toString() || "0")}
                </div>
                {lastUpdated && (
                    <div className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em] flex items-center gap-2">
                        <Calendar className="w-2.5 h-2.5" />
                        {t("dash.refresh").replace("{time}", days.toString() + "d")}
                    </div>
                )}
            </div>
        </div>
    );
}
