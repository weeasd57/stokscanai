"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { Brain, AlertTriangle, Loader2, Info, X, BarChart2, LineChart, Check, TrendingUp, TrendingDown, Activity, Cpu } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useAIScanner } from "@/contexts/AIScannerContext";
import { getLocalModels, predictStock, getAdminConfig, type ScanResult, type AdminConfig } from "@/lib/api";
import { getAiScore } from "@/lib/utils";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";

export default function AIScannerPage() {
    const { t } = useLanguage();
    const { countries } = useAppState();
    const { state, setAiScanner, fetchLatestScanForModel, fetchPublicScanDates, fetchScanResultsByDate, fetchGlobalModelStats, loading } = useAIScanner();
    const { results, selected, detailData, chartType, showEma50, showEma200, showBB, showRsi, showVolume, showMacd, modelName, country } = state;

    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [localModels, setLocalModels] = useState<string[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [resultsLoading, setResultsLoading] = useState(false);
    const [activeScanMetadata, setActiveScanMetadata] = useState<any>(null);
    const [showMobileDetail, setShowMobileDetail] = useState(false);
    const [adminConfig, setAdminConfig] = useState<AdminConfig | null>(null);
    const [availableDates, setAvailableDates] = useState<string[]>([]);
    const [selectedDate, setSelectedDate] = useState<string>("");
    const [globalStats, setGlobalStats] = useState<{ winRate: number; avgPl: number; total: number }>({ winRate: 0, avgPl: 0, total: 0 });

    // Initial Load: Models list & Config
    useEffect(() => {
        let active = true;
        const loadInitial = async () => {
            setModelsLoading(true);
            try {
                const [modelsData, config] = await Promise.all([
                    getLocalModels(),
                    getAdminConfig()
                ]);

                if (!active) return;
                setAdminConfig(config);

                const enabled = config.enabledModels || [];
                const names = modelsData
                    .map((model) => (typeof model === "string" ? model : model.name))
                    .filter((name): name is string => Boolean(name))
                    // Filter based on admin config
                    .filter(name => enabled.includes(name));

                setLocalModels(names);
                if (names.length > 0 && !modelName) {
                    handleModelChange(names[0]);
                }
            } catch (err) {
                console.error("Failed to load models/config:", err);
            } finally {
                if (active) setModelsLoading(false);
            }
        };
        loadInitial();
        return () => { active = false; };
    }, []);



    async function handleModelChange(name: string) {
        setAiScanner(prev => ({ ...prev, modelName: name, selected: null, detailData: null }));
        setResultsLoading(true);
        try {
            // Fetch global stats as well
            fetchGlobalModelStats(name).then(setGlobalStats).catch(console.error);

            // First fetch available dates for this model
            const dates = await fetchPublicScanDates(name);
            setAvailableDates(dates);
            if (dates.length > 0) {
                setSelectedDate(dates[0]);
                const resultsByDate = await fetchScanResultsByDate(name, dates[0]);
                setAiScanner(prev => ({ ...prev, results: resultsByDate }));
                // We don't have separate metadata object from fetchScanResultsByDate currently, 
                // but we can simulate it or fetch it if needed. 
                // Mostly the UI uses results.
                setActiveScanMetadata({ created_at: dates[0] });
            } else {
                setAiScanner(prev => ({ ...prev, results: [] }));
                setActiveScanMetadata(null);
                setSelectedDate("");
            }
        } catch (err) {
            console.error("Error loading model results:", err);
        } finally {
            setResultsLoading(false);
        }
    }

    async function handleDateChange(date: string) {
        setSelectedDate(date);
        setResultsLoading(true);
        try {
            const resultsByDate = await fetchScanResultsByDate(modelName, date);
            setAiScanner(prev => ({ ...prev, results: resultsByDate }));
            setActiveScanMetadata({ created_at: date });
        } catch (err) {
            console.error("Error loading results for date:", err);
        } finally {
            setResultsLoading(false);
        }
    }

    const predictAbortControllerRef = useRef<AbortController | null>(null);

    async function openDetails(row: ScanResult) {
        // Cancel previous request
        if (predictAbortControllerRef.current) {
            predictAbortControllerRef.current.abort();
        }
        const controller = new AbortController();
        predictAbortControllerRef.current = controller;

        setAiScanner((prev) => ({ ...prev, selected: row, detailData: null }));
        setDetailError(null);
        setDetailLoading(true);
        try {
            const res = await predictStock({
                ticker: row.symbol,
                exchange: row.exchange ?? undefined,
                includeFundamentals: false,
                rfPreset: "fast",
                modelName: modelName
            }, controller.signal);
            setAiScanner((prev) => ({ ...prev, detailData: res }));
        } catch (err: any) {
            if (err.name === 'AbortError') return;
            setDetailError(err instanceof Error ? err.message : "Failed to load chart");
        } finally {
            if (predictAbortControllerRef.current === controller) {
                setDetailLoading(false);
            }
        }
        setShowMobileDetail(true);
    }

    useEffect(() => {
        return () => {
            if (predictAbortControllerRef.current) {
                predictAbortControllerRef.current.abort();
            }
        };
    }, []);

    useEffect(() => {
        if (!selected) {
            setShowMobileDetail(false);
        }
    }, [selected]);

    const stats = useMemo(() => {
        if (!results || results.length === 0) return { winRate: 0, avgPl: 0, total: 0 };
        const wins = results.filter(r => r.status === 'win').length;
        const avgPl = results.reduce((acc, r) => acc + (r.profit_loss_pct || 0), 0) / results.length;
        return {
            winRate: (wins / results.length) * 100,
            avgPl: avgPl,
            total: results.length
        };
    }, [results]);

    const isLoading = modelsLoading || resultsLoading;

    return (
        <div className="flex flex-col gap-8 pb-20 max-w-[1600px] mx-auto">
            <header className="flex flex-col gap-2 px-4">
                <h1 className="text-3xl font-black tracking-tighter text-white flex items-center gap-4 uppercase italic">
                    <div className="p-3 rounded-2xl bg-indigo-600 shadow-xl shadow-indigo-600/20">
                        <Brain className="h-6 w-6 text-white" />
                    </div>
                    {t("ai.title")}
                </h1>
                <p className="text-sm text-zinc-500 font-medium max-w-2xl">
                    {t("ai.subtitle")}
                </p>
            </header>

            <div className="relative rounded-[3rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-6 lg:p-10 shadow-[0_0_100px_rgba(0,0,0,0.5)] mx-4">
                {/* mesh background elements */}
                <div className="absolute inset-0 rounded-[3rem] overflow-hidden pointer-events-none">
                    <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-600/15 blur-[120px] rounded-full animate-pulse" />
                    <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-blue-600/10 blur-[100px] rounded-full animate-pulse [animation-delay:2s]" />
                </div>

                <div className="relative z-10 flex flex-col gap-10">

                    {/* Model Tabs & Results State */}

                    {/* 3. Model Tabs & Results State */}
                    <div className="space-y-8">
                        <div className="flex flex-wrap items-center gap-3 p-2 rounded-[2rem] bg-zinc-900/50 border border-white/5 backdrop-blur-3xl sticky top-4 z-[50]">
                            {modelsLoading ? (
                                <div className="flex items-center gap-3 px-6 py-3">
                                    <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
                                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Waking up Neural Cores...</span>
                                </div>
                            ) : (
                                localModels.map((name) => (
                                    <button
                                        key={name}
                                        onClick={() => handleModelChange(name)}
                                        className={`px-6 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all ${modelName === name ? "bg-white text-black shadow-2xl" : "text-zinc-500 hover:text-white hover:bg-white/5"}`}
                                    >
                                        {adminConfig?.modelAliases?.[name] || name.replace(".pkl", "").replace(/_/g, " ")}
                                    </button>
                                ))
                            )}
                        </div>

                        {/* Date Pagination Bar */}
                        {availableDates.length > 0 && (
                            <div className="flex items-center justify-between gap-4 p-4 rounded-[1.5rem] bg-zinc-950/40 border border-white/5 backdrop-blur-3xl">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                                        <Activity className="h-4 w-4 text-indigo-400" />
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest leading-none mb-1">Session Timeline</span>
                                        <span className="text-xs font-black text-white uppercase tracking-tighter">
                                            {selectedDate === availableDates[0] ? "Latest Neural Scan" : `Historical Archive: ${selectedDate}`}
                                        </span>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2 overflow-x-auto custom-scrollbar no-scrollbar py-1 px-1">
                                    {availableDates.slice(0, 7).map((date) => (
                                        <button
                                            key={date}
                                            onClick={() => handleDateChange(date)}
                                            className={`px-4 py-2 rounded-xl text-[9px] font-black uppercase tracking-widest whitespace-nowrap transition-all border ${selectedDate === date ? "bg-white text-black border-white shadow-lg" : "bg-black/40 text-zinc-500 border-white/5 hover:text-white hover:border-white/10"}`}
                                        >
                                            {new Date(date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                        </button>
                                    ))}
                                    {availableDates.length > 7 && <span className="text-zinc-700 font-bold px-2">...</span>}
                                </div>
                            </div>
                        )}

                        {/* 4. Active Model Dashboard Metrics */}
                        {modelName && (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                {[
                                    { label: "Neural Win Rate", value: `${stats.winRate.toFixed(1)}%`, globalValue: `${globalStats.winRate.toFixed(1)}%`, icon: <Check className="h-5 w-5" />, color: "text-emerald-500", bg: "bg-emerald-500/10" },
                                    { label: "Avg Potential P/L", value: `${stats.avgPl > 0 ? '+' : ''}${stats.avgPl.toFixed(2)}%`, globalValue: `${globalStats.avgPl > 0 ? '+' : ''}${globalStats.avgPl.toFixed(2)}%`, icon: <Activity className="h-5 w-5" />, color: stats.avgPl > 0 ? "text-emerald-500" : "text-red-500", bg: stats.avgPl > 0 ? "bg-emerald-500/10" : "bg-red-500/10" },
                                    { label: "Discovery hits", value: stats.total, globalValue: globalStats.total, icon: <Cpu className="h-5 w-5" />, color: "text-indigo-400", bg: "bg-indigo-500/10" },
                                ].map((s, idx) => (
                                    <div key={idx} className="relative group overflow-hidden rounded-[2.5rem] border border-white/5 bg-zinc-950/40 backdrop-blur-2xl p-8 flex flex-col gap-4 hover:border-white/10 transition-all">
                                        <div className="absolute inset-x-0 bottom-0 top-0 bg-gradient-to-tr from-white/[0.02] to-transparent pointer-events-none" />
                                        <div className="flex items-center justify-between relative z-10">
                                            <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{s.label}</span>
                                            <div className={`p-2 rounded-xl ${s.bg} ${s.color}`}>
                                                {s.icon}
                                            </div>
                                        </div>
                                        <div className="flex flex-col gap-1 relative z-10">
                                            <div className={`text-4xl font-black italic tracking-tighter ${s.color}`}>
                                                {isLoading || resultsLoading ? "---" : s.value}
                                            </div>

                                            <div className="mt-4 p-4 rounded-3xl bg-white/[0.03] border border-white/5 flex flex-col gap-1.5 relative overflow-hidden group/sub">
                                                <div className="absolute inset-x-0 bottom-0 h-0.5 bg-indigo-500/20 transform scale-x-0 group-hover/sub:scale-x-100 transition-transform duration-500 origin-left" />
                                                <div className="flex items-center justify-between">
                                                    <span className="text-[9px] font-black text-zinc-500 uppercase tracking-widest leading-none">Global Network Avg</span>
                                                    <div className="px-1.5 py-0.5 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-[7px] font-black text-indigo-400 uppercase tracking-tighter">ALL-TIME</div>
                                                </div>
                                                <div className="text-xl font-black text-white italic tracking-tighter">
                                                    {isLoading || resultsLoading ? "--" : s.globalValue}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>


                    {(results.length > 0 || resultsLoading) && (
                        <div className="flex flex-col gap-8 animate-in fade-in duration-1000">

                            {/* Main Results Table (Now always full width below active chart) */}
                            <div className="rounded-[2.5rem] border border-white/10 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl">
                                <div className="px-6 py-5 border-b border-white/5 bg-zinc-950/80 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Activity className="w-5 h-5 text-emerald-500" />
                                        <div className="flex flex-col">
                                            <h3 className="text-sm font-black text-white uppercase tracking-widest leading-none mb-1">Verified Opportunities ({results.length})</h3>
                                            {activeScanMetadata && (
                                                <span className="text-[8px] text-zinc-600 font-bold uppercase tracking-tighter">
                                                    Neural Scan Executed: {new Date(activeScanMetadata.created_at).toLocaleString()}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                                        <span className="text-[8px] font-black text-indigo-400 uppercase tracking-widest">LIVE SIGNAL FEED</span>
                                    </div>
                                </div>
                                <div className="p-0 overflow-x-auto custom-scrollbar">
                                    {resultsLoading ? (
                                        <div className="p-20 flex flex-col items-center justify-center gap-4">
                                            <Loader2 className="h-10 w-10 animate-spin text-indigo-600" />
                                            <p className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500 animate-pulse">Reconstructing Signal...</p>
                                        </div>
                                    ) : results.length === 0 ? (
                                        <div className="p-20 flex flex-col items-center justify-center gap-4">
                                            <AlertTriangle className="h-10 w-10 text-zinc-800" />
                                            <p className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-600">No scan results found for this model</p>
                                        </div>
                                    ) : (
                                        <table className="w-full text-left text-xs whitespace-nowrap">
                                            <thead className="bg-zinc-950/80 text-[9px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                                                <tr>
                                                    <th className="px-6 py-4">Symbol</th>
                                                    <th className="px-6 py-4">Signal</th>
                                                    <th className="px-6 py-4 text-center">AI Score</th>
                                                    <th className="px-6 py-4 text-center">Council</th>
                                                    <th className="px-6 py-4 text-center text-indigo-400">Consensus</th>
                                                    <th className="px-6 py-4 text-right text-emerald-500">Target</th>
                                                    <th className="px-6 py-4 text-right text-red-500">Stop Loss</th>
                                                    <th className="px-6 py-4 text-center">Status</th>
                                                    <th className="px-6 py-4 text-right">P/L</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5">
                                                {results.map((r) => (
                                                    <tr
                                                        key={r.symbol}
                                                        onClick={() => openDetails(r)}
                                                        className={`cursor-pointer group transition-all ${selected?.symbol === r.symbol ? "bg-indigo-600/10" : "hover:bg-white/[0.02]"}`}
                                                    >
                                                        <td className="px-6 py-4">
                                                            <div className="flex items-center gap-3">
                                                                <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center overflow-hidden shadow-inner group-hover:border-indigo-500/50 transition-all">
                                                                    <StockLogo symbol={r.symbol} className="w-6 h-6 object-contain" />
                                                                </div>
                                                                <div className="flex flex-col">
                                                                    <span className="text-sm font-black text-white">{r.symbol}</span>
                                                                    <span className="text-[10px] text-zinc-500 font-bold uppercase truncate max-w-[120px]">{r.name || "Unknown Asset"}</span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4">
                                                            <span className={`text-[11px] font-black uppercase tracking-widest ${r.signal?.toLowerCase() === 'buy' || r.signal === 'UP' ? 'text-emerald-500' : 'text-red-500'}`}>
                                                                Signal_{r.signal}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <div className="flex items-center justify-center gap-2">
                                                                {(() => {
                                                                    const { score, label, color, bg } = getAiScore(Number(r.precision));
                                                                    return (
                                                                        <div className="flex flex-col items-center gap-1">
                                                                            <div className={`px-2 py-0.5 rounded-md ${bg} ${color} text-[10px] font-black italic`}>
                                                                                {score}/10
                                                                            </div>
                                                                            <span className="text-[7px] font-black uppercase tracking-tighter text-zinc-600">{label}</span>
                                                                        </div>
                                                                    );
                                                                })()}
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="font-black text-white italic">{r.council_score ? `${r.council_score.toFixed(1)}%` : "N/A"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">{r.consensus_ratio || "N/A"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono font-bold text-emerald-400">{r.target_price ? r.target_price.toFixed(2) : "-"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono font-bold text-red-400">{r.stop_loss ? r.stop_loss.toFixed(2) : "-"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            {r.status === 'win' ? (
                                                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                                                                    <TrendingUp className="h-3 w-3" />
                                                                    <span className="text-[9px] font-black uppercase tracking-widest">SUCCESS</span>
                                                                </div>
                                                            ) : r.status === 'loss' ? (
                                                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-red-500/10 text-red-500 border border-red-500/20">
                                                                    <TrendingDown className="h-3 w-3" />
                                                                    <span className="text-[9px] font-black uppercase tracking-widest">DIVERGED</span>
                                                                </div>
                                                            ) : (
                                                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-zinc-500/10 text-zinc-500 border border-white/5">
                                                                    <Activity className="h-3 w-3" />
                                                                    <span className="text-[9px] font-black uppercase tracking-widest">PENDING</span>
                                                                </div>
                                                            )}
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className={`text-sm font-black italic tracking-tight ${Number(r.profit_loss_pct ?? 0) > 0 ? 'text-emerald-500' : Number(r.profit_loss_pct ?? 0) < 0 ? 'text-red-500' : 'text-zinc-500'}`}>
                                                                {Number(r.profit_loss_pct ?? 0) > 0 ? '+' : ''}{Number(r.profit_loss_pct ?? 0).toFixed(2)}%
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    )}
                                </div>

                                <div className="px-6 py-4 border-t border-white/5 bg-zinc-950/40 flex items-center justify-between text-[10px] font-black uppercase tracking-widest text-zinc-500">
                                    <div className="flex items-center gap-6">
                                        <div className="flex items-center gap-2">
                                            <span className="text-zinc-700">Displaying:</span>
                                            <span className="text-indigo-400">{results.length} Interceptions</span>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-zinc-700">Active Engine:</span>
                                        <span className="text-white">{modelName ? (adminConfig?.modelAliases?.[modelName] || modelName.replace(".pkl", "")) : "none"}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>


            {/* Fullscreen Chart Dialog */}
            {showMobileDetail && selected && (
                <div className="fixed inset-0 z-[9999] flex flex-col bg-black/90 backdrop-blur-md">
                    {/* Header */}
                    <div className="flex-none px-6 py-4 border-b border-white/10 bg-zinc-950/90 flex items-center justify-between">
                        <div>
                            <div className="flex items-center gap-4">
                                <div className="text-xl font-black text-white font-mono">{selected.symbol}</div>
                                {selected.council_score && (
                                    <div className="px-3 py-1 rounded-full bg-indigo-600/20 border border-indigo-500/40 flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                                        <span className="text-[10px] font-black text-white uppercase tracking-widest italic">Council: {selected.council_score.toFixed(1)}%</span>
                                    </div>
                                )}
                            </div>
                            <div className="text-[11px] font-black text-zinc-500 uppercase tracking-widest truncate">{selected.name}</div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="hidden md:flex items-center gap-1.5 px-4 py-2 rounded-xl bg-zinc-900/60 border border-white/5">
                                {[
                                    { id: "showEma50", label: "EMA50", color: "bg-orange-500" },
                                    { id: "showEma200", label: "EMA200", color: "bg-cyan-500" },
                                    { id: "showBB", label: "BB", color: "bg-purple-500" },
                                    { id: "showRsi", label: "RSI", color: "bg-pink-500" },
                                    { id: "showMacd", label: "MACD", color: "bg-indigo-500" },
                                    { id: "showVolume", label: "VOL", color: "bg-blue-500" },
                                ].map((ind) => (
                                    <button
                                        key={ind.id}
                                        onClick={() => setAiScanner(p => ({ ...p, [ind.id]: !((p as any)[ind.id]) }))}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[9px] font-black uppercase transition-all ${state[ind.id as keyof typeof state] ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"}`}
                                    >
                                        <div className={`w-1.5 h-1.5 rounded-full ${state[ind.id as keyof typeof state] ? ind.color : "bg-zinc-800"}`} />
                                        {ind.label}
                                    </button>
                                ))}
                            </div>
                            <div className="flex gap-1 rounded-lg bg-zinc-900/60 border border-white/5 p-1">
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, chartType: "candle" }))}
                                    className={`p-2 rounded-md transition-all ${chartType === "candle" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-white"}`}
                                >
                                    <BarChart2 className="h-4 w-4" />
                                </button>
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, chartType: "area" }))}
                                    className={`p-2 rounded-md transition-all ${chartType === "area" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-white"}`}
                                >
                                    <LineChart className="h-4 w-4" />
                                </button>
                            </div>
                            <button
                                onClick={() => {
                                    setShowMobileDetail(false);
                                    setAiScanner((prev) => ({ ...prev, selected: null, detailData: null }));
                                }}
                                className="p-2 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
                            >
                                <X className="h-6 w-6" />
                            </button>
                        </div>
                    </div>

                    {/* Content - Full height chart */}
                    <div className="flex-1 p-4 overflow-y-auto custom-scrollbar bg-zinc-950/80">
                        {detailError && (
                            <div className="p-3 rounded-2xl bg-red-500/5 border border-red-500/20 text-[11px] text-red-300 mb-4">{detailError}</div>
                        )}
                        {detailLoading && (
                            <div className="flex flex-col items-center justify-center h-full gap-4">
                                <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                <p className="text-sm font-black text-zinc-500 uppercase tracking-widest">Loading chart data...</p>
                            </div>
                        )}
                        {detailData && (
                            <div className="flex flex-col gap-10 h-full">
                                <div className="rounded-[2rem] border border-white/5 bg-zinc-900/40 overflow-hidden shadow-2xl" style={{ height: 600 }}>
                                    <CandleChart
                                        rows={detailData.testPredictions}
                                        showEma50={showEma50}
                                        showEma200={showEma200}
                                        showBB={showBB}
                                        showRsi={showRsi}
                                        showMacd={showMacd}
                                        showVolume={showVolume}
                                        chartType={chartType}
                                        height={600}
                                    />
                                </div>

                                {/* Neural Logic Drivers - ALWAYS VISIBLE if data exists */}
                                <div className="rounded-[2.5rem] border border-white/5 bg-white/[0.02] p-10 space-y-8">
                                    <div className="flex items-center gap-4">
                                        <div className="p-3 rounded-2xl bg-indigo-500/20">
                                            <Brain className="h-5 w-5 text-indigo-400" />
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-xs font-black text-indigo-400 uppercase tracking-[0.2em]">Neural Logic Drivers</span>
                                            <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5">Decision Matrix Analysis Locked</span>
                                        </div>
                                    </div>

                                    {detailData.topReasons && detailData.topReasons.length > 0 ? (
                                        <div className="flex flex-wrap gap-4">
                                            {detailData.topReasons.map((reason, idx) => (
                                                <div key={idx} className="px-6 py-3 rounded-2xl bg-black/60 border border-white/5 flex items-center gap-3 group transition-all hover:border-indigo-500/30">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
                                                    <span className="text-xs font-bold text-zinc-300 uppercase tracking-widest leading-none">
                                                        {reason}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="p-12 border border-dashed border-white/5 rounded-[2rem] flex items-center justify-center text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em]">
                                            Neural Logic Datashards Unavailable
                                        </div>
                                    )}
                                </div>

                                <div className="rounded-[2.5rem] border border-white/5 bg-white/[0.02] p-10 shadow-xl overflow-hidden">
                                    <div className="flex items-center gap-4 mb-8">
                                        <div className="p-3 rounded-2xl bg-emerald-500/20">
                                            <Activity className="h-5 w-5 text-emerald-500" />
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-xs font-black text-emerald-500 uppercase tracking-[0.2em]">Historical Performance Slice</span>
                                            <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5">Spectral Backtest Results</span>
                                        </div>
                                    </div>
                                    <div className="overflow-x-auto custom-scrollbar">
                                        <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
