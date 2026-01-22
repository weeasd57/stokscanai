"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { Brain, AlertTriangle, Loader2, Info, X, BarChart2, LineChart, Check, TrendingUp, TrendingDown, Activity, Bookmark, BookmarkCheck, Cpu } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useAIScanner } from "@/contexts/AIScannerContext";
import { getLocalModels, predictStock, getAdminConfig, type ScanResult, type AdminConfig } from "@/lib/api";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";

export default function AIScannerPage() {
    const { t } = useLanguage();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();
    const { countries } = useAppState();
    const { state, setAiScanner, fetchLatestScanForModel, loading } = useAIScanner();
    const { results, selected, detailData, chartType, showEma50, showEma200, showBB, showRsi, showVolume, modelName, country } = state;

    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [localModels, setLocalModels] = useState<string[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [resultsLoading, setResultsLoading] = useState(false);
    const [activeScanMetadata, setActiveScanMetadata] = useState<any>(null);
    const [showMobileDetail, setShowMobileDetail] = useState(false);
    const [adminConfig, setAdminConfig] = useState<AdminConfig | null>(null);

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
            const data = await fetchLatestScanForModel(name);
            if (data) {
                setAiScanner(prev => ({ ...prev, results: data.results }));
                setActiveScanMetadata(data.history);
            } else {
                setAiScanner(prev => ({ ...prev, results: [] }));
                setActiveScanMetadata(null);
            }
        } catch (err) {
            console.error("Error loading model results:", err);
        } finally {
            setResultsLoading(false);
        }
    }

    async function openDetails(row: ScanResult) {
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
            });
            setAiScanner((prev) => ({ ...prev, detailData: res }));
        } catch (err) {
            setDetailError(err instanceof Error ? err.message : "Failed to load chart");
        } finally {
            setDetailLoading(false);
        }
        setShowMobileDetail(true);
    }

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
                                        {name.replace(".pkl", "").replace(/_/g, " ")}
                                    </button>
                                ))
                            )}
                        </div>

                        {/* 4. Active Model Dashboard Metrics */}
                        {modelName && (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                {[
                                    { label: "Neural Win Rate", value: `${stats.winRate.toFixed(1)}%`, icon: <Check className="h-5 w-5" />, color: "text-emerald-500", bg: "bg-emerald-500/10" },
                                    { label: "Avg Potential P/L", value: `${stats.avgPl > 0 ? '+' : ''}${stats.avgPl.toFixed(2)}%`, icon: <Activity className="h-5 w-5" />, color: stats.avgPl > 0 ? "text-emerald-500" : "text-red-500", bg: stats.avgPl > 0 ? "bg-emerald-500/10" : "bg-red-500/10" },
                                    { label: "Discovery hits", value: stats.total, icon: <Cpu className="h-5 w-5" />, color: "text-indigo-400", bg: "bg-indigo-500/10" },
                                ].map((s, idx) => (
                                    <div key={idx} className="relative group overflow-hidden rounded-[2.5rem] border border-white/5 bg-zinc-950/40 backdrop-blur-2xl p-8 flex flex-col gap-4 hover:border-white/10 transition-all">
                                        <div className="absolute inset-x-0 bottom-0 top-0 bg-gradient-to-tr from-white/[0.02] to-transparent pointer-events-none" />
                                        <div className="flex items-center justify-between relative z-10">
                                            <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{s.label}</span>
                                            <div className={`p-2 rounded-xl ${s.bg} ${s.color}`}>
                                                {s.icon}
                                            </div>
                                        </div>
                                        <div className={`text-4xl font-black italic tracking-tighter relative z-10 ${s.color}`}>
                                            {isLoading || resultsLoading ? "---" : s.value}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>


                    {/* 5. Results Table Grid */}
                    {(results.length > 0 || resultsLoading) && (
                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-in fade-in duration-1000">
                            <div className={`rounded-3xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl ${selected && detailData ? "lg:col-span-8" : "lg:col-span-12"}`}>
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
                                                    <th className="px-6 py-4 text-center">Confidence</th>
                                                    <th className="px-6 py-4 text-center">Status</th>
                                                    <th className="px-6 py-4 text-right">P/L</th>
                                                    <th className="px-6 py-4 text-right">Action</th>
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
                                                            <span className={`text-[11px] font-black uppercase tracking-widest ${r.signal === 'buy' || r.signal === 'UP' ? 'text-emerald-500' : 'text-red-500'}`}>
                                                                Signal_{r.signal}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <div className="flex items-center justify-center gap-3">
                                                                <div className="w-12 h-1 rounded-full bg-zinc-900 overflow-hidden">
                                                                    <div
                                                                        className={`h-full ${Number(r.confidence) > 0.7 ? 'bg-emerald-500' : 'bg-indigo-500'}`}
                                                                        style={{ width: `${(Number(r.confidence) * 100)}%` }}
                                                                    />
                                                                </div>
                                                                <span className="text-[11px] font-mono font-black text-indigo-400">{(Number(r.confidence) * 100).toFixed(1)}%</span>
                                                            </div>
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
                                                        <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                                            <button
                                                                onClick={() => isSaved(r.symbol) ? removeSymbolBySymbol(r.symbol) : saveSymbol({ symbol: r.symbol, name: r.name || "", source: "ai_scanner", metadata: {} })}
                                                                className={`p-2 rounded-lg transition-all ${isSaved(r.symbol) ? "text-indigo-400 bg-indigo-500/10" : "text-zinc-600 hover:text-white"}`}
                                                            >
                                                                {isSaved(r.symbol) ? <BookmarkCheck className="h-4 w-4" /> : <Bookmark className="h-4 w-4" />}
                                                            </button>
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
                                        <span className="text-white">{modelName?.replace(".pkl", "")}</span>
                                    </div>
                                </div>
                            </div>

                            {/* Stock Details Sidebar */}
                            {selected && detailData && (
                                <div className="lg:col-span-4 rounded-3xl border border-white/5 bg-zinc-950/40 backdrop-blur-3xl overflow-hidden shadow-2xl animate-in slide-in-from-right-10 duration-500 flex flex-col h-fit lg:sticky lg:top-24">
                                    <div className="p-6 border-b border-white/5 flex items-center justify-between bg-zinc-900/40">
                                        <div className="flex items-center gap-4">
                                            <div className="p-3 rounded-2xl bg-indigo-600 shadow-xl shadow-indigo-600/20">
                                                <Brain className="h-5 w-5 text-white" />
                                            </div>
                                            <div className="flex flex-col">
                                                <h3 className="text-xl font-black text-white italic tracking-tighter uppercase">{selected.symbol}</h3>
                                                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Neural Analysis Feed</span>
                                            </div>
                                        </div>
                                        <button onClick={() => setAiScanner(prev => ({ ...prev, selected: null, detailData: null }))} className="p-2 rounded-xl hover:bg-white/5 transition-all text-zinc-500 hover:text-white">
                                            <X className="h-5 w-5" />
                                        </button>
                                    </div>

                                    <div className="p-6 space-y-8">
                                        {/* Chart Section */}
                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <span className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">Market Visualization</span>
                                                <div className="flex gap-2">
                                                    <button onClick={() => setAiScanner(p => ({ ...p, chartType: 'candle' }))} className={`p-1.5 rounded-lg border transition-all ${chartType === 'candle' ? 'bg-indigo-600 border-indigo-500' : 'bg-white/5 border-white/5'}`}>
                                                        <BarChart2 className="h-3 w-3" />
                                                    </button>
                                                    <button onClick={() => setAiScanner(p => ({ ...p, chartType: 'area' }))} className={`p-1.5 rounded-lg border transition-all ${chartType === 'area' ? 'bg-indigo-600 border-indigo-500' : 'bg-white/5 border-white/5'}`}>
                                                        <LineChart className="h-3 w-3" />
                                                    </button>
                                                </div>
                                            </div>

                                            <div className="h-[250px] rounded-2xl border border-white/5 bg-black/40 overflow-hidden relative shadow-inner">
                                                {detailLoading ? (
                                                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                                                        <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
                                                        <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Rendering Chart...</span>
                                                    </div>
                                                ) : detailData?.testPredictions ? (
                                                    <CandleChart
                                                        rows={detailData.testPredictions}
                                                        chartType={chartType}
                                                        showEma50={showEma50}
                                                        showEma200={showEma200}
                                                        showBB={showBB}
                                                        showRsi={showRsi}
                                                        showVolume={showVolume}
                                                    />
                                                ) : (
                                                    <div className="absolute inset-0 flex items-center justify-center text-zinc-500 text-[10px] font-black uppercase">Data Stream Interrupted</div>
                                                )}
                                            </div>
                                        </div>

                                        {/* Indicators Control */}
                                        <div className="space-y-4">
                                            <span className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">Indicator Overlays</span>
                                            <div className="grid grid-cols-2 gap-3">
                                                {[
                                                    { key: 'showEma50', label: 'EMA 50', icon: 'E50' },
                                                    { key: 'showEma200', label: 'EMA 200', icon: 'E20' },
                                                    { key: 'showBB', label: 'Bollinger', icon: 'BB' },
                                                    { key: 'showVolume', label: 'Volume', icon: 'VOL' },
                                                    { key: 'showRsi', label: 'RSI', icon: 'RSI' },
                                                ].map((idx) => (
                                                    <button
                                                        key={idx.key}
                                                        onClick={() => setAiScanner(p => ({ ...p, [idx.key]: !((p as any)[idx.key]) }))}
                                                        className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${((state as any)[idx.key]) ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400' : 'bg-white/5 border-white/5 text-zinc-600 hover:border-white/10'}`}
                                                    >
                                                        <div className="text-[10px] font-black w-8">{idx.icon}</div>
                                                        <span className="text-[10px] font-black uppercase tracking-widest">{idx.label}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="pt-4 border-t border-white/5">
                                            <div className="p-5 rounded-2xl bg-indigo-600 shadow-2xl shadow-indigo-600/40 flex flex-col items-center gap-3 relative overflow-hidden group">
                                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-all duration-1000" />
                                                <div className="flex items-center gap-3">
                                                    <Brain className="h-5 w-5 text-white" />
                                                    <span className="text-[10px] font-black text-indigo-100 uppercase tracking-[0.2em]">Neural Signal Lock</span>
                                                </div>
                                                <div className="text-3xl font-black italic tracking-tighter text-white">
                                                    {selected.signal === 'UP' || selected.signal === 'buy' ? 'LONG_TARGET' : 'SHORT_ENTRY'}
                                                </div>
                                                <div className="px-4 py-1.5 rounded-full bg-black/20 text-white text-[10px] font-bold">
                                                    Confidence: {(Number(selected.confidence) * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                        </div>

                                        {/* Historical Bench Table */}
                                        <div className="mt-6 pt-6 border-t border-white/5">
                                            <div className="text-[9px] font-black text-indigo-400 uppercase tracking-widest mb-3">Historical Accuracy Bench</div>
                                            <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                        </div>
                                    </div>
                                </div>
                            )}
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
                            <div className="text-xl font-black text-white font-mono">{selected.symbol}</div>
                            <div className="text-[11px] font-black text-zinc-500 uppercase tracking-widest truncate">{selected.name}</div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="flex gap-1 rounded-lg bg-zinc-900/60 border border-white/5 p-1">
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, chartType: "candle" }))}
                                    className={`px-3 py-1.5 rounded-md text-xs font-bold uppercase ${chartType === "candle" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-white"}`}
                                >Candle</button>
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, chartType: "area" }))}
                                    className={`px-3 py-1.5 rounded-md text-xs font-bold uppercase ${chartType === "area" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-white"}`}
                                >Area</button>
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
                            <div className="space-y-4 h-full">
                                <div className="rounded-2xl border border-white/5 bg-zinc-900/30 overflow-hidden">
                                    <CandleChart
                                        rows={detailData.testPredictions}
                                        showEma50={showEma50}
                                        showEma200={showEma200}
                                        showBB={showBB}
                                        showRsi={showRsi}
                                        showVolume={showVolume}
                                        chartType={chartType}
                                    />
                                </div>
                                <div className="p-4 rounded-2xl bg-indigo-600/5 border border-indigo-500/10">
                                    <div className="text-[9px] font-black text-indigo-400 uppercase tracking-widest mb-3">Historical Accuracy Bench</div>
                                    <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

