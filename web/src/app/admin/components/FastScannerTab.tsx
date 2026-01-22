"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { Brain, AlertTriangle, Loader2, Globe, Bookmark, BookmarkCheck, Info, X, BarChart2, LineChart, Cpu, Check, Activity, ChevronDown } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useAIScanner } from "@/contexts/AIScannerContext";
import { getLocalModels, predictStock, getAdminConfig, updateAdminConfig, type ScanResult, type AdminConfig } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";

export default function FastScannerTab() {
    const { t } = useLanguage();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();
    const { countries } = useAppState();
    const { state, setAiScanner, runAiScan, stopAiScan, loading, error, clearAiScannerView, restoreLastAiScan, saveCurrentScan } = useAIScanner();

    const { country, results, progress, hasScanned, showPrecisionInfo, selected, detailData, rfPreset, rfParamsJson, chartType, showEma50, showEma200, showBB, showRsi, showVolume, scanHistory, modelName } = state;
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [showAdvancedRf, setShowAdvancedRf] = useState(false);
    const [showCountryDialog, setShowCountryDialog] = useState(false);
    const [viewMode, setViewMode] = useState<"scan" | "history">("scan");
    const [history, setHistory] = useState<any[]>([]);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [selectedHistoryScan, setSelectedHistoryScan] = useState<any | null>(null);
    const [selectedHistoryResults, setSelectedHistoryResults] = useState<any[]>([]);
    const [historyResultsLoading, setHistoryResultsLoading] = useState(false);
    const [refreshingPerformance, setRefreshingPerformance] = useState(false);
    const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const [localModels, setLocalModels] = useState<string[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [modelsError, setModelsError] = useState<string | null>(null);
    const [showMobileDetail, setShowMobileDetail] = useState(false);
    const [nowTick, setNowTick] = useState(() => Date.now());
    const [adminConfig, setAdminConfig] = useState<AdminConfig | null>(null);

    useEffect(() => {
        getAdminConfig().then(setAdminConfig).catch(console.error);
    }, []);

    const toggleModelVisibility = async (mName: string) => {
        if (!adminConfig) return;
        const currentEnabled = adminConfig.enabledModels || [];
        const nextEnabled = currentEnabled.includes(mName)
            ? currentEnabled.filter(m => m !== mName)
            : [...currentEnabled, mName];

        try {
            const updated = await updateAdminConfig({ enabledModels: nextEnabled });
            setAdminConfig(updated);
        } catch (err) {
            console.error("Failed to update model visibility:", err);
        }
    };

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsModelDropdownOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    useEffect(() => {
        if (!loading) return;
        const id = window.setInterval(() => setNowTick(Date.now()), 1000);
        return () => window.clearInterval(id);
    }, [loading]);

    useEffect(() => {
        let active = true;
        const loadModels = async () => {
            setModelsLoading(true);
            setModelsError(null);
            try {
                const data = await getLocalModels();
                if (!active) return;
                const names = data
                    .map((model) => (typeof model === "string" ? model : model.name))
                    .filter((name): name is string => Boolean(name));
                setLocalModels(names);
                if (!modelName && names.length > 0) {
                    setAiScanner((prev) => ({ ...prev, modelName: names[0] }));
                }
            } catch (err) {
                if (!active) return;
                setModelsError(err instanceof Error ? err.message : "Failed to load local models");
            } finally {
                if (active) setModelsLoading(false);
            }
        };
        loadModels();
        return () => {
            active = false;
        };
    }, [modelName, setAiScanner]);

    const quickRf = useMemo(() => {
        const defaults = {
            n_estimators: 80,
            max_depth: 10,
            min_samples_leaf: 3,
            min_samples_split: 10,
            max_features: "sqrt" as "sqrt" | "log2" | "auto",
        };

        try {
            const v = JSON.parse(rfParamsJson || "{}");
            if (!v || typeof v !== "object" || Array.isArray(v)) return defaults;
            return {
                n_estimators: Number.isFinite(Number((v as any).n_estimators)) ? Number((v as any).n_estimators) : defaults.n_estimators,
                max_depth: (v as any).max_depth === null || (v as any).max_depth === "" ? null : (Number.isFinite(Number((v as any).max_depth)) ? Number((v as any).max_depth) : defaults.max_depth),
                min_samples_leaf: Number.isFinite(Number((v as any).min_samples_leaf)) ? Number((v as any).min_samples_leaf) : defaults.min_samples_leaf,
                min_samples_split: Number.isFinite(Number((v as any).min_samples_split)) ? Number((v as any).min_samples_split) : defaults.min_samples_split,
                max_features: (typeof (v as any).max_features === "string" ? (v as any).max_features : defaults.max_features) as any,
            };
        } catch {
            return defaults;
        }
    }, [rfParamsJson]);

    const parsedRfParams = useMemo(() => {
        try {
            const v = JSON.parse(rfParamsJson || "{}");
            if (!v || typeof v !== "object" || Array.isArray(v)) return null;
            return v as Record<string, unknown>;
        } catch {
            return null;
        }
    }, [rfParamsJson]);

    const effectiveRfParams = useMemo(() => {
        if (showAdvancedRf) return parsedRfParams;
        return {
            n_estimators: Math.max(1, Math.min(2000, Math.round(quickRf.n_estimators))),
            max_depth: quickRf.max_depth === null ? null : Math.max(1, Math.min(256, Math.round(quickRf.max_depth))),
            min_samples_leaf: Math.max(1, Math.min(10000, Math.round(quickRf.min_samples_leaf))),
            min_samples_split: Math.max(2, Math.min(10000, Math.round(quickRf.min_samples_split))),
            max_features: quickRf.max_features === "auto" ? "sqrt" : quickRf.max_features,
            n_jobs: -1,
        } as Record<string, unknown>;
    }, [parsedRfParams, quickRf, showAdvancedRf]);

    async function openDetails(row: ScanResult) {
        setAiScanner((prev) => ({ ...prev, selected: row, detailData: null }));
        setDetailError(null);
        setDetailLoading(true);
        try {
            const res = await predictStock({
                ticker: row.symbol,
                exchange: row.exchange ?? undefined,
                includeFundamentals: false,
                rfPreset: rfPreset,
                rfParams: effectiveRfParams ?? undefined,
            });
            setAiScanner((prev) => ({ ...prev, detailData: res }));
        } catch (err) {
            setDetailError(err instanceof Error ? err.message : "Failed to load chart");
        } finally {
            setDetailLoading(false);
        }
        setShowMobileDetail(true);
    }

    const { fetchScanHistory, fetchScanResults, refreshScanPerformance } = useAIScanner();

    async function loadHistory() {
        setHistoryLoading(true);
        const data = await fetchScanHistory();
        setHistory(data);
        setHistoryLoading(false);
    }

    async function viewScanDetails(scan: any) {
        setSelectedHistoryScan(scan);
        setHistoryResultsLoading(true);
        const results = await fetchScanResults(scan.id);
        setSelectedHistoryResults(results);
        setHistoryResultsLoading(false);
    }

    async function handleRefreshPerformance() {
        if (!selectedHistoryScan) return;
        setRefreshingPerformance(true);
        try {
            await refreshScanPerformance(selectedHistoryScan.id);
            // Re-fetch results
            const results = await fetchScanResults(selectedHistoryScan.id);
            setSelectedHistoryResults(results);
        } catch (err) {
            console.error(err);
        } finally {
            setRefreshingPerformance(false);
        }
    }

    useEffect(() => {
        if (viewMode === "history") {
            loadHistory();
        }
    }, [viewMode]);

    async function runScan() {
        await runAiScan({ rfParams: effectiveRfParams ?? null, minPrecision: 0.6, shouldSave: false });
    }

    return (
        <div className="flex flex-col gap-8 p-6 lg:p-10 max-w-[1600px] mx-auto">
            <header className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div className="flex flex-col gap-2">
                    <h1 className="text-2xl font-black tracking-tighter text-white flex items-center gap-4 uppercase italic">
                        <div className="p-2.5 rounded-xl bg-indigo-600 shadow-lg shadow-indigo-600/20">
                            <Brain className="h-5 w-5 text-white" />
                        </div>
                        {t("ai.title") || "Fast AI Scanner"}
                    </h1>
                    <p className="text-sm text-zinc-500 font-medium max-w-2xl">
                        {viewMode === "scan" ? "Run high-probability predictive analysis across entire markets." : "Analyze the performance of your previous AI predictions."}
                    </p>
                </div>
                <div className="flex items-center gap-4 bg-zinc-950/50 p-1.5 rounded-2xl border border-white/5">
                    <button
                        onClick={() => setViewMode("scan")}
                        className={`px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === "scan" ? "bg-white text-black shadow-xl" : "text-zinc-500 hover:text-white"}`}
                    >
                        RUN SCAN
                    </button>
                    <button
                        onClick={() => setViewMode("history")}
                        className={`px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === "history" ? "bg-white text-black shadow-xl" : "text-zinc-500 hover:text-white"}`}
                    >
                        HISTORY
                    </button>
                </div>
            </header>

            {viewMode === "scan" ? (
                <>
                    <div className="relative rounded-[2.5rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-8 shadow-2xl overflow-hidden">
                        <div className="relative z-10 space-y-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Section 1: Market Configuration */}
                                <div className="group rounded-[2rem] border border-white/5 bg-white/[0.02] p-6 space-y-6 flex flex-col justify-between hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                            <div className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Market Config</div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        <button
                                            onClick={() => setShowCountryDialog(true)}
                                            disabled={loading}
                                            className="h-14 flex items-center justify-between gap-4 rounded-xl border border-white/5 bg-zinc-950/50 px-5 text-xs font-bold text-zinc-200 hover:bg-zinc-900 transition-all group/btn shadow-inner"
                                        >
                                            <div className="flex items-center gap-3">
                                                <Globe className="h-4 w-4 text-zinc-400 group-hover/btn:text-blue-500 transition-colors" />
                                                <div className="flex flex-col items-start">
                                                    <span className="text-[8px] text-zinc-500 font-black uppercase tracking-widest leading-none mb-1">Region</span>
                                                    <span className="uppercase tracking-[0.1em] text-white">{country}</span>
                                                </div>
                                            </div>
                                            <ChevronDown className="h-3 w-3 text-zinc-600" />
                                        </button>

                                        <div className={`h-14 px-5 rounded-xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-4 transition-all shadow-inner ${state.scanAllMarket !== false ? "opacity-30 grayscale cursor-not-allowed" : "hover:border-white/20"}`}>
                                            <div className="flex flex-col">
                                                <span className="text-[8px] font-black uppercase tracking-widest text-zinc-500 leading-none mb-1">Depth Limit</span>
                                                <span className="text-[10px] font-bold text-zinc-300">SYMBOLS</span>
                                            </div>
                                            <input
                                                type="number"
                                                min={1}
                                                max={400}
                                                value={state.limit}
                                                onChange={(e) => setAiScanner((prev) => ({ ...prev, limit: Math.min(400, Math.max(1, Number(e.target.value) || 1)) }))}
                                                disabled={loading || state.scanAllMarket !== false}
                                                className="w-14 h-8 rounded-lg bg-zinc-900 border border-white/10 px-2 text-xs font-mono text-indigo-400 outline-none text-center"
                                            />
                                        </div>
                                    </div>


                                    <label className="h-14 px-5 rounded-xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-6 cursor-pointer group/toggle hover:bg-zinc-900 transition-all shadow-inner border-dashed">
                                        <div className="flex items-center gap-3">
                                            <Activity className="h-4 w-4 text-indigo-400" />
                                            <div className="flex flex-col">
                                                <span className="text-[10px] font-black uppercase tracking-widest text-zinc-300">Scan Full Market</span>
                                            </div>
                                        </div>
                                        <div className="relative">
                                            <input
                                                type="checkbox"
                                                checked={state.scanAllMarket !== false}
                                                onChange={(e) => setAiScanner((prev) => ({ ...prev, scanAllMarket: e.target.checked }))}
                                                disabled={loading}
                                                className="sr-only peer"
                                            />
                                            <div className="w-10 h-5 bg-zinc-800 rounded-full peer peer-checked:bg-indigo-600 transition-all border border-white/5" />
                                            <div className="absolute left-1 top-1 w-3 h-3 bg-zinc-200 rounded-full peer-checked:translate-x-5 transition-all shadow-lg"></div>
                                        </div>
                                    </label>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="flex flex-col gap-1.5">
                                            <span className="text-[8px] font-black uppercase tracking-widest text-zinc-500 ml-1">Start Date</span>
                                            <input
                                                type="date"
                                                value={state.startDate}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, startDate: e.target.value }))}
                                                disabled={loading}
                                                className="h-10 rounded-xl bg-zinc-950/50 border border-white/5 px-3 text-[10px] font-mono text-zinc-300 outline-none focus:border-indigo-500 transition-all"
                                            />
                                        </div>
                                        <div className="flex flex-col gap-1.5">
                                            <span className="text-[8px] font-black uppercase tracking-widest text-zinc-500 ml-1">End Date</span>
                                            <input
                                                type="date"
                                                value={state.endDate}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, endDate: e.target.value }))}
                                                disabled={loading}
                                                className="h-10 rounded-xl bg-zinc-950/50 border border-white/5 px-3 text-[10px] font-mono text-zinc-300 outline-none focus:border-indigo-500 transition-all"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Section 2: AI Configuration */}
                                <div className="group rounded-[2rem] border border-white/5 bg-white/[0.02] p-6 space-y-6 flex flex-col justify-between hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                                            <div className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">AI Intelligence</div>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Selection Neural Core</span>
                                            <div className="relative flex-1 w-full" ref={dropdownRef}>
                                                <button
                                                    type="button"
                                                    onClick={() => !loading && !modelsLoading && setIsModelDropdownOpen(!isModelDropdownOpen)}
                                                    className={`w-full h-14 rounded-xl bg-zinc-950/50 border px-5 text-[10px] font-black uppercase tracking-[0.1em] outline-none transition-all flex items-center justify-between group/sel shadow-inner ${isModelDropdownOpen ? "border-indigo-500" : "border-white/5 hover:bg-zinc-900"}`}
                                                >
                                                    <div className="flex items-center gap-4 truncate">
                                                        <Cpu className="h-4 w-4 text-indigo-400" />
                                                        <span className="text-zinc-100 truncate">{modelName || "SELECT MODEL"}</span>
                                                    </div>
                                                    <ChevronDown className={`h-3 w-3 text-zinc-600 transition-transform ${isModelDropdownOpen ? "rotate-180" : ""}`} />
                                                </button>

                                                {isModelDropdownOpen && (
                                                    <div className="absolute top-full left-0 right-0 mt-2 p-2 bg-zinc-950 border border-white/10 rounded-2xl backdrop-blur-3xl shadow-xl z-50">
                                                        <div className="max-h-[200px] overflow-y-auto custom-scrollbar flex flex-col gap-1 p-1">
                                                            {modelsLoading ? (
                                                                <div className="p-4 text-[9px] text-zinc-500 font-black uppercase text-center flex flex-col items-center gap-2">
                                                                    <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
                                                                    Loading Models...
                                                                </div>
                                                            ) : localModels.map((name) => (
                                                                <button
                                                                    key={name}
                                                                    onClick={() => {
                                                                        setAiScanner((prev) => ({ ...prev, modelName: name }));
                                                                        setIsModelDropdownOpen(false);
                                                                    }}
                                                                    className={`w-full flex items-center justify-between px-4 py-3 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${modelName === name ? "bg-indigo-600 text-white" : "text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-100"}`}
                                                                >
                                                                    <span>{name}</span>
                                                                    {modelName === name && <Check className="h-3 w-3" />}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Section 3: Model Visibility Management */}
                                <div className="rounded-[2rem] border border-white/5 bg-white/[0.02] p-6 space-y-6 hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                            <div className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Model Visibility Control</div>
                                        </div>
                                        <Globe className="h-4 w-4 text-green-500/50" />
                                    </div>

                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                                        {modelsLoading ? (
                                            <div className="col-span-full flex items-center justify-center p-8">
                                                <Loader2 className="h-6 w-6 animate-spin text-zinc-700" />
                                            </div>
                                        ) : localModels.map((m) => (
                                            <button
                                                key={m}
                                                onClick={() => toggleModelVisibility(m)}
                                                className={`flex items-center justify-between p-3.5 rounded-xl border transition-all text-left group/m ${adminConfig?.enabledModels?.includes(m)
                                                    ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400"
                                                    : "bg-zinc-950/50 border-white/5 text-zinc-600 hover:border-white/20"
                                                    }`}
                                            >
                                                <div className="flex flex-col gap-0.5">
                                                    <span className="text-[10px] font-black uppercase tracking-wider truncate max-w-[120px]">{m.replace(".pkl", "")}</span>
                                                    <span className="text-[8px] font-bold opacity-50 uppercase">{adminConfig?.enabledModels?.includes(m) ? "Publicly Visible" : "Admin Only"}</span>
                                                </div>
                                                <div className={`w-8 h-4 rounded-full transition-all relative ${adminConfig?.enabledModels?.includes(m) ? "bg-indigo-600" : "bg-zinc-800"}`}>
                                                    <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${adminConfig?.enabledModels?.includes(m) ? "left-4" : "left-1"}`} />
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <div className="flex flex-col items-center gap-6">
                                <div className="flex flex-wrap items-center justify-center gap-4">
                                    {results.length > 0 && (
                                        <button
                                            onClick={() => clearAiScannerView()}
                                            disabled={loading}
                                            className="h-12 px-6 rounded-xl border border-white/5 bg-zinc-900/50 text-[10px] font-black uppercase tracking-widest text-zinc-600 hover:text-red-400 transition-all"
                                        >
                                            Clear Results
                                        </button>
                                    )}

                                    {results.length > 0 && (
                                        <button
                                            onClick={() => saveCurrentScan()}
                                            disabled={loading}
                                            className="h-12 px-6 rounded-xl border border-white/5 bg-indigo-600/10 text-[10px] font-black uppercase tracking-widest text-indigo-400 hover:bg-indigo-600 hover:text-white transition-all flex items-center gap-2"
                                        >
                                            <Bookmark className="h-4 w-4" />
                                            Save to History
                                        </button>
                                    )}

                                    {loading ? (
                                        <button
                                            onClick={stopAiScan}
                                            className="h-14 flex items-center gap-4 rounded-2xl bg-red-600/10 border border-red-500/20 px-10 text-xs font-black uppercase tracking-[0.2em] text-red-500 hover:bg-red-500/20 transition-all"
                                        >
                                            <div className="h-2 w-2 rounded-full bg-red-500 animate-ping" />
                                            STOP SCAN
                                        </button>
                                    ) : (
                                        <button
                                            onClick={runScan}
                                            className="h-16 flex items-center gap-6 rounded-2xl bg-indigo-600 px-16 text-xs font-black uppercase tracking-[0.3em] text-white shadow-lg shadow-indigo-600/20 hover:shadow-indigo-600/40 transition-all hover:-translate-y-1 active:scale-95"
                                        >
                                            <Brain className="h-5 w-5" />
                                            START AI SCAN
                                        </button>
                                    )}
                                </div>
                            </div>

                            {loading && (
                                <div className="w-full max-w-xl mx-auto space-y-4 py-4 animate-in fade-in slide-in-from-bottom-4">
                                    <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest">
                                        <span className="text-zinc-400">Scanning {country} Market...</span>
                                        <span className="text-indigo-400">{Math.round((progress.current / progress.total) * 100)}%</span>
                                    </div>
                                    <div className="h-2 w-full bg-zinc-900 rounded-full border border-white/5 overflow-hidden">
                                        <div
                                            className="h-full bg-indigo-600 transition-all duration-500"
                                            style={{ width: `${(progress.current / progress.total) * 100}%` }}
                                        />
                                    </div>
                                    <div className="text-center text-[8px] text-zinc-600 font-black uppercase tracking-widest animate-pulse">
                                        Analyzing {progress.current} of {progress.total} assets
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {(results.length > 0 || loading) && (
                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-in fade-in duration-1000">
                            <div className={`rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl ${selected && detailData ? "lg:col-span-8" : "lg:col-span-12"}`}>
                                <div className="px-6 py-4 border-b border-white/5 bg-zinc-950/80 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Activity className="w-4 h-4 text-emerald-500" />
                                        <h3 className="text-xs font-black text-white uppercase tracking-widest">Opportunities ({results.length})</h3>
                                    </div>
                                </div>
                                <div className="overflow-x-auto custom-scrollbar">
                                    <table className="w-full text-left text-xs whitespace-nowrap">
                                        <thead className="bg-zinc-950/80 text-[9px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                                            <tr>
                                                <th className="px-6 py-4">Symbol</th>
                                                <th className="px-6 py-4">Name</th>
                                                <th className="px-6 py-4 text-right">Price</th>
                                                <th className="px-6 py-4 text-center">Confidence</th>
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
                                                            <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="sm" />
                                                            <span className="font-mono font-black text-indigo-400">{r.symbol}</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-6 py-4">
                                                        <span className="text-[10px] font-bold text-zinc-400 uppercase truncate max-w-[150px] block">{r.name}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-right">
                                                        <span className="font-mono font-bold text-zinc-100">{r.last_close.toFixed(2)}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-center">
                                                        <span className={`px-2 py-1 rounded text-[9px] font-black ${r.precision > 0.7 ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"}`}>
                                                            {(r.precision * 100).toFixed(1)}%
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                                        <button
                                                            onClick={() => isSaved(r.symbol) ? removeSymbolBySymbol(r.symbol) : saveSymbol({ symbol: r.symbol, name: r.name, source: "ai_scanner", metadata: {} })}
                                                            className={`p-2 rounded-lg transition-all ${isSaved(r.symbol) ? "text-indigo-400 bg-indigo-500/10" : "text-zinc-600 hover:text-white"}`}
                                                        >
                                                            {isSaved(r.symbol) ? <BookmarkCheck className="h-4 w-4" /> : <Bookmark className="h-4 w-4" />}
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {selected && detailData && (
                                <div className="lg:col-span-4 space-y-6">
                                    <div className="rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl animate-in slide-in-from-right-4">
                                        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
                                            <div className="font-mono font-black text-white">{selected.symbol}</div>
                                            <button onClick={() => setAiScanner(prev => ({ ...prev, selected: null, detailData: null }))} className="p-1 text-zinc-500 hover:text-red-400"><X className="w-4 h-4" /></button>
                                        </div>
                                        <div className="p-4 space-y-4">
                                            <CandleChart rows={detailData.testPredictions} chartType={chartType} />
                                            <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </>
            ) : (
                <div className="relative rounded-[2.5rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-8 shadow-2xl min-h-[600px]">
                    <div className="space-y-8">
                        <div className="flex flex-col gap-2">
                            <h2 className="text-lg font-black text-white uppercase tracking-widest flex items-center gap-3">
                                <Activity className="h-5 w-5 text-indigo-500" />
                                Neural Scan Archive
                            </h2>
                            <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                                Track the historical performance of your automated AI scanners.
                            </p>
                        </div>

                        {historyLoading ? (
                            <div className="flex flex-col items-center justify-center py-24 gap-4">
                                <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Retrieving Secure Archives...</span>
                            </div>
                        ) : history.length === 0 ? (
                            <div className="text-center py-24 text-zinc-600 uppercase font-black text-xs tracking-widest border border-white/5 rounded-[2rem] bg-white/[0.01]">
                                No historical data found in Supabase.
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                {history.map((scan) => (
                                    <button
                                        key={scan.id}
                                        onClick={() => viewScanDetails(scan)}
                                        className={`flex flex-col gap-6 p-6 rounded-[2rem] border transition-all text-left group ${selectedHistoryScan?.id === scan.id ? "bg-indigo-600/10 border-indigo-500/30" : "bg-white/[0.02] border-white/5 hover:border-white/10 hover:bg-white/[0.04]"}`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-indigo-400">{scan.model_name}</span>
                                            <div className="px-2 py-1 rounded bg-black/40 border border-white/5 text-[8px] font-bold text-zinc-500 uppercase">
                                                {new Date(scan.created_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                        <div className="flex flex-col gap-1">
                                            <span className="text-base font-black text-white uppercase tracking-tight">{scan.country} MARKET</span>
                                            <div className="flex items-center gap-2 text-[10px] text-zinc-500 font-bold">
                                                <span>{scan.scanned_count} ASSETS ANALYZED</span>
                                                <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                <span>{(scan.duration_ms / 1000).toFixed(1)}S</span>
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}

                        {selectedHistoryScan && (
                            <div className="mt-12 space-y-6 border-t border-white/5 pt-10 animate-in fade-in slide-in-from-bottom-4">
                                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                                    <div className="flex flex-col gap-1">
                                        <h3 className="text-xl font-black text-white uppercase tracking-tighter">Results Distribution</h3>
                                        <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">{selectedHistoryScan.model_name} Neural Core Performance</span>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        <div className="flex items-center gap-4">
                                            <div className="flex flex-col items-end">
                                                <span className="text-[8px] font-black text-zinc-600 uppercase tracking-widest leading-none mb-1">Win Rate</span>
                                                <span className="text-sm font-black text-emerald-500 leading-none">
                                                    {(selectedHistoryResults.filter(r => r.status === 'win').length / (selectedHistoryResults.length || 1) * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="flex flex-col items-end border-l border-white/5 pl-4">
                                                <span className="text-[8px] font-black text-zinc-600 uppercase tracking-widest leading-none mb-1">Avg P/L</span>
                                                <span className={`text-sm font-black leading-none ${selectedHistoryResults.reduce((acc, r) => acc + (r.profit_loss_pct || 0), 0) / (selectedHistoryResults.length || 1) > 0 ? "text-emerald-500" : "text-red-500"}`}>
                                                    {(selectedHistoryResults.reduce((acc, r) => acc + (r.profit_loss_pct || 0), 0) / (selectedHistoryResults.length || 1)).toFixed(2)}%
                                                </span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={handleRefreshPerformance}
                                            disabled={refreshingPerformance}
                                            className="h-10 px-6 rounded-xl bg-white/5 border border-white/5 text-[9px] font-black uppercase tracking-widest text-zinc-400 hover:text-white hover:bg-indigo-600/20 transition-all flex items-center gap-2"
                                        >
                                            {refreshingPerformance ? <Loader2 className="h-3 w-3 animate-spin" /> : <Activity className="h-3 w-3" />}
                                            {refreshingPerformance ? "Updating..." : "Refresh Performance"}
                                        </button>
                                    </div>
                                </div>

                                <div className="rounded-[1.5rem] border border-white/5 overflow-hidden bg-black/40 shadow-2xl">
                                    <div className="overflow-x-auto custom-scrollbar">
                                        <table className="w-full text-left text-xs whitespace-nowrap">
                                            <thead className="bg-white/[0.03] text-[9px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                                                <tr>
                                                    <th className="px-8 py-5">Symbol</th>
                                                    <th className="px-8 py-5 text-center">Confidence</th>
                                                    <th className="px-8 py-5 text-right">Entry Price</th>
                                                    <th className="px-8 py-5 text-right">Target/Current</th>
                                                    <th className="px-8 py-5 text-right">Returns</th>
                                                    <th className="px-8 py-5 text-center">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5">
                                                {historyResultsLoading ? (
                                                    <tr>
                                                        <td colSpan={6} className="px-8 py-16 text-center">
                                                            <Loader2 className="h-6 w-6 animate-spin mx-auto text-indigo-500" />
                                                        </td>
                                                    </tr>
                                                ) : selectedHistoryResults.length === 0 ? (
                                                    <tr>
                                                        <td colSpan={6} className="px-8 py-16 text-center text-zinc-600 font-bold uppercase tracking-widest text-[10px]">No valid trade setups detected during this scan.</td>
                                                    </tr>
                                                ) : selectedHistoryResults.map((r) => (
                                                    <tr key={r.id} className="hover:bg-white/[0.01] transition-colors group">
                                                        <td className="px-8 py-5">
                                                            <span className="font-mono font-black text-indigo-400 group-hover:text-indigo-300 transition-colors uppercase">{r.symbol}</span>
                                                        </td>
                                                        <td className="px-8 py-5 text-center">
                                                            <span className={`px-2 py-1 rounded text-[9px] font-black ${r.precision > 0.75 ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"}`}>
                                                                {(r.precision * 100).toFixed(1)}%
                                                            </span>
                                                        </td>
                                                        <td className="px-8 py-5 text-right font-mono font-bold text-zinc-400">{r.last_close.toFixed(2)}</td>
                                                        <td className="px-8 py-5 text-right font-mono font-bold text-zinc-100">{r.exit_price?.toFixed(2) || "—"}</td>
                                                        <td className="px-8 py-5 text-right">
                                                            <span className={`font-mono font-black ${r.profit_loss_pct > 0 ? "text-emerald-500" : r.profit_loss_pct < 0 ? "text-red-500" : "text-zinc-500"}`}>
                                                                {r.profit_loss_pct !== null ? `${r.profit_loss_pct > 0 ? "+" : ""}${r.profit_loss_pct.toFixed(2)}%` : "0.00%"}
                                                            </span>
                                                        </td>
                                                        <td className="px-8 py-5 text-center">
                                                            <span className={`px-2.5 py-1 rounded-md text-[8px] font-black uppercase tracking-[0.15em] ${r.status === 'win' ? "bg-emerald-500 text-white" : r.status === 'loss' ? "bg-red-500 text-white" : "bg-zinc-800 text-zinc-400"}`}>
                                                                {r.status || 'open'}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            <CountrySelectDialog
                open={showCountryDialog}
                onClose={() => setShowCountryDialog(false)}
                countries={countries}
                selectedCountry={country}
                onSelect={(c) => setAiScanner(prev => ({ ...prev, country: c }))}
            />
        </div>
    );
}
