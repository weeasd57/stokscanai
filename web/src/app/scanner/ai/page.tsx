"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { Brain, AlertTriangle, Loader2, Globe, Bookmark, BookmarkCheck, Info, X, BarChart2, LineChart, Sliders, Activity, ChevronDown, Cpu, Check } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useAIScanner } from "@/contexts/AIScannerContext";
import { getLocalModels, predictStock, type ScanResult } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";

export default function AIScannerPage() {
    const { t } = useLanguage();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();
    const { countries } = useAppState();
    const { state, setAiScanner, runAiScan, stopAiScan, loading, error, clearAiScannerView, restoreLastAiScan } = useAIScanner();

    const { country, results, progress, hasScanned, showPrecisionInfo, selected, detailData, rfPreset, rfParamsJson, chartType, showEma50, showEma200, showBB, showRsi, showVolume, scanHistory, modelName } = state;
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [showAdvancedRf, setShowAdvancedRf] = useState(false);
    const [showCountryDialog, setShowCountryDialog] = useState(false);
    const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const [localModels, setLocalModels] = useState<string[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [modelsError, setModelsError] = useState<string | null>(null);
    const [showMobileDetail, setShowMobileDetail] = useState(false);
    const [nowTick, setNowTick] = useState(() => Date.now());

    const debugInfo = useMemo(() => {
        const started = state.lastScanStartedAt;
        const ended = state.lastScanEndedAt;
        const baseDuration = state.lastDurationMs;
        const liveDuration = started && loading ? Date.now() - started : null;
        const duration = liveDuration ?? baseDuration ?? null;
        return {
            loading,
            error,
            resultsCount: results.length,
            progress,
            modelName: modelName || "(none)",
            rfPreset,
            rfParamsJson,
            scanHistory: scanHistory.length,
            selectedSymbol: selected?.symbol || null,
            limit: state.limit,
            started,
            ended,
            duration,
        };
    }, [loading, error, results.length, progress, modelName, rfPreset, rfParamsJson, scanHistory.length, selected, state.limit, state.lastScanStartedAt, state.lastScanEndedAt, state.lastDurationMs, state]);

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

    function fmtDuration(ms: number | null): string {
        if (ms === null || ms === undefined) return "-";
        if (ms < 1000) return `${ms} ms`;
        const s = ms / 1000;
        return `${s.toFixed(1)} s`;
    }

    function fmtTime(ts: number | null): string {
        if (!ts) return "-";
        const d = new Date(ts);
        return d.toLocaleTimeString();
    }

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

    function mergeRfJson(nextPartial: Record<string, unknown>) {
        setAiScanner((prev) => {
            let base: Record<string, unknown> = {};
            try {
                const parsed = JSON.parse((prev as any).rfParamsJson || "{}");
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                    base = parsed;
                }
            } catch {
            }
            return { ...prev, rfParamsJson: JSON.stringify({ ...base, ...nextPartial }) };
        });
    }

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
        // Always open dialog view as well
        setShowMobileDetail(true);
    }

    useEffect(() => {
        if (!selected) {
            setShowMobileDetail(false);
        }
    }, [selected]);

    async function runScan() {
        if (showAdvancedRf && rfParamsJson && parsedRfParams === null) {
            return;
        }
        await runAiScan({ rfParams: effectiveRfParams ?? null, minPrecision: 0.6 });
    }

    return (
        <div className="flex flex-col gap-8 pb-20 max-w-[1600px] mx-auto">
            <header className="flex flex-col gap-2">
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


            <div className="relative rounded-[3rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-10 shadow-[0_0_100px_rgba(0,0,0,0.5)]">
                {/* mesh background elements - wrapped in a container that can have overflow-hidden without clipping the dropdown */}
                <div className="absolute inset-0 rounded-[3rem] overflow-hidden pointer-events-none">
                    <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-600/15 blur-[120px] rounded-full animate-pulse" />
                    <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-blue-600/10 blur-[100px] rounded-full animate-pulse [animation-delay:2s]" />
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.03)_0%,transparent_70%)]" />
                </div>

                <div className="relative z-10 space-y-10">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Section 1: Market Configuration */}
                        <div className="group rounded-[2.5rem] border border-white/5 bg-white/[0.02] p-8 space-y-6 flex flex-col justify-between hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                                    <div className="text-[11px] font-black text-zinc-400 uppercase tracking-[0.2em]">{t("ai.market_config") || "Market Configuration"}</div>
                                </div>
                                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20">
                                    <Globe className="h-3 w-3 text-blue-500" />
                                    <span className="text-[9px] font-bold text-blue-400 uppercase tracking-tighter">Live Market</span>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <button
                                    onClick={() => setShowCountryDialog(true)}
                                    disabled={loading}
                                    className="h-16 flex items-center justify-between gap-4 rounded-2xl border border-white/5 bg-zinc-950/50 px-6 text-sm font-bold text-zinc-200 hover:bg-zinc-900 hover:border-white/20 transition-all group/btn shadow-inner"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="p-2.5 rounded-xl bg-zinc-900 group-hover/btn:bg-zinc-800 transition-colors border border-white/5">
                                            <Globe className="h-4 w-4 text-zinc-400 group-hover/btn:text-blue-500 transition-colors" />
                                        </div>
                                        <div className="flex flex-col items-start">
                                            <span className="text-[9px] text-zinc-500 font-black uppercase tracking-widest leading-none mb-1">Region</span>
                                            <span className="uppercase tracking-[0.1em] text-white">{country}</span>
                                        </div>
                                    </div>
                                    <ChevronDown className="h-4 w-4 text-zinc-600" />
                                </button>

                                <div className={`h-16 px-6 rounded-2xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-4 transition-all shadow-inner ${state.scanAllMarket !== false ? "opacity-30 grayscale cursor-not-allowed" : "hover:border-white/20"}`}>
                                    <div className="flex flex-col">
                                        <span className="text-[9px] font-black uppercase tracking-widest text-zinc-500 leading-none mb-1">Depth Limit</span>
                                        <span className="text-xs font-bold text-zinc-300">SYMBOLS</span>
                                    </div>
                                    <input
                                        type="number"
                                        min={1}
                                        max={400}
                                        value={state.limit}
                                        onChange={(e) => setAiScanner((prev) => ({ ...prev, limit: Math.min(400, Math.max(1, Number(e.target.value) || 1)) }))}
                                        disabled={loading || state.scanAllMarket !== false}
                                        className="w-16 h-10 rounded-xl bg-zinc-900/80 border border-white/10 px-3 text-sm font-mono text-indigo-400 outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all text-center disabled:cursor-not-allowed shadow-xl"
                                    />
                                </div>
                            </div>

                            <label className="h-16 px-6 rounded-2xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-6 cursor-pointer group/toggle hover:bg-zinc-900 transition-all shadow-inner border-dashed">
                                <div className="flex items-center gap-3">
                                    <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                                        <Activity className="h-4 w-4 text-indigo-400" />
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-[10px] font-black uppercase tracking-widest text-zinc-300">Scan Full Market</span>
                                        <span className="text-[9px] text-zinc-500 font-bold uppercase tracking-tighter">RECOMMENDED FOR MAX COVERAGE</span>
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
                                    <div className="w-12 h-6 bg-zinc-800 rounded-full peer peer-checked:bg-indigo-600 transition-all border border-white/5 group-hover/toggle:border-white/10 overflow-hidden relative">
                                        <div className="absolute inset-x-0 bottom-0 top-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />
                                    </div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-zinc-200 rounded-full peer-checked:translate-x-6 peer-checked:bg-white transition-all shadow-lg ring-1 ring-black/10"></div>
                                </div>
                            </label>
                        </div>

                        {/* Section 2: AI Configuration */}
                        <div className="group rounded-[2.5rem] border border-white/5 bg-white/[0.02] p-8 space-y-6 flex flex-col justify-between hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500 relative">
                            <div className="flex items-center justify-between relative z-10">
                                <div className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                                    <div className="text-[11px] font-black text-zinc-400 uppercase tracking-[0.2em]">{t("ai.model_config") || "Model & Intelligence"}</div>
                                </div>
                                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                                    <Brain className="h-3 w-3 text-indigo-500" />
                                    <span className="text-[9px] font-bold text-indigo-400 uppercase tracking-tighter">AI Active</span>
                                </div>
                            </div>

                            <div className="relative z-10 space-y-4">
                                <div className="space-y-2">
                                    <span className="text-[9px] font-black text-zinc-500 uppercase tracking-widest ml-1">Selection Neural Core</span>
                                    <div className="relative flex-1 w-full" ref={dropdownRef}>
                                        <button
                                            type="button"
                                            onClick={() => !loading && !modelsLoading && setIsModelDropdownOpen(!isModelDropdownOpen)}
                                            className={`w-full h-16 rounded-2xl bg-zinc-950/50 border px-6 text-[11px] font-black uppercase tracking-[0.15em] outline-none transition-all flex items-center justify-between group/sel shadow-inner ${isModelDropdownOpen ? "border-indigo-500 shadow-indigo-500/10" : "border-white/5 hover:border-white/20 hover:bg-zinc-900"}`}
                                        >
                                            <div className="flex items-center gap-4 truncate">
                                                <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 group-hover/sel:scale-110 transition-transform">
                                                    <Cpu className="h-4 w-4 text-indigo-400" />
                                                </div>
                                                <div className="flex flex-col items-start truncate">
                                                    <span className="text-[9px] text-indigo-500/60 font-black uppercase tracking-tighter leading-none mb-1">Architecture</span>
                                                    <span className="text-zinc-100 truncate">{modelName || "SELECT LOCAL DATASET"}</span>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <div className="h-4 w-px bg-white/5" />
                                                <ChevronDown className={`h-4 w-4 text-zinc-600 transition-transform duration-500 ${isModelDropdownOpen ? "rotate-180 text-indigo-400" : ""}`} />
                                            </div>
                                        </button>

                                        {isModelDropdownOpen && (
                                            <div className="absolute top-full left-0 right-0 mt-3 p-2 bg-zinc-950 border border-white/10 rounded-3xl backdrop-blur-3xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] z-[200] animate-in fade-in slide-in-from-top-4 duration-500 ring-1 ring-white/10">
                                                <div className="max-h-[260px] overflow-y-auto custom-scrollbar flex flex-col gap-1.5 p-1 relative z-10">
                                                    {modelsLoading ? (
                                                        <div className="px-4 py-8 text-[10px] text-zinc-500 font-black uppercase tracking-widest text-center flex flex-col items-center gap-2">
                                                            <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
                                                            Initializing Core...
                                                        </div>
                                                    ) : modelsError ? (
                                                        <div className="px-4 py-8 text-[10px] text-red-400 font-bold uppercase tracking-widest text-center">
                                                            {modelsError}
                                                        </div>
                                                    ) : localModels.length === 0 ? (
                                                        <div className="px-4 py-8 text-[10px] text-zinc-500 font-bold uppercase tracking-widest text-center italic border border-dashed border-white/5 rounded-2xl">
                                                            <div className="mb-2 flex justify-center opacity-20"><Brain className="h-8 w-8" /></div>
                                                            No local models available
                                                        </div>
                                                    ) : (
                                                        localModels.map((name) => (
                                                            <button
                                                                key={name}
                                                                onClick={() => {
                                                                    setAiScanner((prev) => ({ ...prev, modelName: name }));
                                                                    setIsModelDropdownOpen(false);
                                                                }}
                                                                className={`group/item w-full flex items-center justify-between px-5 py-4 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${modelName === name
                                                                    ? "bg-indigo-600 text-white shadow-xl shadow-indigo-600/30 border border-indigo-400/50"
                                                                    : "text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-100 hover:border-white/10 border border-transparent"
                                                                    }`}
                                                            >
                                                                <div className="flex items-center gap-4">
                                                                    <div className={`w-2 h-2 rounded-full transition-all duration-300 ${modelName === name ? "bg-white scale-125 shadow-[0_0_10px_#fff]" : "bg-indigo-500/30 group-hover/item:bg-indigo-500"}`} />
                                                                    <span className="truncate">{name}</span>
                                                                </div>
                                                                {modelName === name && (
                                                                    <div className="flex items-center gap-2">
                                                                        <Check className="h-3 w-3" />
                                                                    </div>
                                                                )}
                                                            </button>
                                                        ))
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                                <div className="text-[8px] text-zinc-500 font-bold uppercase tracking-widest leading-relaxed px-1">
                                    {modelName ? `Utilizing ${modelName} optimized for ${country} market dynamics.` : "Select a pre-trained Random Forest model to begin advanced high-probability analysis."}
                                </div>
                            </div>

                            {/* Decorative Grid */}
                            <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
                        </div>
                    </div>

                    <div className="flex flex-col items-center gap-6 pt-4">
                        <div className="flex flex-wrap items-center justify-center gap-4">
                            {results.length > 0 && (
                                <button
                                    type="button"
                                    onClick={() => clearAiScannerView()}
                                    disabled={loading}
                                    className="h-14 px-8 rounded-2xl border border-white/5 bg-zinc-900/50 text-[11px] font-black uppercase tracking-widest text-zinc-600 hover:text-red-400 hover:bg-red-400/5 hover:border-red-400/10 transition-all"
                                >
                                    {t("tech.clear_results")}
                                </button>
                            )}

                            {results.length === 0 && scanHistory.length > 0 && (
                                <button
                                    type="button"
                                    onClick={() => void restoreLastAiScan()}
                                    disabled={loading}
                                    className="h-14 px-8 rounded-2xl border border-white/5 bg-zinc-900/50 text-[11px] font-black uppercase tracking-widest text-zinc-600 hover:text-indigo-400 hover:bg-indigo-400/5 hover:border-indigo-400/10 transition-all"
                                >
                                    {t("tech.restore_last")}
                                </button>
                            )}

                            {loading ? (
                                <button
                                    onClick={stopAiScan}
                                    className="h-20 flex items-center gap-6 rounded-[2.5rem] bg-red-600/10 border border-red-500/20 px-12 text-[14px] font-black uppercase tracking-[0.4em] text-red-500 hover:bg-red-500/20 transition-all duration-500 shadow-[0_0_40px_rgba(239,68,68,0.1)] group relative overflow-hidden"
                                >
                                    <div className="absolute -inset-1 opacity-20 blur-xl bg-red-500 animate-pulse" />
                                    <div className="relative z-10 flex items-center gap-4">
                                        <div className="h-4 w-4 rounded-full bg-red-500 animate-ping" />
                                        {t("ai.stop_scan") || "STOP ANALYSIS"}
                                    </div>
                                </button>
                            ) : (
                                <button
                                    onClick={runScan}
                                    className="h-20 group relative flex items-center gap-6 rounded-[2.5rem] bg-indigo-600 px-20 text-[14px] font-black uppercase tracking-[0.4em] text-white shadow-[0_0_60px_rgba(99,102,241,0.4)] hover:shadow-[0_0_80px_rgba(99,102,241,0.6)] transition-all duration-500 hover:-translate-y-2 overflow-hidden active:scale-95"
                                >
                                    <div className="absolute inset-x-0 bottom-0 top-0 bg-gradient-to-tr from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                                    <div className="absolute -inset-1 opacity-0 group-hover:opacity-100 transition-opacity blur-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 animate-pulse" />

                                    <div className="relative z-10 flex items-center gap-6">
                                        <div className="p-2.5 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 group-hover:rotate-[30deg] transition-transform duration-700">
                                            <Brain className="h-6 w-6" />
                                        </div>
                                        <div className="flex flex-col items-start leading-none pt-1">
                                            <span className="text-[10px] text-white/60 font-black tracking-widest mb-1 opacity-0 group-hover:opacity-100 transition-all -translate-y-2 group-hover:translate-y-0 duration-500">INITIATE DEEP ANALYSIS</span>
                                            {t("ai.start_scan")}
                                        </div>
                                    </div>

                                    {/* Scanning Beam effect */}
                                    <div className="absolute top-0 bottom-0 left-0 w-12 bg-white/20 blur-xl skew-x-12 -translate-x-32 group-hover:translate-x-[600px] transition-all duration-[1.5s] ease-in-out" />
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {loading && (
                    <div className="flex flex-col items-center justify-center pt-10 gap-8 relative z-10 animate-in fade-in slide-in-from-bottom-5 duration-700">
                        <div className="w-full max-w-2xl px-6 py-8 rounded-[2.5rem] bg-indigo-500/5 border border-indigo-500/10 backdrop-blur-md relative overflow-hidden group">
                            {/* Scanning Beam Animation */}
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-indigo-500/10 to-transparent -translate-x-full animate-[shimmer_3s_infinite]" />

                            <div className="flex flex-col items-center gap-6 relative z-10">
                                <div className="flex items-center gap-6 w-full">
                                    <div className="relative">
                                        <div className="p-4 rounded-2xl bg-indigo-600 shadow-2xl shadow-indigo-500/40 relative z-10">
                                            <Loader2 className="h-8 w-8 animate-spin text-white" />
                                        </div>
                                        <div className="absolute inset-0 blur-2xl bg-indigo-600/60 animate-pulse scale-150" />
                                    </div>

                                    <div className="flex-1 space-y-3">
                                        <div className="flex items-center justify-between">
                                            <div className="flex flex-col">
                                                <span className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-2">
                                                    Deep Scanning {country} Market
                                                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-indigo-500 text-[8px] font-black text-white uppercase">AI LIVE</span>
                                                </span>
                                                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-[0.2em] mt-1 italic">
                                                    Applying Random Forest weights for probability analysis...
                                                </span>
                                            </div>
                                            <div className="text-right">
                                                <span className="text-2xl font-mono font-black text-indigo-400">
                                                    {progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0}%
                                                </span>
                                            </div>
                                        </div>

                                        <div className="w-full bg-zinc-900/80 rounded-full h-3.5 overflow-hidden border border-white/10 ring-1 ring-white/5 shadow-inner">
                                            <div
                                                className="bg-[linear-gradient(45deg,rgba(99,102,241,1)_25%,rgba(129,140,248,1)_50%,rgba(99,102,241,1)_75%)] bg-[length:40px_40px] h-full transition-all duration-1000 ease-out shadow-[0_0_20px_rgba(99,102,241,0.6)] relative overflow-hidden animate-[progress-stripe_2s_linear_infinite]"
                                                style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
                                            >
                                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-4">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]" />
                                                    <span className="text-[9px] font-black text-zinc-400 uppercase tracking-widest">PROBABILITY CHECK</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_5px_#f59e0b]" />
                                                    <span className="text-[9px] font-black text-zinc-400 uppercase tracking-widest">PATTERN RECOGNITION</span>
                                                </div>
                                            </div>
                                            <span className="text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em]">
                                                <span className="text-indigo-400">{progress.current}</span> / {progress.total} SYMBOLS
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Analysis Steps Ticker */}
                        <div className="flex items-center gap-3 px-8 py-3 rounded-2xl bg-white/[0.02] border border-white/5 backdrop-blur-xl shadow-2xl">
                            <Activity className="h-4 w-4 text-indigo-500 animate-pulse" />
                            <div className="flex items-center gap-2 overflow-hidden max-w-md whitespace-nowrap">
                                <span className="text-[9px] font-black uppercase tracking-[0.3em] text-zinc-400 animate-in fade-in slide-in-from-right-10 duration-1000">
                                    {progress.current % 3 === 0 ? "CALCULATING TECHNICAL MOMENTUM DEVIATION" :
                                        progress.current % 3 === 1 ? "EXTRACTING FUNDAMENTAL VARIANCE VECTORS" :
                                            "VALIDATING CROSS-EXCHANGE LIQUIDITY DEPTH"}
                                </span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <CountrySelectDialog
                open={showCountryDialog}
                onClose={() => setShowCountryDialog(false)}
                countries={countries}
                selectedCountry={country}
                onSelect={(c) => setAiScanner(prev => ({ ...prev, country: c }))}
            />

            {(results.length > 0 || loading) && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-6 duration-1000">

                    {/* Results Table */}
                    <div className={`overflow-hidden rounded-[2rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl shadow-2xl flex flex-col ${selected && detailData ? "lg:col-span-8" : "lg:col-span-12"}`}>
                        <div className="bg-zinc-950/80 px-8 py-5 border-b border-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Activity className="w-4 h-4 text-emerald-500" />
                                <h3 className="text-sm font-black text-white uppercase tracking-widest">{t("ai.matches")}</h3>
                            </div>
                            <button
                                type="button"
                                onClick={() => setAiScanner(prev => ({ ...prev, showPrecisionInfo: !prev.showPrecisionInfo }))}
                                className="inline-flex items-center gap-2 rounded-xl border border-white/5 bg-zinc-900/50 px-4 py-2 text-[10px] font-black uppercase tracking-widest text-zinc-400 hover:text-white transition-all"
                            >
                                <Info className="h-3.5 w-3.5 text-indigo-500" />
                                Stats Key
                            </button>
                        </div>

                        {showPrecisionInfo && (
                            <div className="px-8 py-4 text-[11px] font-medium text-zinc-500 bg-indigo-500/5 border-b border-indigo-500/10 leading-relaxed italic">
                                {t("ai.precision_info")}
                            </div>
                        )}

                        <div className="overflow-x-auto custom-scrollbar">
                            <table className="w-full text-left text-sm whitespace-nowrap">
                                <thead className="bg-zinc-950/80 text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500 border-b border-white/5">
                                    <tr>
                                        <th className="px-8 py-5">{t("ai.table.symbol")}</th>
                                        <th className="px-6 py-5">{t("ai.table.name")}</th>
                                        <th className="px-6 py-5 text-right">{t("ai.table.price")}</th>
                                        <th className="px-6 py-5 text-center">{t("ai.table.precision")}</th>
                                        <th className="px-8 py-5 text-right">{t("ai.table.save")}</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    {results.map((r) => (
                                        <tr
                                            key={r.symbol}
                                            onClick={() => openDetails(r)}
                                            className={`cursor-pointer group transition-all ${selected?.symbol === r.symbol ? "bg-indigo-600/10" : "hover:bg-white/[0.02]"}`}
                                        >
                                            <td className="px-8 py-5">
                                                <div className="flex items-center gap-4">
                                                    <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="md" />
                                                    <span className="font-mono font-black text-indigo-400 group-hover:text-indigo-300 transition-colors">{r.symbol}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-5">
                                                <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-tighter truncate max-w-[200px] block">{r.name}</span>
                                            </td>
                                            <td className="px-6 py-5 text-right">
                                                <span className="font-mono font-black text-zinc-100">{r.last_close.toFixed(2)}</span>
                                            </td>
                                            <td className="px-6 py-5 text-center">
                                                <div className={`inline-flex items-center h-7 px-3 rounded-lg text-[10px] font-black border tracking-widest ${r.precision > 0.7
                                                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                                    : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                                                    }`}>
                                                    {(r.precision * 100).toFixed(1)}%
                                                </div>
                                            </td>
                                            <td className="px-8 py-5 text-right" onClick={(e) => e.stopPropagation()}>
                                                <button
                                                    onClick={() => {
                                                        if (isSaved(r.symbol)) {
                                                            removeSymbolBySymbol(r.symbol);
                                                        } else {
                                                            saveSymbol({
                                                                symbol: r.symbol,
                                                                name: r.name,
                                                                source: "ai_scanner",
                                                                metadata: {
                                                                    precision: r.precision,
                                                                    last_close: r.last_close,
                                                                    logo_url: r.logo_url
                                                                }
                                                            });
                                                        }
                                                    }}
                                                    className={`p-2.5 rounded-xl transition-all ${isSaved(r.symbol)
                                                        ? "text-indigo-400 bg-indigo-500/10"
                                                        : "text-zinc-600 hover:text-white hover:bg-zinc-800"
                                                        }`}
                                                >
                                                    {isSaved(r.symbol) ? <BookmarkCheck className="h-4.5 w-4.5" /> : <Bookmark className="h-4.5 w-4.5" />}
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <div className="bg-zinc-950/80 px-8 py-3 text-[9px] font-black uppercase tracking-[0.2em] text-zinc-600 border-t border-white/5 flex justify-between">
                            <span>{t("dash.scanned").replace("{count}", progress.current.toString())}</span>
                            <span>{results.length} OPPORTUNITIES FILTERED</span>
                        </div>
                    </div>

                    {/* Details Panel - only show when data is loaded */}
                    {selected && detailData && (
                        <div className="hidden lg:block lg:col-span-4 rounded-[2rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl shadow-2xl overflow-hidden h-fit lg:sticky lg:top-8 animate-in slide-in-from-right-8 duration-700">
                            <div className="px-6 py-5 border-b border-white/5 flex items-center justify-between bg-zinc-950/60">
                                <div className="space-y-0.5 min-w-0">
                                    <div className="text-xl font-black text-white font-mono tracking-tighter truncate">{selected.symbol}</div>
                                    <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest truncate">{selected.name}</div>
                                </div>
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, selected: null, detailData: null }))}
                                    className="p-2.5 rounded-xl text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all"
                                >
                                    <X className="h-4.5 w-4.5" />
                                </button>
                            </div>

                            <div className="p-6 space-y-8">
                                {/* Chart Controls */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest italic">{t("ai.chart_ctrl")}</span>
                                        <div className="flex gap-1.5 p-1 rounded-xl bg-zinc-900/50 border border-white/5">
                                            <button
                                                onClick={() => setAiScanner(prev => ({ ...prev, chartType: "candle" }))}
                                                className={`p-2 rounded-lg transition-all ${chartType === "candle" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-zinc-600 hover:text-zinc-400"}`}
                                            >
                                                <BarChart2 className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => setAiScanner(prev => ({ ...prev, chartType: "area" }))}
                                                className={`p-2 rounded-lg transition-all ${chartType === "area" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-zinc-600 hover:text-zinc-400"}`}
                                            >
                                                <LineChart className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        {[
                                            { id: "showEma50", label: "EMA 50", color: "bg-orange-500" },
                                            { id: "showEma200", label: "EMA 200", color: "bg-cyan-500" },
                                            { id: "showBB", label: "BB", color: "bg-purple-500" },
                                            { id: "showRsi", label: "RSI", color: "bg-pink-500" },
                                        ].map((ind) => (
                                            <label key={ind.id} className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-zinc-900/30 border border-white/5 cursor-pointer hover:bg-zinc-800 transition-all group">
                                                <div className={`w-3 h-3 rounded-full border border-white/10 ${state[ind.id as keyof typeof state] ? ind.color : "bg-transparent"}`} />
                                                <input
                                                    type="checkbox"
                                                    className="hidden"
                                                    checked={!!state[ind.id as keyof typeof state]}
                                                    onChange={(e) => setAiScanner(prev => ({ ...prev, [ind.id]: e.target.checked }))}
                                                />
                                                <span className="text-[9px] font-black uppercase tracking-widest text-zinc-500 group-hover:text-zinc-300">{ind.label}</span>
                                            </label>
                                        ))}
                                    </div>
                                </div>

                                {detailError && (
                                    <div className="p-4 rounded-2xl bg-red-500/5 border border-red-500/20 flex flex-col items-center gap-2 text-center">
                                        <AlertTriangle className="h-5 w-5 text-red-500" />
                                        <p className="text-[11px] font-bold text-red-400/80">{detailError}</p>
                                    </div>
                                )}

                                {detailLoading && (
                                    <div className="flex flex-col items-center justify-center py-20 gap-4">
                                        <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                                        <p className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Building visual report...</p>
                                    </div>
                                )}

                                {detailData && (
                                    <article className="space-y-6 animate-in fade-in duration-500">
                                        <div className="rounded-2xl border border-white/5 bg-zinc-900/20 overflow-hidden">
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
                                    </article>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

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
                                        height={500}
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
