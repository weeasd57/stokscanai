"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { Brain, AlertTriangle, Loader2, Globe, Info, X, BarChart2, LineChart, Cpu, Check, Activity, ChevronDown, Settings, Trash2, Archive, ListChecks, Filter, CheckSquare, Square } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useAIScanner } from "@/contexts/AIScannerContext";
import { getLocalModels, predictStock, getAdminConfig, updateAdminConfig, type ScanResult, type AdminConfig, type LocalModelMeta } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";
import { getAiScore } from "@/lib/utils";

function ModelVisibilityItem({
    m,
    isEnabled,
    alias,
    onToggle,
    onAliasChange
}: {
    m: string | LocalModelMeta;
    isEnabled: boolean;
    alias: string;
    onToggle: () => void;
    onAliasChange: (val: string) => void;
}) {
    const meta = typeof m === "string" ? null : m;
    const name = typeof m === "string" ? m : m.name;
    const [localAlias, setLocalAlias] = useState(alias);

    useEffect(() => {
        setLocalAlias(alias);
    }, [alias]);

    return (
        <div
            className={`flex flex-col gap-5 p-6 rounded-[2rem] border transition-all duration-500 group/item ${isEnabled
                ? "bg-indigo-600/[0.03] border-indigo-500/20 shadow-[0_8px_30px_rgba(79,70,229,0.04)]"
                : "bg-zinc-950/40 border-white/5 grayscale opacity-60 hover:opacity-100 hover:grayscale-0 hover:bg-zinc-900/40"
                }`}
        >
            <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-center gap-2 mb-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${isEnabled ? "bg-indigo-500 animate-pulse" : "bg-zinc-700"}`} />
                        <span className="text-[8px] font-black text-zinc-500 uppercase tracking-[0.3em]">AI Core Engine</span>
                    </div>
                    <h3 className="text-xs font-black text-white uppercase tracking-widest truncate group-hover/item:text-indigo-400 transition-colors">
                        {name}
                    </h3>
                </div>
                <button
                    onClick={onToggle}
                    className={`group/btn w-12 h-6 rounded-full transition-all relative border-2 shrink-0 ${isEnabled ? "bg-indigo-600 border-indigo-400" : "bg-zinc-900 border-white/10"}`}
                >
                    <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all shadow-md ${isEnabled ? "left-6.5" : "left-0.5"}`} />
                </button>
            </div>

            {meta && (
                <div className="grid grid-cols-2 gap-3 py-4 border-y border-white/[0.03]">
                    <div className="flex flex-col">
                        <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Architecture</span>
                        <span className="text-[9px] font-bold text-zinc-400 uppercase tracking-tighter">{meta.type || 'BOOSTER'}</span>
                    </div>
                    <div className="flex flex-col items-end text-right">
                        <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Features</span>
                        <span className="text-[9px] font-mono font-bold text-indigo-400/80 tracking-tighter">{meta.num_features || '—'}</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Parameters</span>
                        <span className="text-[9px] font-mono font-bold text-zinc-400 tracking-tighter">{meta.num_parameters || '—'}</span>
                    </div>
                    <div className="flex flex-col items-end text-right">
                        <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Samples</span>
                        <span className="text-[9px] font-mono font-bold text-zinc-400 tracking-tighter">{meta.trainingSamples || '—'}</span>
                    </div>
                    {/* Sniper Strategy Settings */}
                    {meta.target_pct != null && (
                        <>
                            <div className="flex flex-col col-span-2 pt-2 mt-2 border-t border-white/[0.03]">
                                <span className="text-[7px] font-black text-emerald-600 uppercase tracking-tighter mb-1.5">Sniper Strategy</span>
                            </div>
                            <div className="flex flex-col">
                                <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Target</span>
                                <span className="text-[9px] font-mono font-bold text-emerald-400 tracking-tighter">+{((meta.target_pct ?? 0.15) * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex flex-col items-end text-right">
                                <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Stop Loss</span>
                                <span className="text-[9px] font-mono font-bold text-rose-400 tracking-tighter">-{((meta.stop_loss_pct ?? 0.05) * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex flex-col col-span-2">
                                <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter mb-0.5">Look Forward</span>
                                <span className="text-[9px] font-mono font-bold text-zinc-400 tracking-tighter">{meta.look_forward_days ?? 20} Days</span>
                            </div>
                        </>
                    )}
                </div>
            )}

            <div className="space-y-2">
                <span className="text-[7px] font-black text-zinc-700 uppercase tracking-widest ml-1">Public Display Identity</span>
                <div className="relative group/input">
                    <input
                        type="text"
                        placeholder="Assign an alias..."
                        className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-[10px] font-black uppercase tracking-[0.15em] text-white placeholder:text-zinc-800 w-full outline-none focus:border-indigo-500/50 focus:bg-zinc-900/50 transition-all shadow-inner"
                        value={localAlias}
                        onChange={(e) => setLocalAlias(e.target.value)}
                        onBlur={() => localAlias !== alias && onAliasChange(localAlias)}
                    />
                    {localAlias !== alias && (
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1.5 bg-indigo-500/10 px-2 py-1 rounded-md border border-indigo-500/20">
                            <div className="w-1 h-1 rounded-full bg-indigo-500 animate-pulse" />
                            <span className="text-[6px] font-black text-indigo-500 uppercase tracking-tighter">UNSAVED</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default function FastScannerTab() {
    const { t } = useLanguage();
    const { countries } = useAppState();
    const { state, setAiScanner, runAiScan, stopAiScan, loading, error, clearAiScannerView, restoreLastAiScan, saveCurrentScan } = useAIScanner();

    const { country, results, progress, hasScanned, showPrecisionInfo, selected, detailData, rfPreset, rfParamsJson, chartType, showEma50, showEma200, showBB, showRsi, showVolume, showMacd, scanHistory, modelName } = state;
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
    const [localModels, setLocalModels] = useState<(string | LocalModelMeta)[]>([]);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [modelsError, setModelsError] = useState<string | null>(null);
    const [showMobileDetail, setShowMobileDetail] = useState(false);
    const [nowTick, setNowTick] = useState(() => Date.now());
    const [adminConfig, setAdminConfig] = useState<AdminConfig | null>(null);
    const [historyFilters, setHistoryFilters] = useState({
        country: "",
        model: "",
        startDate: "",
        endDate: ""
    });
    const [localScanDays, setLocalScanDays] = useState(450);
    const [isPublishing, setIsPublishing] = useState(false);
    const [publishStatus, setPublishStatus] = useState<"idle" | "success" | "error">("idle");
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [showModelVisibility, setShowModelVisibility] = useState(false);
    const [refreshingAll, setRefreshingAll] = useState(false);
    const [publishedResults, setPublishedResults] = useState<ScanResult[]>([]);
    const [publishedLoading, setPublishedLoading] = useState(false);
    const [globalStats, setGlobalStats] = useState<{ winRate: number; avgPl: number; total: number }>({ winRate: 0, avgPl: 0, total: 0 });
    const [globalStatsLoading, setGlobalStatsLoading] = useState(false);

    // Sorting States
    const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
    const [pubSortConfig, setPubSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);

    const handleSort = (key: string) => {
        setSortConfig(prev => {
            if (prev?.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: 'desc' }; // Default to desc for metrics
        });
    };

    const handlePubSort = (key: string) => {
        setPubSortConfig(prev => {
            if (prev?.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: 'desc' };
        });
    };

    const sortedResults = useMemo(() => {
        if (!sortConfig) return results;
        return [...results].sort((a: any, b: any) => {
            let aVal = a[sortConfig.key];
            let bVal = b[sortConfig.key];

            // Special handling for nested or derived values
            if (sortConfig.key === 'precision') {
                aVal = Number(a.precision);
                bVal = Number(b.precision);
            }

            if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
            if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [results, sortConfig]);

    const sortedPublishedResults = useMemo(() => {
        if (!pubSortConfig) return publishedResults;
        return [...publishedResults].sort((a: any, b: any) => {
            let aVal = a[pubSortConfig.key];
            let bVal = b[pubSortConfig.key];

            if (pubSortConfig.key === 'profit_loss_pct') {
                aVal = a.profit_loss_pct || 0;
                bVal = b.profit_loss_pct || 0;
            } else if (pubSortConfig.key === 'confidence' || pubSortConfig.key === 'precision') {
                aVal = Number(a.confidence || a.precision || 0);
                bVal = Number(b.confidence || b.precision || 0);
            } else if (pubSortConfig.key === 'technical_score') {
                aVal = Number(a.technical_score || 0);
                bVal = Number(b.technical_score || 0);
            }

            if (aVal < bVal) return pubSortConfig.direction === 'asc' ? -1 : 1;
            if (aVal > bVal) return pubSortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [publishedResults, pubSortConfig]);

    const handleToggleSelect = (id: string) => {
        setSelectedIds(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
    };

    const handleSelectAll = (results: ScanResult[]) => {
        if (selectedIds.length === results.length) {
            setSelectedIds([]);
        } else {
            setSelectedIds(results.map(r => r.id).filter((id): id is string => !!id));
        }
    };

    const handleRefreshAll = async () => {
        if (refreshingAll || publishedResults.length === 0) return;
        setRefreshingAll(true);
        try {
            // Find unique batch IDs that have at least one 'open' or 'pending' result
            const openBatches = Array.from(new Set(
                publishedResults
                    .filter(r => !r.status || r.status === 'open' || r.status === 'pending')
                    .map(r => (r as any).batch_id)
                    .filter(Boolean)
            ));

            if (openBatches.length === 0) {
                // If no 'open' results, just refresh everything published recently? 
                // Let's just do all unique batches in the current view to be safe
                const allBatches = Array.from(new Set(
                    publishedResults
                        .map(r => (r as any).batch_id)
                        .filter(Boolean)
                )) as string[];

                for (const bId of allBatches) {
                    await refreshScanPerformance(bId);
                }
            } else {
                for (const bId of openBatches as string[]) {
                    await refreshScanPerformance(bId);
                }
            }
            await loadPublishedResults();
        } catch (err) {
            console.error("Refresh all failed:", err);
        } finally {
            setRefreshingAll(false);
        }
    };

    const handleBulkUnpublish = async () => {
        if (selectedIds.length === 0) return;
        if (!confirm(`Are you sure you want to unpublish ${selectedIds.length} signals?`)) return;

        try {
            const ok = await bulkUpdatePublicStatus(selectedIds, false);
            if (ok) {
                setSelectedIds([]);
                loadPublishedResults();
            }
        } catch (err) {
            console.error("Bulk unpublish failed:", err);
        }
    };


    // Context Integration
    const {
        saveSelectedResults,
        toggleResultPublicStatus,
        bulkUpdatePublicStatus,
        fetchPublicScanDates,
        fetchScanResultsByDate,
        fetchPublishedResults,
        fetchScanHistory,
        fetchScanResults,
        refreshScanPerformance,
        updateResultStatus,
        fetchGlobalModelStats
    } = useAIScanner();

    useEffect(() => {
        if (modelName) {
            setGlobalStatsLoading(true);
            fetchGlobalModelStats(modelName).then(stats => {
                setGlobalStats(stats);
                setGlobalStatsLoading(false);
            }).catch(err => {
                console.error("Failed to fetch global stats:", err);
                setGlobalStatsLoading(false);
            });
        }
    }, [modelName, fetchGlobalModelStats]);

    useEffect(() => {
        getAdminConfig().then(cfg => {
            setAdminConfig(cfg);
            if (cfg.scanDays) setLocalScanDays(cfg.scanDays);
        }).catch(console.error);
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

    const updateModelAlias = async (mName: string, alias: string) => {
        if (!adminConfig) return;
        const nextAliases = { ...(adminConfig.modelAliases || {}), [mName]: alias };
        try {
            const updated = await updateAdminConfig({ modelAliases: nextAliases });
            setAdminConfig(updated);
        } catch (err) {
            console.error("Failed to update alias:", err);
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
                setLocalModels(data);
                if (!modelName && data.length > 0) {
                    const first = data[0];
                    const firstName = typeof first === "string" ? first : first.name;
                    setAiScanner((prev) => ({ ...prev, modelName: firstName }));
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
                rfPreset: rfPreset,
                rfParams: effectiveRfParams ?? undefined,
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


    async function loadHistory() {
        setHistoryLoading(true);
        try {
            const data = await fetchScanHistory({
                country: historyFilters.country || undefined,
                model: historyFilters.model || undefined
            });
            setHistory(data);
        } catch (err) {
            console.error("Failed to load history:", err);
        } finally {
            setHistoryLoading(false);
        }
    }

    const loadPublishedResults = async () => {
        setPublishedLoading(true);
        try {
            const data = await fetchPublishedResults({
                country: historyFilters.country || undefined,
                model: historyFilters.model || undefined,
                startDate: historyFilters.startDate || undefined,
                endDate: historyFilters.endDate || undefined
            });
            setPublishedResults(data);
        } catch (err) {
            console.error("Failed to load published results:", err);
        } finally {
            setPublishedLoading(false);
        }
    };

    useEffect(() => {
        if (viewMode === "history") {
            loadPublishedResults();
        }
    }, [viewMode, historyFilters.country, historyFilters.model, historyFilters.startDate, historyFilters.endDate]);

    const handleUnpublish = async (id: string | undefined) => {
        if (!id) return;
        if (confirm("Are you sure you want to unpublish this result?")) {
            await toggleResultPublicStatus(id, false);
            loadPublishedResults();
        }
    };

    const filteredHistory = useMemo(() => {
        return history.filter(h => {
            const matchCountry = !historyFilters.country || h.country === historyFilters.country;
            const matchModel = !historyFilters.model || h.model_name === historyFilters.model;

            let matchDate = true;
            if (historyFilters.startDate || historyFilters.endDate) {
                const createdAt = new Date(h.created_at).getTime();
                if (historyFilters.startDate) {
                    matchDate = matchDate && createdAt >= new Date(historyFilters.startDate).getTime();
                }
                if (historyFilters.endDate) {
                    // Set to end of day
                    const end = new Date(historyFilters.endDate);
                    end.setHours(23, 59, 59, 999);
                    matchDate = matchDate && createdAt <= end.getTime();
                }
            }

            return matchCountry && matchModel && matchDate;
        });
    }, [history, historyFilters]);

    async function viewScanDetails(scan: any) {
        setSelectedHistoryScan(scan);
        setHistoryResultsLoading(true);
        const resultsRes = await fetchScanResults(scan.batch_id);
        setSelectedHistoryResults(resultsRes);
        setHistoryResultsLoading(false);
    }

    async function handleRefreshPerformance() {
        if (!selectedHistoryScan) return;
        setRefreshingPerformance(true);
        try {
            await refreshScanPerformance(selectedHistoryScan.batch_id);
            // Re-fetch results
            const resultsRes = await fetchScanResults(selectedHistoryScan.batch_id);
            setSelectedHistoryResults(resultsRes);
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
        await runAiScan({
            rfParams: effectiveRfParams ?? null,
            minPrecision: state.buyThreshold,
            shouldSave: false,
            buy_threshold: state.buyThreshold,
            target_pct: state.targetPct,
            stop_loss_pct: state.stopLossPct,
            look_forward_days: state.lookForwardDays,
            councilModel: state.councilModel,
            validatorModel: state.validatorModel
        });
    }

    return (
        <div className="flex flex-col gap-8 p-6 lg:p-10 max-w-[1800px] mx-auto">
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
                <div className="flex items-center gap-4 bg-zinc-950/50 p-1.5 rounded-2xl border border-white/5 md:w-80">
                    <button
                        onClick={() => setViewMode("scan")}
                        className={`flex-1 px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === "scan" ? "bg-white text-black shadow-xl" : "text-zinc-500 hover:text-white"}`}
                    >
                        RUN SCAN
                    </button>
                    <button
                        onClick={() => setViewMode("history")}
                        className={`flex-1 px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === "history" ? "bg-white text-black shadow-xl" : "text-zinc-500 hover:text-white"}`}
                    >
                        PUBLISHED
                    </button>
                </div>
            </header>

            {viewMode === "scan" && modelName && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {[
                        { label: "Neural Win Rate", value: `${globalStats.winRate.toFixed(1)}%`, icon: <Check className="h-5 w-5" />, color: "text-emerald-500", bg: "bg-emerald-500/10" },
                        { label: "Avg Potential P/L", value: `${globalStats.avgPl > 0 ? '+' : ''}${globalStats.avgPl.toFixed(2)}%`, icon: <Activity className="h-5 w-5" />, color: globalStats.avgPl > 0 ? "text-emerald-500" : "text-red-500", bg: globalStats.avgPl > 0 ? "bg-emerald-500/10" : "bg-red-500/10" },
                        { label: "Discovery hits", value: globalStats.total, icon: <Cpu className="h-5 w-5" />, color: "text-indigo-400", bg: "bg-indigo-500/10" },
                    ].map((s, idx) => (
                        <div key={idx} className="relative group overflow-hidden rounded-[2.5rem] border border-white/5 bg-zinc-950/40 backdrop-blur-2xl p-6 flex flex-col gap-3 hover:border-white/10 transition-all">
                            <div className="absolute inset-x-0 bottom-0 top-0 bg-gradient-to-tr from-white/[0.01] to-transparent pointer-events-none" />
                            <div className="flex items-center justify-between relative z-10">
                                <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">{s.label}</span>
                                <div className={`p-2 rounded-xl ${s.bg} ${s.color}`}>
                                    {s.icon}
                                </div>
                            </div>
                            <div className="flex flex-col gap-0.5 relative z-10">
                                <div className={`text-2xl font-black italic tracking-tighter ${s.color}`}>
                                    {globalStatsLoading ? "---" : s.value}
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-[8px] font-black text-zinc-500 uppercase tracking-[0.2em]">All-Time Performance</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

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
                                        <button
                                            onClick={() => setShowModelVisibility(true)}
                                            className="p-2 rounded-xl bg-zinc-950/50 border border-white/5 text-zinc-500 hover:text-indigo-400 hover:border-indigo-500/30 transition-all"
                                            title="Model Visibility Settings"
                                        >
                                            <Settings className="h-3 w-3" />
                                        </button>
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

                                    <div className="flex flex-col gap-1.5 w-full">
                                        <div className="flex items-center justify-between ml-1">
                                            <span className="text-[8px] font-black uppercase tracking-widest text-zinc-500">Scan Reference Date (End)</span>
                                            <span className="text-[7px] font-black text-indigo-500 uppercase tracking-tighter">
                                                Auto-Calculates {adminConfig?.scanDays || 450} Days History
                                            </span>
                                        </div>
                                        <input
                                            type="date"
                                            value={state.endDate}
                                            onChange={(e) => setAiScanner(prev => ({ ...prev, endDate: e.target.value }))}
                                            disabled={loading}
                                            className="h-10 rounded-xl bg-zinc-950/50 border border-white/5 px-3 text-[10px] font-mono text-zinc-300 outline-none focus:border-indigo-500 transition-all w-full"
                                        />
                                    </div>

                                    <div className="flex flex-col gap-4">
                                        <div className={`h-14 px-5 rounded-xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-4 transition-all shadow-inner ${state.scanAllHistory ? "opacity-30 grayscale cursor-not-allowed" : "hover:border-white/20"}`}>
                                            <div className="flex flex-col">
                                                <span className="text-[8px] font-black uppercase tracking-widest text-zinc-500 leading-none mb-1">Lookback Window</span>
                                                <span className="text-[10px] font-bold text-zinc-300 uppercase">Days of Data</span>
                                            </div>
                                            <input
                                                type="number"
                                                min={100}
                                                max={2000}
                                                value={localScanDays}
                                                onChange={(e) => setLocalScanDays(Number(e.target.value))}
                                                onBlur={() => {
                                                    const val = Math.min(2000, Math.max(100, localScanDays || 450));
                                                    setLocalScanDays(val);
                                                    updateAdminConfig({ scanDays: val }).then(cfg => {
                                                        setAdminConfig(cfg);
                                                        setAiScanner(prev => ({ ...prev, scanDays: val }));
                                                    }).catch(console.error);
                                                }}
                                                disabled={loading || state.scanAllHistory}
                                                className="w-16 h-8 rounded-lg bg-zinc-900 border border-white/10 px-2 text-xs font-mono text-indigo-400 outline-none text-center"
                                            />
                                        </div>

                                        <label className="h-10 px-4 rounded-xl border border-white/5 bg-zinc-950/50 flex items-center justify-between gap-6 cursor-pointer group/toggle hover:bg-zinc-900 transition-all shadow-inner">
                                            <div className="flex items-center gap-3">
                                                <div className="flex flex-col">
                                                    <span className="text-[9px] font-black uppercase tracking-widest text-indigo-400">Scan All Life</span>
                                                </div>
                                            </div>
                                            <div className="relative">
                                                <input
                                                    type="checkbox"
                                                    checked={state.scanAllHistory}
                                                    onChange={(e) => setAiScanner((prev) => ({ ...prev, scanAllHistory: e.target.checked }))}
                                                    disabled={loading}
                                                    className="sr-only peer"
                                                />
                                                <div className="w-8 h-4 bg-zinc-800 rounded-full peer peer-checked:bg-indigo-600 transition-all border border-white/5" />
                                                <div className="absolute left-0.5 top-0.5 w-3 h-3 bg-zinc-200 rounded-full peer-checked:translate-x-4 transition-all"></div>
                                            </div>
                                        </label>
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
                                                            ) : localModels.map((m) => {
                                                                const nameRes = typeof m === "string" ? m : m.name;
                                                                return (
                                                                    <button
                                                                        key={nameRes}
                                                                        onClick={() => {
                                                                            setAiScanner((prev) => ({ ...prev, modelName: nameRes }));
                                                                            setIsModelDropdownOpen(false);
                                                                        }}
                                                                        className={`w-full flex items-center justify-between px-4 py-3 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${modelName === nameRes ? "bg-indigo-600 text-white" : "text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-100"}`}
                                                                    >
                                                                        <span>{nameRes}</span>
                                                                        {modelName === nameRes && <Check className="h-3 w-3" />}
                                                                    </button>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">The Council</span>
                                                <select
                                                    value={state.councilModel}
                                                    onChange={(e) => setAiScanner(prev => ({ ...prev, councilModel: e.target.value }))}
                                                    disabled={loading}
                                                    className="w-full h-14 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-indigo-400 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                                >
                                                    <option value="" className="bg-zinc-900 italic opacity-50">-- DISABLED --</option>
                                                    {localModels.map(m => {
                                                        const name = typeof m === "string" ? m : m.name;
                                                        return <option key={name} value={name} className="bg-zinc-900">{name}</option>;
                                                    })}
                                                </select>
                                            </div>
                                            <div className="space-y-2">
                                                <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Validator</span>
                                                <select
                                                    value={state.validatorModel}
                                                    onChange={(e) => setAiScanner(prev => ({ ...prev, validatorModel: e.target.value }))}
                                                    disabled={loading}
                                                    className="w-full h-14 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                                >
                                                    <option value="" className="bg-zinc-900 italic opacity-50">-- DISABLED --</option>
                                                    {localModels.map(m => {
                                                        const name = typeof m === "string" ? m : m.name;
                                                        return <option key={name} value={name} className="bg-zinc-900">{name}</option>;
                                                    })}
                                                </select>
                                            </div>
                                        </div>

                                    </div>
                                </div>

                                {/* Section 3: Strategy Configuration */}
                                <div className="group rounded-[2rem] border border-white/5 bg-white/[0.02] p-6 space-y-6 flex flex-col justify-between hover:bg-white/[0.04] hover:border-white/10 transition-all duration-500">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                            <div className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Strategy Settings</div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Sensitivity</span>
                                            <select
                                                value={state.buyThreshold}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, buyThreshold: Number(e.target.value) }))}
                                                disabled={loading}
                                                className="w-full h-12 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-indigo-400 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                            >
                                                {[0.10, 0.20, 0.30, 0.40, 0.45, 0.50].map(v => (
                                                    <option key={v} value={v} className="bg-zinc-900">{(100 - v * 100).toFixed(0)}% (prob {v})</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="space-y-2">
                                            <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Look Forward</span>
                                            <select
                                                value={state.lookForwardDays}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, lookForwardDays: Number(e.target.value) }))}
                                                disabled={loading}
                                                className="w-full h-12 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-zinc-200 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                            >
                                                {[10, 15, 20, 30].map(v => (
                                                    <option key={v} value={v} className="bg-zinc-900">{v} Days</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="space-y-2">
                                            <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Target %</span>
                                            <select
                                                value={state.targetPct}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, targetPct: Number(e.target.value) }))}
                                                disabled={loading}
                                                className="w-full h-12 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-emerald-400 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                            >
                                                {[0.05, 0.10, 0.15, 0.20, 0.30].map(v => (
                                                    <option key={v} value={v} className="bg-zinc-900">{(v * 100).toFixed(0)}%</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="space-y-2">
                                            <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest ml-1">Stop Loss %</span>
                                            <select
                                                value={state.stopLossPct}
                                                onChange={(e) => setAiScanner(prev => ({ ...prev, stopLossPct: Number(e.target.value) }))}
                                                disabled={loading}
                                                className="w-full h-12 rounded-xl bg-zinc-950/50 border border-white/5 px-4 text-[10px] font-black uppercase tracking-widest text-rose-400 outline-none focus:border-indigo-500 transition-all shadow-inner appearance-none"
                                            >
                                                {[0.03, 0.05, 0.07, 0.10].map(v => (
                                                    <option key={v} value={v} className="bg-zinc-900">{(v * 100).toFixed(0)}%</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex flex-col items-center gap-8 py-10 border-t border-white/5">
                                <div className="flex flex-wrap items-center justify-center gap-6">
                                    {results.length > 0 && (
                                        <div className="flex flex-col items-center gap-2">
                                            <span className="text-[9px] font-black text-zinc-600 uppercase tracking-widest">Global Action</span>
                                            <button
                                                onClick={async () => {
                                                    setIsPublishing(true);
                                                    setPublishStatus("idle");
                                                    const success = await saveCurrentScan(true);
                                                    setIsPublishing(false);
                                                    if (success) {
                                                        setPublishStatus("success");
                                                        setTimeout(() => setPublishStatus("idle"), 4000);
                                                    } else {
                                                        setPublishStatus("error");
                                                        setTimeout(() => setPublishStatus("idle"), 4000);
                                                    }
                                                }}
                                                disabled={loading || isPublishing}
                                                className={`h-10 px-6 rounded-xl border transition-all flex items-center gap-3 text-[10px] uppercase font-black tracking-widest ${publishStatus === "success"
                                                    ? "bg-emerald-600 border-emerald-500 text-white"
                                                    : publishStatus === "error"
                                                        ? "bg-red-600 border-red-500 text-white"
                                                        : "border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"
                                                    }`}
                                            >
                                                <Globe className="h-4 w-4" />
                                                Archive & Publish All
                                            </button>
                                        </div>
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
                        </div>
                    </div>



                    {loading && (
                        <div className="w-full max-w-xl mx-auto space-y-4 py-4 animate-in fade-in slide-in-from-bottom-4">
                            <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest">
                                <span className="text-zinc-400">Scanning {country} Market...</span>
                                <span className="text-indigo-400">
                                    {progress.total > 0 ? `${Math.round((progress.current / progress.total) * 100)}%` : 'Loading...'}
                                </span>
                            </div>
                            <div className="h-2 w-full bg-zinc-900 rounded-full border border-white/5 overflow-hidden">
                                <div
                                    className="h-full bg-indigo-600 transition-all duration-500"
                                    style={{ width: progress.total > 0 ? `${(progress.current / progress.total) * 100}%` : '0%' }}
                                />
                            </div>
                            <div className="text-center text-[8px] text-zinc-600 font-black uppercase tracking-widest animate-pulse">
                                {progress.total > 0
                                    ? `Analyzing ${progress.current} of ${progress.total} assets`
                                    : 'Fetching symbol count...'}
                            </div>
                        </div>
                    )}

                    {(results.length > 0 || loading) && (
                        <div className="flex flex-col gap-6 animate-in fade-in duration-1000">
                            {selectedIds.length > 0 && (
                                <div className="flex items-center justify-between px-6 py-4 rounded-2xl bg-indigo-600 border border-indigo-400 shadow-xl shadow-indigo-600/20 animate-in slide-in-from-top-4">
                                    <div className="flex items-center gap-4">
                                        <div className="bg-white/20 p-2 rounded-lg text-white">
                                            <Check className="w-4 h-4" />
                                        </div>
                                        <span className="text-[11px] font-black text-white uppercase tracking-widest">{selectedIds.length} Assets Selected</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={async () => {
                                                setIsPublishing(true);
                                                const selectedResults = results.filter(r => selectedIds.includes(r.symbol));
                                                const success = await saveSelectedResults(selectedResults, true);
                                                setIsPublishing(false);
                                                if (success) {
                                                    setSelectedIds([]);
                                                    setPublishStatus("success");
                                                    setTimeout(() => setPublishStatus("idle"), 4000);
                                                } else {
                                                    setPublishStatus("error");
                                                    setTimeout(() => setPublishStatus("idle"), 4000);
                                                }
                                            }}
                                            disabled={isPublishing}
                                            className="px-6 py-2 rounded-xl bg-white text-indigo-600 text-[10px] font-black uppercase tracking-widest hover:bg-zinc-100 transition-all disabled:opacity-50"
                                        >
                                            {isPublishing ? "Publishing..." : "Publish Selected"}
                                        </button>
                                        <button
                                            onClick={() => setSelectedIds([])}
                                            className="p-2 text-white/60 hover:text-white transition-all"
                                        >
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            )}

                            <div className="rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl">
                                <div className="px-6 py-4 border-b border-white/5 bg-zinc-950/80 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Activity className="w-4 h-4 text-emerald-500" />
                                        <h3 className="text-xs font-black text-white uppercase tracking-widest">Opportunities ({results.length})</h3>
                                    </div>
                                    <button
                                        onClick={() => {
                                            if (selectedIds.length === results.length) setSelectedIds([]);
                                            else setSelectedIds(results.map(r => r.symbol));
                                        }}
                                        className="text-[9px] font-black text-indigo-400 uppercase tracking-widest hover:text-indigo-300"
                                    >
                                        {selectedIds.length === results.length ? "Deselect All" : "Select All"}
                                    </button>
                                </div>
                                <div className="overflow-x-auto custom-scrollbar">
                                    <table className="w-full text-left text-xs whitespace-nowrap">
                                        <thead className="bg-zinc-950/80 text-[9px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                                            <tr>
                                                <th className="px-6 py-4 w-10 sticky left-0 z-20 bg-zinc-950/90 backdrop-blur border-r border-white/5">
                                                    {/* Spacer for checkbox */}
                                                </th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('symbol')}>
                                                    <div className="flex items-center gap-1">
                                                        Symbol
                                                        {sortConfig?.key === 'symbol' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4">Name</th>
                                                <th className="px-6 py-4 text-right cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('last_close')}>
                                                    <div className="flex items-center justify-end gap-1">
                                                        Price
                                                        {sortConfig?.key === 'last_close' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-right text-emerald-500 cursor-pointer hover:text-emerald-400 transition-colors" onClick={() => handleSort('target_price')}>
                                                    <div className="flex items-center justify-end gap-1">
                                                        Target
                                                        {sortConfig?.key === 'target_price' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-right text-red-500 cursor-pointer hover:text-red-400 transition-colors" onClick={() => handleSort('stop_loss')}>
                                                    <div className="flex items-center justify-end gap-1">
                                                        Stop Loss
                                                        {sortConfig?.key === 'stop_loss' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('precision')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        AI Confidence
                                                        {sortConfig?.key === 'precision' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('technical_score')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        Technical
                                                        {sortConfig?.key === 'technical_score' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('fundamental_score')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        Fundamental
                                                        {sortConfig?.key === 'fundamental_score' && (sortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center text-indigo-400">Council</th>
                                                <th className="px-6 py-4 text-center text-indigo-400">Consensus</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5">
                                            {sortedResults.map((r) => (
                                                <tr
                                                    key={r.symbol}
                                                    className={`group transition-all ${selected?.symbol === r.symbol ? "bg-indigo-600/10" : "hover:bg-white/[0.02]"}`}
                                                >
                                                    <td className={`px-6 py-4 sticky left-0 z-10 border-r border-white/5 backdrop-blur ${selected?.symbol === r.symbol ? "bg-indigo-600/10" : "bg-zinc-950/90 group-hover:bg-white/[0.02]"}`}>
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedIds.includes(r.symbol)}
                                                            onChange={(e) => {
                                                                if (e.target.checked) setSelectedIds(prev => [...prev, r.symbol]);
                                                                else setSelectedIds(prev => prev.filter(id => id !== r.symbol));
                                                            }}
                                                            className="w-4 h-4 rounded border-white/10 bg-zinc-900 text-indigo-600 focus:ring-indigo-500"
                                                        />
                                                    </td>
                                                    <td className="px-6 py-4 cursor-pointer" onClick={() => openDetails(r)}>
                                                        <div className="flex items-center gap-3">
                                                            <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="sm" />
                                                            <span className="font-mono font-black text-indigo-400">{r.symbol}</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-6 py-4 cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="text-[10px] font-bold text-zinc-400 uppercase truncate max-w-[150px] block">{r.name}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-right cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="font-mono font-bold text-zinc-100">{r.last_close.toFixed(2)}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-right cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="font-mono font-bold text-emerald-400">{r.target_price ? r.target_price.toFixed(2) : "-"}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-right cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="font-mono font-bold text-red-400">{r.stop_loss ? r.stop_loss.toFixed(2) : "-"}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-center cursor-pointer" onClick={() => openDetails(r)}>
                                                        {(() => {
                                                            const { score, label, color, bg } = getAiScore(Number(r.precision));
                                                            return (
                                                                <div className="flex flex-col items-center gap-0.5">
                                                                    <div className={`px-2 py-0.5 rounded ${bg} ${color} text-[9px] font-black`}>
                                                                        {score}/10
                                                                    </div>
                                                                    <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter">{label}</span>
                                                                </div>
                                                            );
                                                        })()}
                                                    </td>
                                                    <td className="px-6 py-4 text-center cursor-pointer" onClick={() => openDetails(r)}>
                                                        {(() => {
                                                            const score = r.technical_score || 0;
                                                            const color = score >= 7 ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" :
                                                                score >= 4 ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
                                                                    "text-red-400 bg-red-500/10 border-red-500/20";
                                                            return (
                                                                <div className={`px-2 py-0.5 rounded border ${color} text-[9px] font-black inline-block`}>
                                                                    {score}/10
                                                                </div>
                                                            );
                                                        })()}
                                                    </td>
                                                    <td className="px-6 py-4 text-center cursor-pointer" onClick={() => openDetails(r)}>
                                                        {(() => {
                                                            const score = r.fundamental_score || 0;
                                                            const color = score >= 7 ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" :
                                                                score >= 4 ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
                                                                    "text-red-400 bg-red-500/10 border-red-500/20";
                                                            return (
                                                                <div className={`px-2 py-0.5 rounded border ${color} text-[9px] font-black inline-block`}>
                                                                    {score}/10
                                                                </div>
                                                            );
                                                        })()}
                                                    </td>
                                                    <td className="px-6 py-4 text-center cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="font-black text-white italic">{`${(r.council_score ?? 0).toFixed(1)}%`}</span>
                                                    </td>
                                                    <td className="px-6 py-4 text-center cursor-pointer" onClick={() => openDetails(r)}>
                                                        <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">{r.consensus_ratio || "0/0"}</span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Full Screen Chart Dialog */}
                    {showMobileDetail && selected && (
                        <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl flex flex-col animate-in fade-in duration-300">
                            <div className="h-20 border-b border-white/10 flex items-center justify-between px-6 bg-zinc-950/50">
                                <div className="flex flex-col">
                                    <div className="flex items-center gap-3">
                                        <StockLogo symbol={selected.symbol} logoUrl={selected.logo_url} size="sm" />
                                        <h3 className="text-xl font-black text-white italic tracking-tighter flex items-center gap-2">
                                            {selected.symbol}
                                            <span className={`px-2 py-0.5 rounded bg-white/5 text-[10px] text-zinc-500 font-bold uppercase`}>
                                                {selected.exchange}
                                            </span>
                                        </h3>
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

                            <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                                <div className="max-w-[1400px] mx-auto space-y-8">
                                    <div className="min-h-[60vh] rounded-3xl border border-white/5 bg-black/40 overflow-y-auto relative shadow-2xl custom-scrollbar">
                                        {detailLoading ? (
                                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                                                <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                                <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest animate-pulse italic">Synchronizing Neural Datashards...</div>
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
                                                showMacd={showMacd}
                                            />
                                        ) : null}
                                    </div>

                                    <div className="flex flex-col gap-10">
                                        {/* Reasons Section - ABOVE */}
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
                                            {detailData?.topReasons && detailData.topReasons.length > 0 ? (
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
                                                <div className="text-[10px] font-black text-zinc-700 uppercase italic">Limited explainability data available for this spectral slice.</div>
                                            )}
                                        </div>

                                        {/* Table Data Preview - BELOW */}
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
                                                {detailData?.testPredictions && (
                                                    <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            ) : (
                <div className="relative rounded-[2.5rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-8 shadow-2xl min-h-[600px]">
                    <div className="space-y-8">
                        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                            <div className="flex flex-col gap-2">
                                <h2 className="text-lg font-black text-white uppercase tracking-widest flex items-center gap-3">
                                    <Activity className="h-5 w-5 text-indigo-500" />
                                    Signals Management
                                </h2>
                                <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                                    Monitor and manage published AI signals across all markets.
                                </p>
                            </div>

                            <div className="flex flex-wrap items-center gap-3">
                                {/* Consolidated Filters */}
                                <div className="flex flex-wrap items-center gap-4 w-full lg:w-auto">
                                    <div className="flex flex-wrap items-center gap-3 flex-1 lg:flex-none">
                                        <select
                                            className="bg-zinc-950/50 border border-white/5 rounded-xl px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-zinc-400 outline-none focus:border-indigo-500/50 transition-all flex-1 lg:flex-none min-w-[140px]"
                                            value={historyFilters.country}
                                            onChange={(e) => setHistoryFilters(prev => ({ ...prev, country: e.target.value }))}
                                        >
                                            <option value="">All Regions</option>
                                            {countries.map(c => (
                                                <option key={c} value={c}>{c}</option>
                                            ))}
                                        </select>
                                        <select
                                            className="bg-zinc-950/50 border border-white/5 rounded-xl px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-zinc-400 outline-none focus:border-indigo-500/50 transition-all flex-1 lg:flex-none min-w-[140px]"
                                            value={historyFilters.model}
                                            onChange={(e) => setHistoryFilters(prev => ({ ...prev, model: e.target.value }))}
                                        >
                                            <option value="">All Neural Cores</option>
                                            {localModels.map(m => {
                                                const name = typeof m === "string" ? m : m.name;
                                                return <option key={name} value={name}>{name}</option>;
                                            })}
                                        </select>
                                    </div>

                                    <div className="flex items-center gap-1 bg-zinc-950/50 border border-white/5 rounded-xl px-2 flex-1 lg:flex-none">
                                        <Filter className="w-3 h-3 text-zinc-600 ml-1" />
                                        <input
                                            type="date"
                                            className="bg-transparent border-none text-[8px] font-black text-zinc-400 outline-none p-2 flex-1"
                                            value={historyFilters.startDate}
                                            onChange={(e) => setHistoryFilters(prev => ({ ...prev, startDate: e.target.value }))}
                                        />
                                        <span className="text-zinc-800">-</span>
                                        <input
                                            type="date"
                                            className="bg-transparent border-none text-[8px] font-black text-zinc-400 outline-none p-2 flex-1"
                                            value={historyFilters.endDate}
                                            onChange={(e) => setHistoryFilters(prev => ({ ...prev, endDate: e.target.value }))}
                                        />
                                    </div>

                                    <div className="flex items-center gap-2 w-full lg:w-auto">
                                        <button
                                            onClick={handleRefreshAll}
                                            disabled={publishedLoading || refreshingAll}
                                            className="flex-1 lg:flex-none px-6 py-2.5 rounded-xl bg-indigo-600 shadow-xl shadow-indigo-600/20 text-white text-[10px] font-black uppercase tracking-widest hover:bg-indigo-500 transition-all flex items-center justify-center gap-2"
                                            title="Re-evaluate all open published signals"
                                        >
                                            {refreshingAll ? <Loader2 className="h-3 w-3 animate-spin" /> : <Activity className="h-3 w-3" />}
                                            <span>{refreshingAll ? "Syncing..." : "Refresh Status"}</span>
                                        </button>

                                        {selectedIds.length > 0 && (
                                            <button
                                                onClick={handleBulkUnpublish}
                                                className="flex-1 lg:flex-none px-6 py-2.5 rounded-xl bg-red-600 shadow-xl shadow-red-600/20 text-white text-[10px] font-black uppercase tracking-widest hover:bg-red-500 transition-all flex items-center justify-center gap-2 animate-in slide-in-from-right-2"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                                <span>Unpublish {selectedIds.length}</span>
                                            </button>
                                        )}
                                    </div>
                                </div>


                            </div>
                        </div>

                        {publishedLoading ? (
                            <div className="flex flex-col items-center justify-center py-24 gap-4">
                                <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Syncing Published Assets...</span>
                            </div>
                        ) : publishedResults.length === 0 ? (
                            <div className="text-center py-24 text-zinc-600 uppercase font-black text-xs tracking-widest border border-white/5 rounded-[2rem] bg-white/[0.01]">
                                No published results found with current filters.
                            </div>
                        ) : (
                            <div className="rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-xl">
                                <div className="overflow-x-auto custom-scrollbar">
                                    <table className="w-full text-left text-xs whitespace-nowrap">
                                        <thead className="bg-zinc-950/80 text-[9px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                                            <tr>
                                                <th className="px-6 py-4 w-10 sticky left-0 z-20 bg-zinc-950/90 backdrop-blur border-r border-white/5">
                                                    <button
                                                        onClick={() => handleSelectAll(publishedResults)}
                                                        className="text-zinc-600 hover:text-white transition-colors flex items-center justify-center"
                                                    >
                                                        {selectedIds.length === publishedResults.length && publishedResults.length > 0 ? (
                                                            <CheckSquare className="w-4 h-4 text-indigo-500" />
                                                        ) : (
                                                            <Square className="w-4 h-4" />
                                                        )}
                                                    </button>
                                                </th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-white transition-colors" onClick={() => handlePubSort('symbol')}>
                                                    <div className="flex items-center gap-1">
                                                        Symbol
                                                        {pubSortConfig?.key === 'symbol' && (pubSortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4">Name</th>
                                                <th className="px-6 py-4 text-right">Entry Price</th>
                                                <th className="px-6 py-4 text-right">Last Price</th>
                                                <th className="px-6 py-4 text-right text-emerald-500">Target</th>
                                                <th className="px-6 py-4 text-right text-emerald-500/50">T. %</th>
                                                <th className="px-6 py-4 text-right text-red-500">SL</th>
                                                <th className="px-6 py-4 text-right text-red-500/50">L. %</th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handlePubSort('confidence')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        AI Confidence
                                                        {pubSortConfig?.key === 'confidence' && (pubSortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handlePubSort('technical_score')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        Technical
                                                        {pubSortConfig?.key === 'technical_score' && (pubSortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center text-indigo-400">Council</th>
                                                <th className="px-6 py-4 text-center text-indigo-400">Consensus</th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handlePubSort('created_at')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        Date
                                                        {pubSortConfig?.key === 'created_at' && (pubSortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-center">Last Checked</th>
                                                <th className="px-6 py-4 text-center cursor-pointer hover:text-white transition-colors" onClick={() => handlePubSort('status')}>
                                                    <div className="flex items-center justify-center gap-1">
                                                        Status
                                                        {pubSortConfig?.key === 'status' && (pubSortConfig.direction === 'asc' ? <ChevronDown className="w-3 h-3 rotate-180" /> : <ChevronDown className="w-3 h-3" />)}
                                                    </div>
                                                </th>
                                                <th className="px-6 py-4 text-right">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5">
                                            {sortedPublishedResults.map((r, idx) => {
                                                const entry = r.last_close;
                                                const targetPct = r.target_price ? ((r.target_price - entry) / entry * 100).toFixed(1) : "-";
                                                const lossPct = r.stop_loss ? ((r.stop_loss - entry) / entry * 100).toFixed(1) : "-";

                                                return (
                                                    <tr
                                                        key={`${r.symbol}-${idx}`}
                                                        className={`group transition-colors ${selectedIds.includes(r.id || '') ? 'bg-indigo-600/5' : 'hover:bg-white/[0.02]'}`}
                                                    >
                                                        <td className={`px-6 py-4 sticky left-0 z-10 border-r border-white/5 backdrop-blur ${selectedIds.includes(r.id || '') ? "bg-indigo-600/5" : "bg-zinc-950/90 group-hover:bg-white/[0.02]"}`}>
                                                            <button
                                                                onClick={() => r.id && handleToggleSelect(r.id)}
                                                                className="text-zinc-600 group-hover:text-white transition-colors"
                                                            >
                                                                {selectedIds.includes(r.id || '') ? (
                                                                    <CheckSquare className="w-4 h-4 text-indigo-500" />
                                                                ) : (
                                                                    <Square className="w-4 h-4" />
                                                                )}
                                                            </button>
                                                        </td>
                                                        <td className="px-6 py-4">
                                                            <div className="flex items-center gap-3">
                                                                <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="sm" />
                                                                <span className="font-mono font-black text-indigo-400">{r.symbol}</span>
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4">
                                                            <span className="text-[10px] font-bold text-zinc-400 uppercase truncate max-w-[150px] block">{r.name || r.symbol}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono font-bold text-zinc-500">{r.last_close.toFixed(2)}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className={`font-mono font-black ${(r.profit_loss_pct || 0) > 0 ? 'text-emerald-500' : (r.profit_loss_pct || 0) < 0 ? 'text-red-500' : 'text-zinc-100'}`}>
                                                                {r.exit_price ? r.exit_price.toFixed(2) : r.last_close.toFixed(2)}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono font-bold text-emerald-400/80">{r.target_price ? r.target_price.toFixed(2) : "-"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono text-[9px] font-black text-emerald-500/40">{targetPct}%</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono font-bold text-red-400/80">{r.stop_loss ? r.stop_loss.toFixed(2) : "-"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <span className="font-mono text-[9px] font-black text-red-500/40">{lossPct}%</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            {(() => {
                                                                const { score, label, color, bg } = getAiScore(Number(r.confidence || r.precision));
                                                                return (
                                                                    <div className="flex flex-col items-center gap-0.5">
                                                                        <div className={`px-2 py-0.5 rounded ${bg} ${color} text-[9px] font-black`}>
                                                                            {score}/10
                                                                        </div>
                                                                        <span className="text-[7px] font-black text-zinc-600 uppercase tracking-tighter">{label}</span>
                                                                    </div>
                                                                );
                                                            })()}
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            {(() => {
                                                                const score = r.technical_score || 0;
                                                                const color = score >= 7 ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" :
                                                                    score >= 4 ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
                                                                        "text-red-400 bg-red-500/10 border-red-500/20";
                                                                return (
                                                                    <div className={`px-2 py-0.5 rounded border ${color} text-[9px] font-black inline-block`}>
                                                                        {score}/10
                                                                    </div>
                                                                );
                                                            })()}
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="font-black text-white italic">{`${(r.council_score ?? 0).toFixed(1)}%`}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">{r.consensus_ratio || "0/0"}</span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="text-[9px] font-bold text-zinc-500 uppercase">
                                                                {r.created_at ? new Date(r.created_at).toLocaleDateString() : '-'}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <span className="text-[9px] font-bold text-zinc-600 uppercase">
                                                                {r.updated_at ? new Date(r.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-'}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 text-center">
                                                            <div className="flex items-center justify-center gap-1">
                                                                {(!r.status || r.status === 'pending' || r.status === 'open') ? (
                                                                    <span className="px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-500 text-[8px] font-black uppercase tracking-widest">OPEN</span>
                                                                ) : (
                                                                    <>
                                                                        {((r.status === 'win') || ((r.profit_loss_pct || 0) > 0)) && <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-500 text-[8px] font-black uppercase">WIN</span>}
                                                                        {((r.status === 'loss' || r.status === 'hit_stop') && (r.profit_loss_pct || 0) <= 0) && <span className="px-2 py-0.5 rounded-md bg-red-500/10 text-red-500 text-[8px] font-black uppercase">LOSS</span>}
                                                                    </>
                                                                )}
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 text-right">
                                                            <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                <button
                                                                    onClick={() => openDetails(r)}
                                                                    className="p-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-500 hover:bg-indigo-500 hover:text-white transition-all"
                                                                    title="View Chart"
                                                                >
                                                                    <LineChart className="w-3 h-3" />
                                                                </button>

                                                                <button
                                                                    onClick={() => handleUnpublish(r.id!)}
                                                                    className="p-1.5 rounded-lg bg-zinc-500/10 border border-white/5 text-zinc-500 hover:bg-red-500 hover:text-white transition-all"
                                                                    title="Unpublish"
                                                                >
                                                                    <Trash2 className="w-3 h-3" />
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
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
                                                    <th className="px-8 py-5 text-center">AI Score</th>
                                                    <th className="px-8 py-5 text-right">Entry Price</th>
                                                    <th className="px-8 py-5 text-right">Target/Current</th>
                                                    <th className="px-8 py-5 text-right">Returns</th>
                                                    <th className="px-8 py-5 text-center">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5">
                                                {selectedHistoryResults.map((r, idx) => (
                                                    <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
                                                        <td className="px-8 py-5">
                                                            <div className="flex items-center gap-3">
                                                                <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="sm" />
                                                                <span className="font-mono font-black text-indigo-400">{r.symbol}</span>
                                                            </div>
                                                        </td>
                                                        <td className="px-8 py-5 text-center">
                                                            {(() => {
                                                                const { score, color, bg } = getAiScore(Number(r.confidence));
                                                                return (
                                                                    <div className={`inline-flex px-2 py-0.5 rounded ${bg} ${color} text-[9px] font-black`}>
                                                                        {score}/10
                                                                    </div>
                                                                );
                                                            })()}
                                                        </td>
                                                        <td className="px-8 py-5 text-right font-mono font-bold text-zinc-400">{r.last_close.toFixed(2)}</td>
                                                        <td className="px-8 py-5 text-right font-mono font-bold text-zinc-100">{r.exit_price?.toFixed(2) || "—"}</td>
                                                        <td className="px-8 py-5 text-right">
                                                            <span className={`font-mono font-black ${Number(r.profit_loss_pct) > 0 ? "text-emerald-500" : Number(r.profit_loss_pct) < 0 ? "text-red-500" : "text-zinc-500"}`}>
                                                                {r.profit_loss_pct !== null ? `${Number(r.profit_loss_pct) > 0 ? "+" : ""}${Number(r.profit_loss_pct).toFixed(2)}%` : "0.00%"}
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

            {/* Full Screen Chart Dialog */}
            {showMobileDetail && selected && (
                <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl flex flex-col animate-in fade-in duration-300">
                    <div className="h-20 border-b border-white/10 flex items-center justify-between px-6 bg-zinc-950/50">
                        <div className="flex flex-col">
                            <div className="flex items-center gap-3">
                                <StockLogo symbol={selected.symbol} logoUrl={selected.logo_url} size="sm" />
                                <h3 className="text-xl font-black text-white italic tracking-tighter flex items-center gap-2">
                                    {selected.symbol}
                                    <span className={`px-2 py-0.5 rounded bg-white/5 text-[10px] text-zinc-500 font-bold uppercase`}>
                                        {selected.exchange}
                                    </span>
                                </h3>
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

                    <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                        <div className="max-w-[1400px] mx-auto space-y-8">
                            <div className="min-h-[60vh] rounded-3xl border border-white/5 bg-black/40 overflow-y-auto relative shadow-2xl custom-scrollbar">
                                {detailLoading ? (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                                        <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                        <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest animate-pulse italic">Synchronizing Neural Datashards...</div>
                                    </div>
                                ) : detailData?.testPredictions ? (
                                    <CandleChart
                                        rows={detailData.testPredictions}
                                        savedDate={selected.created_at ? selected.created_at.split("T")[0] : undefined}
                                        targetPrice={selected.target_price}
                                        stopPrice={selected.stop_loss}
                                        chartType={chartType}
                                        showEma50={showEma50}
                                        showEma200={showEma200}
                                        showBB={showBB}
                                        showRsi={showRsi}
                                        showVolume={showVolume}
                                        showMacd={showMacd}
                                        height={600}
                                    />
                                ) : (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                                        <AlertTriangle className="h-8 w-8 text-amber-500" />
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{detailError || "No data available"}</span>
                                    </div>
                                )}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="p-6 rounded-3xl border border-white/5 bg-white/[0.02] space-y-4">
                                    <h4 className="text-xs font-black text-white uppercase tracking-widest flex items-center gap-2">
                                        <Brain className="h-4 w-4 text-indigo-500" />
                                        Core Analysis
                                    </h4>
                                    <div className="space-y-3">
                                        {selected.top_reasons?.map((reason, idx) => (
                                            <div key={idx} className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/5">
                                                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5" />
                                                <p className="text-[11px] font-medium text-zinc-400 leading-relaxed">{reason}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="p-6 rounded-3xl border border-white/5 bg-white/[0.02] space-y-4">
                                    <h4 className="text-xs font-black text-white uppercase tracking-widest flex items-center gap-2">
                                        <Activity className="h-4 w-4 text-emerald-500" />
                                        Signal Parameters
                                    </h4>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-4 rounded-2xl bg-zinc-950/50 border border-white/5">
                                            <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Target</div>
                                            <div className="font-mono text-xl font-black text-emerald-400">{selected.target_price?.toFixed(2) || "—"}</div>
                                        </div>
                                        <div className="p-4 rounded-2xl bg-zinc-950/50 border border-white/5">
                                            <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Stop Loss</div>
                                            <div className="font-mono text-xl font-black text-red-400">{selected.stop_loss?.toFixed(2) || "—"}</div>
                                        </div>
                                        <div className="p-4 rounded-2xl bg-zinc-950/50 border border-white/5">
                                            <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Confidence</div>
                                            <div className="font-mono text-xl font-black text-indigo-400">{Number(selected.confidence || selected.precision || 0).toFixed(2)}</div>
                                        </div>
                                        <div className="p-4 rounded-2xl bg-zinc-950/50 border border-white/5">
                                            <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Risk/Reward</div>
                                            <div className="font-mono text-xl font-black text-white">
                                                {selected.target_price && selected.stop_loss && selected.last_close
                                                    ? ((selected.target_price - selected.last_close) / (selected.last_close - selected.stop_loss)).toFixed(2)
                                                    : "—"}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
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

            <ModelVisibilityDialog
                open={showModelVisibility}
                onClose={() => setShowModelVisibility(false)}
                models={localModels}
                loading={modelsLoading}
                adminConfig={adminConfig}
                toggleModelVisibility={toggleModelVisibility}
                updateModelAlias={updateModelAlias}
            />
        </div>
    );
}

function ModelVisibilityDialog({
    open,
    onClose,
    models,
    loading,
    adminConfig,
    toggleModelVisibility,
    updateModelAlias
}: {
    open: boolean;
    onClose: () => void;
    models: (string | LocalModelMeta)[];
    loading: boolean;
    adminConfig: AdminConfig | null;
    toggleModelVisibility: (name: string) => void;
    updateModelAlias: (name: string, alias: string) => void;
}) {
    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-xl" onClick={onClose} />
            <div className="relative w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-[2.5rem] border border-white/10 bg-zinc-950/90 shadow-2xl flex flex-col animate-in zoom-in-95 duration-300">
                <div className="px-8 py-6 border-b border-white/5 flex items-center justify-between bg-zinc-950">
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                            <Cpu className="w-4 h-4 text-indigo-500" />
                            <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Neural Core Settings</span>
                        </div>
                        <h2 className="text-xl font-black text-white uppercase italic tracking-tighter">Model Visibility Control</h2>
                    </div>
                    <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-zinc-500 hover:text-white transition-all">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-4">
                        {loading ? (
                            <div className="col-span-full flex flex-col items-center justify-center py-20 gap-4">
                                <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest animate-pulse">Syncing Neural Cores...</span>
                            </div>
                        ) : models.map((m) => {
                            const mName = typeof m === "string" ? m : m.name;
                            return (
                                <ModelVisibilityItem
                                    key={mName}
                                    m={m}
                                    isEnabled={adminConfig?.enabledModels?.includes(mName) || false}
                                    alias={adminConfig?.modelAliases?.[mName] || ""}
                                    onToggle={() => toggleModelVisibility(mName)}
                                    onAliasChange={(val) => updateModelAlias(mName, val)}
                                />
                            );
                        })}
                    </div>
                </div>

                <div className="px-8 py-4 bg-zinc-900/50 border-t border-white/5 flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-8 py-2 rounded-xl bg-white text-black text-[10px] font-black uppercase tracking-widest hover:bg-zinc-200 transition-all"
                    >
                        Save & Close
                    </button>
                </div>
            </div>
        </div>
    );
}
