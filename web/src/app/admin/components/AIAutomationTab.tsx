"use client";

import { Zap, ChevronDown, Check, Loader2, Download, Database, Info, History, Trash2 } from "lucide-react";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import ConfirmDialog from "@/components/ConfirmDialog";

interface AIAutomationTabProps {
    dbInventory: any[];
    trainingExchange: string;
    setTrainingExchange: (ex: string) => void;
    isExchangeDropdownOpen: boolean;
    setIsExchangeDropdownOpen: (open: boolean) => void;
    isTraining: boolean;
    fetchTrainedModels: () => void;
    loadingModels: boolean;
    trainedModels: any[];
    handleDownloadModel: (filename: string) => void;
    setIsTraining: (training: boolean) => void;
}

interface LocalModel {
    name: string;
    size_bytes: number;
    size_mb: number;
    created_at: string;
    modified_at: string;
    type: string;
    num_features?: number;
    num_parameters?: number;
    trainingSamples?: number;
}

export default function AIAutomationTab({
    dbInventory,
    trainingExchange,
    setTrainingExchange,
    isExchangeDropdownOpen,
    setIsExchangeDropdownOpen,
    isTraining,
    fetchTrainedModels,
    loadingModels,
    trainedModels,
    handleDownloadModel,
    setIsTraining
}: AIAutomationTabProps) {
    const [workflow, setWorkflow] = useState<"ai-training" | "data-sync">("ai-training");
    const [trainingTab, setTrainingTab] = useState<"classic" | "genetic">("classic");
    const [when, setWhen] = useState<string>(""); // datetime-local value
    const [days, setDays] = useState<number>(365);
    const [updatePrices, setUpdatePrices] = useState<boolean>(true);
    const [updateFunds, setUpdateFunds] = useState<boolean>(false);
    const [unified, setUnified] = useState<boolean>(false);
    const [scheduling, setScheduling] = useState<boolean>(false);
    const [triggeringTraining, setTriggeringTraining] = useState<boolean>(false);
    const [useEarlyStopping, setUseEarlyStopping] = useState<boolean>(true);
    const [nEstimators, setNEstimators] = useState<number>(1000);
    const [trainingStrategy, setTrainingStrategy] = useState<"golden" | "grid_small" | "random">("golden");
    const [randomSearchIter, setRandomSearchIter] = useState<number>(5);
    const [lastTrainingSummary, setLastTrainingSummary] = useState<{
        exchange: string;
        useEarlyStopping: boolean;
        nEstimators: number;
        trainingSamples: number;
        timestamp: string;
        numFeatures?: number;
        symbolsUsed?: number;
        rawRows?: number;
    } | null>(null);
    const [localModels, setLocalModels] = useState<LocalModel[]>([]);
    const [loadingLocalModels, setLoadingLocalModels] = useState<boolean>(false);
    const [deletingModels, setDeletingModels] = useState<Set<string>>(new Set());

    const [trainingStatus, setTrainingStatus] = useState<{
        running: boolean;
        exchange: string | null;
        started_at: string | null;
        completed_at: string | null;
        error: string | null;
        last_message: string | null;
        last_update?: string | null;
        phase?: string | null;
        stats?: Record<string, any> | null;
    } | null>(null);
    const liveStats = trainingStatus?.stats ?? null;
    const symbolsProcessed = typeof liveStats?.symbols_processed === "number" ? liveStats.symbols_processed : null;
    const symbolsTotal = typeof liveStats?.symbols_total === "number" ? liveStats.symbols_total : null;
    const symbolProgressPct = symbolsProcessed !== null && symbolsTotal ? Math.min(100, Math.round((symbolsProcessed / symbolsTotal) * 100)) : null;
    const rowsLoaded = typeof liveStats?.rows_loaded === "number" ? liveStats.rows_loaded : null;
    const rowsTotal = typeof liveStats?.rows_total === "number" ? liveStats.rows_total : null;
    const rowsProgressPct = rowsLoaded !== null && rowsTotal ? Math.min(100, Math.round((rowsLoaded / rowsTotal) * 100)) : null;
    const phaseProgressMap: Record<string, number> = {
        data_loaded: 10,
        processing_symbols: 40,
        data_prepared: 60,
        features_ready: 70,
        training: 85,
        saved: 95,
        uploaded: 100,
    };
    const phaseTimeline = ["loading_rows", "data_loaded", "processing_symbols", "data_prepared", "features_ready", "training", "saved", "uploaded"];
    const phaseLabels: Record<string, string> = {
        loading_rows: "Loading",
        data_loaded: "Loaded",
        processing_symbols: "Symbols",
        data_prepared: "Prepared",
        features_ready: "Features",
        training: "Training",
        saved: "Saved",
        uploaded: "Uploaded",
    };
    const overallProgressPct = (() => {
        const phase = trainingStatus?.phase ?? null;
        if (!phase) return null;
        if (phase === "processing_symbols" && symbolProgressPct !== null) {
            return Math.min(60, 10 + Math.round(symbolProgressPct * 0.5));
        }
        return phaseProgressMap[phase] ?? null;
    })();
    const etaSeconds = (() => {
        if (!trainingStatus?.last_update) return null;
        const elapsed = (Date.now() - new Date(trainingStatus.last_update).getTime()) / 1000;
        if (!Number.isFinite(elapsed)) return null;
        if (symbolProgressPct !== null && symbolProgressPct > 0 && symbolProgressPct < 100) {
            return Math.max(0, Math.round((elapsed / symbolProgressPct) * (100 - symbolProgressPct)));
        }
        if (rowsProgressPct !== null && rowsProgressPct > 0 && rowsProgressPct < 100) {
            return Math.max(0, Math.round((elapsed / rowsProgressPct) * (100 - rowsProgressPct)));
        }
        return null;
    })();
    const formatEta = (secs: number | null) => {
        if (!secs && secs !== 0) return null;
        const minutes = Math.floor(secs / 60);
        const seconds = Math.floor(secs % 60);
        return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
    };
    const [trainingLogs, setTrainingLogs] = useState<Array<{ ts: string; msg: string }>>([]);
    const [lastLoggedMessage, setLastLoggedMessage] = useState<string | null>(null);
    const [lastStatusCompletedAt, setLastStatusCompletedAt] = useState<string | null>(null);
    const [modelName, setModelName] = useState<string>("");
    const [featurePreset, setFeaturePreset] = useState<"core" | "extended" | "max">("extended");
    const [maxFeatures, setMaxFeatures] = useState<number | null>(null);
    const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
    const [pendingDeleteModel, setPendingDeleteModel] = useState<string | null>(null);

    // Fetch local models once on mount (further updates happen on training completion or manual refresh)
    useEffect(() => {
        fetchLocalModels();
    }, []);

    // Keep a small client-side log of training status messages for UI visibility.
    useEffect(() => {
        if (!trainingStatus?.last_message) return;
        const msg = String(trainingStatus.last_message);
        if (!msg.trim()) return;
        if (msg === lastLoggedMessage) return;
        setLastLoggedMessage(msg);
        setTrainingLogs((prev) => {
            const next = [...prev, { ts: new Date().toISOString(), msg }];
            return next.length > 250 ? next.slice(next.length - 250) : next;
        });
    }, [trainingStatus?.last_message, lastLoggedMessage]);

    // Live status via SSE with fallback polling.
    useEffect(() => {
        let pollId: ReturnType<typeof setInterval> | null = null;
        const startPolling = () => {
            if (pollId) return;
            pollId = setInterval(() => {
                refreshTrainingStatus();
            }, 2000);
        };

        if (typeof window === "undefined") return () => undefined;

        const es = new EventSource("/api/admin/train/stream");
        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setTrainingStatus(data);
                if (typeof data?.running === "boolean") {
                    setIsTraining(data.running);
                }
            } catch {
                startPolling();
            }
        };
        es.onerror = () => {
            es.close();
            startPolling();
        };

        if (trainingStatus?.running || isTraining) {
            startPolling();
        }

        return () => {
            es.close();
            if (pollId) clearInterval(pollId);
        };
    }, [trainingStatus?.running, isTraining]);

    const fetchLocalModels = async () => {
        setLoadingLocalModels(true);
        try {
            const res = await fetch("/api/admin/train/models");
            const data = await res.json();
            // Backend already returns objects shaped like LocalModel from list_local_models
            setLocalModels((data?.models as LocalModel[]) || []);
        } catch (error) {
            console.error("Error fetching local models:", error);
            setLocalModels([]);
        } finally {
            setLoadingLocalModels(false);
        }
    };

    const refreshTrainingStatus = async () => {
        try {
            const res = await fetch("/api/admin/train/status");
            if (!res.ok) return;
            const data = await res.json();

            setTrainingStatus(data);

            if (typeof data.running === "boolean") {
                setIsTraining(data.running);
            }

            if (!data.running && data.completed_at && data.completed_at !== lastStatusCompletedAt) {
                setLastStatusCompletedAt(data.completed_at);
                try {
                    const sRes = await fetch("/api/admin/train/summary");
                    if (sRes.ok) {
                        const sData = await sRes.json();
                        if (sData.status === "ok" && sData.summary) {
                            setLastTrainingSummary(sData.summary);
                        }
                    }
                } catch {
                    // ignore UI summary errors
                }

                try {
                    fetchLocalModels();
                } catch {
                    // ignore refresh errors
                }
            }
        } catch {
            // ignore connection errors for status fetch
        }
    };

    const deleteLocalModel = async (modelName: string) => {
        setPendingDeleteModel(modelName);
        setConfirmDeleteOpen(true);
    };

    const confirmDeleteLocalModel = async () => {
        if (!pendingDeleteModel) return;
        const modelName = pendingDeleteModel;
        setDeletingModels(prev => new Set(prev).add(modelName));
        try {
            const res = await fetch(`/api/admin/train/models/${encodeURIComponent(modelName)}`, {
                method: "DELETE"
            });
            
            if (res.ok) {
                toast.success("Model removed", {
                    description: modelName
                });
                await fetchLocalModels();
            } else {
                const contentType = res.headers.get("content-type") || "";
                let errorMessage = "Failed to remove model";
                if (contentType.includes("application/json")) {
                    const error = await res.json();
                    errorMessage = error.detail || error.message || errorMessage;
                } else {
                    const text = await res.text();
                    if (text) {
                        errorMessage = text;
                    }
                }
                toast.error(errorMessage);
            }
        } catch (error) {
            toast.error("Connection error");
        } finally {
            setDeletingModels(prev => {
                const newSet = new Set(prev);
                newSet.delete(modelName);
                return newSet;
            });
            setConfirmDeleteOpen(false);
            setPendingDeleteModel(null);
        }
    };

    return (
        <div className="p-8 max-w-7xl mx-auto w-full space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="flex flex-col gap-2">
                <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
                    <Zap className="h-8 w-8 text-indigo-500" />
                    AI & Automation
                </h1>
                <p className="text-sm text-zinc-500 font-medium">
                    Manage local AI training and browse trained model artifacts.
                </p>
            </header>

            <div className="flex items-center gap-2">
                <button
                    type="button"
                    onClick={() => setTrainingTab("classic")}
                    className={`h-10 px-4 rounded-2xl border text-[11px] font-black uppercase tracking-widest transition-all ${
                        trainingTab === "classic"
                            ? "bg-zinc-100 text-black border-zinc-100"
                            : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-600"
                    }`}
                >
                    Classic
                </button>
                <button
                    type="button"
                    onClick={() => setTrainingTab("genetic")}
                    className={`h-10 px-4 rounded-2xl border text-[11px] font-black uppercase tracking-widest transition-all ${
                        trainingTab === "genetic"
                            ? "bg-zinc-100 text-black border-zinc-100"
                            : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-600"
                    }`}
                >
                    Genetic Algorithms
                </button>
            </div>

            {trainingTab === "classic" ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                    {/* Training Control Cluster */}
                    <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-6">
                        <div className="flex items-center gap-4">
                            <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                <Zap className="w-6 h-6" />
                            </div>
                            <div>
                                <h2 className="text-xl font-black text-white">Local Training</h2>
                                <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Manual Server Processing</p>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="grid grid-cols-1 gap-4">
                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Model Name</label>
                                    <input
                                        type="text"
                                        value={modelName}
                                        onChange={(e) => setModelName(e.target.value)}
                                        placeholder="Optional (e.g. model_SA_v2)"
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                    />
                                </div>
                            </div>
                            <div className="space-y-1 mt-3">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">
                                    Feature Preset
                                </label>
                                <div className="flex gap-2">
                                    <button
                                        type="button"
                                        onClick={() => setFeaturePreset("core")}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${
                                            featurePreset === "core"
                                                ? "bg-zinc-100 text-black border-zinc-100"
                                                : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                                        }`}
                                    >
                                        Core
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setFeaturePreset("extended")}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${
                                            featurePreset === "extended"
                                                ? "bg-zinc-100 text-black border-zinc-100"
                                                : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                                        }`}
                                    >
                                        Extended
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setFeaturePreset("max")}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${
                                            featurePreset === "max"
                                                ? "bg-zinc-100 text-black border-zinc-100"
                                                : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                                        }`}
                                    >
                                        Max
                                    </button>
                                </div>
                            </div>

                            <div className="space-y-1 mt-2">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">
                                    Max Features (optional)
                                </label>
                                <input
                                    type="number"
                                    min={1}
                                    max={200}
                                    step={1}
                                    value={maxFeatures ?? ""}
                                    onChange={(e) => {
                                        const val = e.target.value;
                                        if (!val) {
                                            setMaxFeatures(null);
                                            return;
                                        }
                                        const num = Number(val);
                                        setMaxFeatures(Number.isFinite(num) && num > 0 ? num : null);
                                    }}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                    placeholder="Leave empty to use all preset features"
                                />
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                                <div className="flex items-center gap-3">
                                    <input
                                        id="useEarlyStopping"
                                        type="checkbox"
                                        checked={useEarlyStopping}
                                        onChange={(e) => setUseEarlyStopping(e.target.checked)}
                                        disabled={trainingStrategy !== "golden"}
                                        className="h-4 w-4 rounded border-zinc-700 bg-black disabled:opacity-50"
                                    />
                                    <label htmlFor="useEarlyStopping" className="text-xs text-zinc-400">
                                        Enable Early Stopping
                                    </label>
                                </div>

                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">n_estimators</label>
                                    <input
                                        type="number"
                                        min={50}
                                        max={5000}
                                        step={50}
                                        value={nEstimators}
                                        onChange={(e) => setNEstimators(Number(e.target.value) || 100)}
                                        disabled={trainingStrategy !== "golden" || useEarlyStopping}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 disabled:opacity-50"
                                    />
                                </div>

                                <div className="space-y-1 sm:col-span-2">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Training Strategy</label>
                                    <select
                                        value={trainingStrategy}
                                        onChange={(e) => setTrainingStrategy(e.target.value as any)}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                    >
                                        <option value="golden">Golden Mix (default)</option>
                                        <option value="grid_small">Full Grid Search (small data)</option>
                                        <option value="random">Random Search (fast tuning)</option>
                                    </select>
                                </div>

                                {trainingStrategy === "random" && (
                                    <div className="space-y-1">
                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Random Search Iterations</label>
                                        <input
                                            type="number"
                                            min={1}
                                            max={50}
                                            step={1}
                                            value={randomSearchIter}
                                            onChange={(e) => setRandomSearchIter(Number(e.target.value) || 5)}
                                            className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                        />
                                    </div>
                                )}
                            </div>

                            <div className="space-y-2 mt-4">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Target Exchange</label>
                                <div className="relative">
                                    <button
                                        onClick={() => setIsExchangeDropdownOpen(!isExchangeDropdownOpen)}
                                        className={`w-full bg-black border ${isExchangeDropdownOpen ? 'border-indigo-500 ring-1 ring-indigo-500/50' : 'border-zinc-800'} rounded-2xl p-4 text-sm text-left transition-all flex items-center justify-between group hover:border-zinc-700`}
                                    >
                                        <span className={`${trainingExchange ? 'text-white font-medium' : 'text-zinc-500'}`}>
                                            {trainingExchange ? (
                                                <span className="flex items-center gap-2">
                                                    <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"></span>
                                                    {trainingExchange}
                                                    <span className="text-zinc-600 text-xs ml-1">
                                                        ({dbInventory.find(i => i.exchange === trainingExchange)?.priceCount || 0} symbols)
                                                    </span>
                                                </span>
                                            ) : (
                                                "Select an exchange to train on..."
                                            )}
                                        </span>
                                        <ChevronDown className={`w-4 h-4 text-zinc-500 transition-transform duration-300 ${isExchangeDropdownOpen ? 'rotate-180 text-indigo-500' : 'group-hover:text-zinc-300'}`} />
                                    </button>

                                    {/* Dropdown Menu */}
                                    <div className={`absolute z-20 top-full left-0 right-0 mt-2 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden transition-all duration-200 origin-top transform ${isExchangeDropdownOpen ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 -translate-y-2 pointer-events-none'}`}>
                                        <div className="p-2 space-y-1 max-h-[300px] overflow-y-auto custom-scrollbar">
                                            {dbInventory.filter(i => i.priceCount > 0).length === 0 && (
                                                <div className="p-4 text-center text-xs text-zinc-600 italic">No exchanges with data available.</div>
                                            )}
                                            {dbInventory.filter(i => i.priceCount > 0).map(i => (
                                                <button
                                                    key={i.exchange}
                                                    onClick={() => {
                                                        setTrainingExchange(i.exchange);
                                                        setIsExchangeDropdownOpen(false);
                                                    }}
                                                    className={`w-full p-3 rounded-xl text-left text-sm font-medium transition-all flex items-center justify-between group ${trainingExchange === i.exchange ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' : 'hover:bg-zinc-800 text-zinc-400 hover:text-white'}`}
                                                >
                                                    <span className="flex items-center gap-3">
                                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-black uppercase tracking-wider ${trainingExchange === i.exchange ? 'bg-white/20 text-white' : 'bg-black border border-zinc-800 text-zinc-500 group-hover:border-zinc-700'}`}>
                                                            {i.exchange.substring(0, 2)}
                                                        </div>
                                                        <div>
                                                            <div className="">{i.exchange}</div>
                                                            <div className={`text-[10px] ${trainingExchange === i.exchange ? 'text-indigo-200' : 'text-zinc-600 group-hover:text-zinc-500'}`}>{i.priceCount} Active Symbols</div>
                                                        </div>
                                                    </span>
                                                    {trainingExchange === i.exchange && <Check className="w-4 h-4" />}
                                                </button>
                                            ))}
                                        </div>
                                        <div className="px-4 py-3 bg-zinc-950 border-t border-zinc-900 text-[10px] text-zinc-600 text-center font-medium">
                                            Select an exchange to initialize the AI model.
                                        </div>
                                    </div>
                                </div>

                                {/* Backdrop to close */}
                                {isExchangeDropdownOpen && (
                                    <div className="fixed inset-0 z-10 bg-transparent" onClick={() => setIsExchangeDropdownOpen(false)} />
                                )}
                            </div>

                            {lastTrainingSummary && (
                                <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-400 flex flex-col gap-4">
                                    <div className="flex items-center justify-between">
                                        <span className="font-bold text-zinc-200">Last Training Summary</span>
                                        <span className="text-[10px] text-zinc-500">
                                            {new Date(lastTrainingSummary.timestamp).toLocaleString()}
                                        </span>
                                    </div>
                                    <div className="flex flex-wrap gap-3">
                                        <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-700 text-[10px]">
                                            Exchange: <span className="text-zinc-200">{lastTrainingSummary.exchange}</span>
                                        </span>
                                        <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-700 text-[10px]">
                                            Mode: <span className="text-zinc-200">{lastTrainingSummary.useEarlyStopping ? "Early Stopping" : "Fixed n_estimators"}</span>
                                        </span>
                                    </div>

                                    {formatEta(etaSeconds) && (
                                        <div className="flex items-center justify-between text-[10px] text-zinc-500">
                                            <span>ETA</span>
                                            <span className="text-zinc-300 font-bold">{formatEta(etaSeconds)}</span>
                                        </div>
                                    )}
                                    <div className="flex flex-wrap gap-2">
                                        {phaseTimeline.map((phaseKey) => {
                                            const active = trainingStatus?.phase === phaseKey;
                                            const completed = phaseProgressMap[phaseKey] !== undefined && (phaseProgressMap[phaseKey] ?? 0) <= (overallProgressPct ?? 0);
                                            return (
                                                <span
                                                    key={phaseKey}
                                                    className={`px-2 py-1 rounded-full border text-[10px] font-bold uppercase tracking-widest ${active
                                                        ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/40"
                                                        : completed
                                                            ? "bg-emerald-600/10 text-emerald-300 border-emerald-500/30"
                                                            : "bg-zinc-900 text-zinc-500 border-zinc-800"}`}
                                                >
                                                    {phaseLabels[phaseKey] ?? phaseKey}
                                                </span>
                                            );
                                        })}
                                    </div>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                            <div className="text-[10px] text-zinc-500 uppercase font-black">Estimators</div>
                                            <div className="text-sm font-black text-zinc-100 mt-1">{lastTrainingSummary.nEstimators}</div>
                                        </div>
                                        <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                            <div className="text-[10px] text-zinc-500 uppercase font-black">Samples</div>
                                            <div className="text-sm font-black text-zinc-100 mt-1">{lastTrainingSummary.trainingSamples}</div>
                                        </div>
                                        {typeof lastTrainingSummary.numFeatures === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Features</div>
                                                <div className={`text-sm font-black mt-1 ${lastTrainingSummary.numFeatures > 20 ? "text-emerald-300" : "text-red-300"}`}>
                                                    {lastTrainingSummary.numFeatures}
                                                </div>
                                            </div>
                                        )}
                                        {typeof lastTrainingSummary.symbolsUsed === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Symbols Used</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{lastTrainingSummary.symbolsUsed}</div>
                                            </div>
                                        )}
                                        {typeof lastTrainingSummary.rawRows === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3 col-span-2">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Raw Rows</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{lastTrainingSummary.rawRows.toLocaleString()}</div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={async () => {
                                    if (!trainingExchange) return;
                                    setIsTraining(true);
                                    try {
                                        const res = await fetch("/api/admin/train/local", {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({
                                                exchange: trainingExchange,
                                                useEarlyStopping,
                                                nEstimators,
                                                modelName: modelName.trim() || undefined,
                                                featurePreset,
                                                trainingStrategy,
                                                randomSearchIter: trainingStrategy === "random" ? randomSearchIter : undefined,
                                                maxFeatures: maxFeatures ?? undefined,
                                            }),
                                        });
                                        const data = await res.json();
                                        if (res.ok) {
                                            toast.success("Local training started", {
                                                description: data.message
                                            });
                                            await refreshTrainingStatus();
                                        } else {
                                            toast.error(data.detail || "Failed to start local training");
                                            setIsTraining(false);
                                        }
                                    } catch (e) {
                                        toast.error("Connection error");
                                        setIsTraining(false);
                                    }
                                }}
                                disabled={!trainingExchange || isTraining}
                                className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl text-xs font-black flex items-center justify-center gap-3 disabled:opacity-50 transition-all shadow-[0_0_20px_rgba(99,102,241,0.2)]"
                            >
                                {isTraining ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
                                RUN SERVER TRAINING
                            </button>

                            {trainingStatus && (
                                <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-400 flex flex-col gap-1">
                                    <div className="flex items-center justify-between">
                                        <span className="font-bold text-zinc-200">
                                            {trainingStatus.running
                                                ? "Training is running"
                                                : trainingStatus.error
                                                    ? "Training failed"
                                                    : trainingStatus.completed_at
                                                        ? "Last training finished"
                                                        : "No recent training"}
                                        </span>
                                        {trainingStatus.running && (
                                            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                                        )}
                                    </div>
                                    <div className="flex flex-wrap gap-3 mt-1">
                                        {trainingStatus.exchange && (
                                            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-700 text-[10px]">
                                                Exchange: <span className="text-zinc-200">{trainingStatus.exchange}</span>
                                            </span>
                                        )}
                                        {trainingStatus.started_at && (
                                            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-700 text-[10px]">
                                                Started: <span className="text-zinc-200">{new Date(trainingStatus.started_at).toLocaleTimeString()}</span>
                                            </span>
                                        )}
                                        {trainingStatus.completed_at && (
                                            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-700 text-[10px]">
                                                Completed: <span className="text-zinc-200">{new Date(trainingStatus.completed_at).toLocaleTimeString()}</span>
                                            </span>
                                        )}
                                    </div>
                                    {trainingStatus.last_message && (
                                        <p className="mt-2 text-[11px] text-zinc-500">
                                            {trainingStatus.last_message}
                                        </p>
                                    )}
                                    {trainingStatus.error && (
                                        <p className="mt-1 text-[11px] text-red-400">
                                            Error: {trainingStatus.error}
                                        </p>
                                    )}
                                </div>
                            )}

                            {trainingStatus?.running && trainingStatus.stats && (
                                <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-400 flex flex-col gap-4">
                                    <div className="flex items-center justify-between">
                                        <span className="font-bold text-zinc-200">Live Training Stats</span>
                                        {trainingStatus.phase && (
                                            <span className="text-[10px] text-zinc-500 uppercase tracking-widest">
                                                {trainingStatus.phase}
                                            </span>
                                        )}
                                    </div>
                                    {rowsProgressPct !== null && (
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-[10px] text-zinc-500">
                                                <span>Rows Loaded</span>
                                                <span>{rowsLoaded?.toLocaleString()}/{rowsTotal?.toLocaleString()}</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden">
                                                <div
                                                    className="h-full bg-sky-500 transition-all"
                                                    style={{ width: `${rowsProgressPct}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                    {symbolProgressPct !== null && (
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-[10px] text-zinc-500">
                                                <span>Symbols Progress</span>
                                                <span>{symbolsProcessed}/{symbolsTotal}</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden">
                                                <div
                                                    className="h-full bg-indigo-500 transition-all"
                                                    style={{ width: `${symbolProgressPct}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                    {overallProgressPct !== null && (
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-[10px] text-zinc-500">
                                                <span>Overall Progress</span>
                                                <span>{overallProgressPct}%</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden">
                                                <div
                                                    className="h-full bg-emerald-500 transition-all"
                                                    style={{ width: `${overallProgressPct}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                    <div className="grid grid-cols-2 gap-3">
                                        {typeof trainingStatus.stats?.n_estimators === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Estimators</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{trainingStatus.stats.n_estimators}</div>
                                            </div>
                                        )}
                                        {typeof trainingStatus.stats?.samples === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Samples</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{trainingStatus.stats.samples}</div>
                                            </div>
                                        )}
                                        {typeof trainingStatus.stats?.features === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Features</div>
                                                <div className={`text-sm font-black mt-1 ${trainingStatus.stats.features > 20 ? "text-emerald-300" : "text-red-300"}`}>
                                                    {trainingStatus.stats.features}
                                                </div>
                                            </div>
                                        )}
                                        {typeof trainingStatus.stats?.symbols_used === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Symbols Used</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{trainingStatus.stats.symbols_used}</div>
                                            </div>
                                        )}
                                        {typeof trainingStatus.stats?.symbols_total === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Symbols Total</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{trainingStatus.stats.symbols_total}</div>
                                            </div>
                                        )}
                                        {typeof trainingStatus.stats?.raw_rows === "number" && (
                                            <div className="rounded-xl border border-zinc-800 bg-black/40 p-3 col-span-2">
                                                <div className="text-[10px] text-zinc-500 uppercase font-black">Raw Rows</div>
                                                <div className="text-sm font-black text-zinc-100 mt-1">{trainingStatus.stats.raw_rows.toLocaleString()}</div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
                                <div className="flex items-center justify-between">
                                    <div className="text-xs font-black text-white">Local Training Logs</div>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setTrainingLogs([]);
                                            setLastLoggedMessage(null);
                                        }}
                                        className="text-[10px] font-black uppercase tracking-widest text-zinc-500 hover:text-zinc-200 transition-colors"
                                    >
                                        Clear
                                    </button>
                                </div>

                                <div className="mt-3 max-h-48 overflow-y-auto pr-2 custom-scrollbar space-y-2">
                                    {trainingLogs.length === 0 ? (
                                        <div className="text-[11px] text-zinc-600">No logs yet.</div>
                                    ) : (
                                        trainingLogs
                                            .slice()
                                            .reverse()
                                            .map((l, idx) => (
                                                <div key={`${l.ts}-${idx}`} className="text-[11px] text-zinc-500 font-mono break-words">
                                                    <span className="text-zinc-700">[{new Date(l.ts).toLocaleTimeString()}]</span> {l.msg}
                                                </div>
                                            ))
                                    )}
                                </div>
                            </div>

                            <button
                                onClick={async () => {
                                    if (!trainingExchange) {
                                        toast.error("Select an exchange first");
                                        return;
                                    }
                                    setTriggeringTraining(true);
                                    try {
                                        const res = await fetch("/api/admin/train/trigger", {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ exchange: trainingExchange })
                                        });
                                        const data = await res.json();
                                        if (res.ok) {
                                            toast.success("GitHub training triggered", {
                                                description: data.message || `Exchange ${trainingExchange}`
                                            });
                                        } else {
                                            toast.error(data.detail || "Failed to trigger GitHub training");
                                        }
                                    } catch (e) {
                                        toast.error("Connection error");
                                    } finally {
                                        setTriggeringTraining(false);
                                    }
                                }}
                                disabled={!trainingExchange || triggeringTraining}
                                className="w-full mt-3 py-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-100 rounded-2xl text-xs font-black flex items-center justify-center gap-3 disabled:opacity-50 border border-zinc-700"
                            >
                                {triggeringTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4 rotate-45" />}
                                TRIGGER GITHUB TRAINING
                            </button>

                            <div className="flex items-start gap-3 p-6 rounded-2xl bg-zinc-950 border border-zinc-800/50">
                                <Info className="w-5 h-5 text-zinc-500 shrink-0 mt-0.5" />
                                <div className="space-y-1">
                                    <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-widest">Server Processing</p>
                                    <p className="text-xs text-zinc-600 font-medium leading-relaxed">
                                        This will run the training script directly on the API server. For scheduled automation, use <span className="text-zinc-400">GitHub Actions cron</span> in the repo workflows.
                                    </p>
                                </div>
                            </div>

                            <div className="rounded-2xl border border-zinc-800/60 bg-zinc-950 p-6 space-y-4">
                                <div className="flex items-center gap-3">
                                    <div className="p-2.5 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-400">
                                        <Database className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-black text-white">GitHub Actions Automation</p>
                                        <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Cron Updates</p>
                                    </div>
                                </div>
                                <div className="space-y-2 text-xs text-zinc-500 font-medium">
                                    <div className="flex items-start gap-2">
                                        <span className="mt-1 h-1 w-1 rounded-full bg-zinc-700" />
                                        <span><span className="text-zinc-300">ai-training.yml</span> runs weekly (default: Sunday 00:00 UTC).</span>
                                    </div>
                                    <div className="flex items-start gap-2">
                                        <span className="mt-1 h-1 w-1 rounded-full bg-zinc-700" />
                                        <span><span className="text-zinc-300">data-sync.yml</span> runs daily (default: 22:30 UTC).</span>
                                    </div>
                                    <div className="flex items-start gap-2">
                                        <span className="mt-1 h-1 w-1 rounded-full bg-zinc-700" />
                                        <span>Edit the <span className="text-zinc-300">schedule</span> cron in workflows to match market hours.</span>
                                    </div>
                                </div>

                                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Workflow</label>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => setWorkflow("ai-training")}
                                                className={`flex-1 py-2 text-xs font-bold rounded-lg border ${workflow === "ai-training" ? "bg-indigo-600 text-white border-indigo-600" : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"}`}
                                            >
                                                AI Training
                                            </button>
                                            <button
                                                onClick={() => setWorkflow("data-sync")}
                                                className={`flex-1 py-2 text-xs font-bold rounded-lg border ${workflow === "data-sync" ? "bg-indigo-600 text-white border-indigo-600" : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"}`}
                                            >
                                                Data Sync
                                            </button>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Run At (Local)</label>
                                        <input
                                            type="datetime-local"
                                            value={when}
                                            onChange={(e) => setWhen(e.target.value)}
                                            className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                        />
                                    </div>
                                </div>

                                {workflow === "data-sync" && (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                                        <div className="space-y-2">
                                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Days</label>
                                            <input
                                                type="number"
                                                min={1}
                                                value={days}
                                                onChange={(e) => setDays(Number(e.target.value) || 1)}
                                                className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                            />
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <input id="updPrices" type="checkbox" checked={updatePrices} onChange={(e) => setUpdatePrices(e.target.checked)} className="h-4 w-4 rounded border-zinc-700 bg-black" />
                                            <label htmlFor="updPrices" className="text-xs text-zinc-400">Update Prices</label>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <input id="updFunds" type="checkbox" checked={updateFunds} onChange={(e) => setUpdateFunds(e.target.checked)} className="h-4 w-4 rounded border-zinc-700 bg-black" />
                                            <label htmlFor="updFunds" className="text-xs text-zinc-400">Update Fundamentals</label>
                                        </div>
                                        <div className="flex items-center gap-3 md:col-span-3">
                                            <input id="unified" type="checkbox" checked={unified} onChange={(e) => setUnified(e.target.checked)} className="h-4 w-4 rounded border-zinc-700 bg-black" />
                                            <label htmlFor="unified" className="text-xs text-zinc-400">Unified Dates</label>
                                        </div>
                                    </div>
                                )}

                                <button
                                    onClick={async () => {
                                        if (!trainingExchange) {
                                            toast.error("Select an exchange first");
                                            return;
                                        }
                                        setScheduling(true);
                                        try {
                                            const whenIso = when ? new Date(when).toISOString() : new Date().toISOString();
                                            const body: any = {
                                                workflow,
                                                when: whenIso,
                                                exchange: trainingExchange,
                                            };
                                            if (workflow === "data-sync") {
                                                body.days = days;
                                                body.updatePrices = updatePrices;
                                                body.updateFunds = updateFunds;
                                                body.unified = unified;
                                            }
                                            const res = await fetch("/api/admin/actions/schedule", {
                                                method: "POST",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify(body),
                                            });
                                            const data = await res.json();
                                            if (res.ok) {
                                                toast.success("Action scheduled", { description: `${workflow} in ${data.delay_seconds}s` });
                                            } else {
                                                toast.error(data.detail || "Failed to schedule action");
                                            }
                                        } catch (e) {
                                            toast.error("Connection error");
                                        } finally {
                                            setScheduling(false);
                                        }
                                    }}
                                    disabled={!trainingExchange || scheduling}
                                    className="w-full mt-3 py-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 rounded-xl text-xs font-black disabled:opacity-50"
                                >
                                    {scheduling ? "Scheduling..." : "Schedule GitHub Action"}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="space-y-6">
                    {/* Trained Models Inventory */}
                    <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 flex flex-col h-full space-y-6">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
                                    <Database className="w-6 h-6" />
                                </div>
                                <div>
                                    <h2 className="text-xl font-black text-white">Model Artifacts</h2>
                                    <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Available .pkl Modules</p>
                                </div>
                            </div>
                            <button
                                onClick={async () => {
                                    // Refresh remote-trained models (if any)
                                    try {
                                        await fetchTrainedModels();
                                    } catch {
                                        // ignore errors in this manual refresh
                                    }

                                    // Single-shot refresh of local training status + summary + local models
                                    await refreshTrainingStatus();
                                }}
                                className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-500 hover:text-indigo-400 transition-all"
                            >
                                <History className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex-1 min-h-[400px] space-y-3 overflow-y-auto pr-2 custom-scrollbar">
                            {/* Local Models Section */}
                            {loadingLocalModels && localModels.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-full gap-4 text-zinc-600 grayscale">
                                    <Loader2 className="w-8 h-8 animate-spin" />
                                    <p className="text-xs font-bold uppercase tracking-widest">Fetching models...</p>
                                </div>
                            ) : (
                                <>
                                    {/* Local Models */}
                                    {localModels.length > 0 && (
                                        <>
                                            <div className="px-1 pt-2">
                                                <p className="text-[10px] text-zinc-500 font-black uppercase tracking-widest mb-2">Local Files</p>
                                            </div>
                                            {localModels.map(model => (
                                                <div
                                                    key={model.name}
                                                    className="p-5 rounded-2xl bg-black border border-zinc-800/50 hover:border-zinc-700 transition-all flex items-center justify-between group"
                                                >
                                                    <div className="flex items-center gap-4 min-w-0 flex-1">
                                                        <div className="p-2.5 rounded-xl bg-zinc-900 text-zinc-500 group-hover:bg-indigo-500/10 group-hover:text-indigo-400 transition-all">
                                                            <Database className="w-5 h-5" />
                                                        </div>
                                                        <div className="min-w-0 flex-1">
                                                            <div className="text-sm font-black text-zinc-100 truncate">{model.name}</div>
                                                            <div className="flex items-center gap-2 mt-1 flex-wrap">
                                                                <span className="text-[10px] text-zinc-600 font-mono">{model.size_mb} MB</span>
                                                                <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                                <span className="text-[10px] text-zinc-600">{new Date(model.modified_at).toLocaleDateString()}</span>
                                                                {model.num_features !== undefined && (
                                                                    <>
                                                                        <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                                        <span className="text-[10px] bg-indigo-600/20 text-indigo-300 px-2 py-0.5 rounded-md font-bold">
                                                                            {model.num_features} Features
                                                                        </span>
                                                                    </>
                                                                )}
                                                                {model.num_parameters !== undefined && model.num_parameters > 0 && (
                                                                    <>
                                                                        <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                                        <span className="text-[10px] bg-amber-600/20 text-amber-300 px-2 py-0.5 rounded-md font-bold">
                                                                            {model.num_parameters} Params
                                                                        </span>
                                                                    </>
                                                                )}
                                                                {typeof model.trainingSamples === "number" && model.trainingSamples > 0 && (
                                                                    <>
                                                                        <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                                        <span className="text-[10px] bg-emerald-600/20 text-emerald-300 px-2 py-0.5 rounded-md font-bold">
                                                                            {model.trainingSamples} Samples
                                                                        </span>
                                                                    </>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={() => deleteLocalModel(model.name)}
                                                        disabled={deletingModels.has(model.name)}
                                                        className="ml-4 p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-500 hover:bg-red-600/20 hover:text-red-400 hover:border-red-600/50 transition-all disabled:opacity-50"
                                                    >
                                                        {deletingModels.has(model.name) ? (
                                                            <Loader2 className="w-5 h-5 animate-spin" />
                                                        ) : (
                                                            <Trash2 className="w-5 h-5" />
                                                        )}
                                                    </button>
                                                </div>
                                            ))}
                                        </>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div className="rounded-3xl bg-zinc-900 border border-zinc-800 p-8 space-y-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="text-xs text-zinc-500 font-black uppercase tracking-widest">Genetic Controls</div>
                                <div className="text-lg font-black text-white mt-1">Experiment Setup</div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Population Size</label>
                                <input
                                    type="number"
                                    value={200}
                                    readOnly
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-500"
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Mutation Rate</label>
                                <input
                                    type="text"
                                    value="0.05"
                                    readOnly
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-500"
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Generations</label>
                                <input
                                    type="number"
                                    value={100}
                                    readOnly
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-500"
                                />
                            </div>

                            <button
                                type="button"
                                disabled
                                className="w-full py-5 bg-zinc-800 text-zinc-500 rounded-2xl text-xs font-black flex items-center justify-center gap-3 opacity-60 cursor-not-allowed"
                            >
                                Coming Soon
                            </button>
                        </div>
                    </div>

                    <div className="lg:col-span-2 rounded-3xl bg-zinc-900 border border-zinc-800 p-8 space-y-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="text-xs text-zinc-500 font-black uppercase tracking-widest">Live Charts</div>
                                <div className="text-lg font-black text-white mt-1">Fitness & Evolution</div>
                            </div>
                            <div className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">placeholder</div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="rounded-2xl border border-zinc-800 bg-black/40 p-4">
                                <div className="text-[10px] text-zinc-500 uppercase font-black">Best Fitness</div>
                                <div className="mt-3 h-28 rounded-xl border border-zinc-800 bg-zinc-950" />
                            </div>
                            <div className="rounded-2xl border border-zinc-800 bg-black/40 p-4">
                                <div className="text-[10px] text-zinc-500 uppercase font-black">Average Fitness</div>
                                <div className="mt-3 h-28 rounded-xl border border-zinc-800 bg-zinc-950" />
                            </div>
                        </div>

                        <div className="rounded-2xl border border-zinc-800 bg-black/40 p-4">
                            <div className="flex items-center justify-between">
                                <div className="text-[10px] text-zinc-500 uppercase font-black">States</div>
                                <div className="text-[10px] text-zinc-600 font-mono">idle</div>
                            </div>
                            <div className="mt-3 text-xs text-zinc-500">
                                This tab is a dedicated workspace for evolutionary algorithms. You can add more algorithms here later without mixing with Classic training.
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <ConfirmDialog
                isOpen={confirmDeleteOpen}
                title="Remove Model"
                message={`You're about to remove ${pendingDeleteModel ?? "this model"}. This action cannot be undone.`}
                onClose={() => {
                    if (!pendingDeleteModel || !deletingModels.has(pendingDeleteModel)) {
                        setConfirmDeleteOpen(false);
                        setPendingDeleteModel(null);
                    }
                }}
                onConfirm={confirmDeleteLocalModel}
                isLoading={!!pendingDeleteModel && deletingModels.has(pendingDeleteModel)}
                confirmLabel="Remove Model"
                cancelLabel="Keep Model"
                variant="danger"
            />
        </div>
    );
}
