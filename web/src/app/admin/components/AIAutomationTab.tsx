"use client";

import { Zap, ChevronDown, Check, Loader2, Download, Database, Info, History, Trash2, TrendingUp, Clock, Sparkles } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { useState, useEffect, useMemo } from "react";
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
    n_estimators?: number;
    num_trees?: number;
    exchange?: string;
    featurePreset?: string;
    bestIteration?: number;
    target_pct?: number;
    stop_loss_pct?: number;
    look_forward_days?: number;
    learning_rate?: number;

    uses_exchange_index_json?: boolean;
    exchange_index_json_path?: string;
    uses_fundamentals?: boolean;
    fundamentals_loaded?: boolean;
    has_meta_labeling?: boolean;
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
    const [when, setWhen] = useState<string>(""); // datetime-local value
    const [days, setDays] = useState<number>(365);
    const [updatePrices, setUpdatePrices] = useState<boolean>(true);
    const [updateFunds, setUpdateFunds] = useState<boolean>(false);
    const [unified, setUnified] = useState<boolean>(false);
    const [scheduling, setScheduling] = useState<boolean>(false);
    const [triggeringTraining, setTriggeringTraining] = useState<boolean>(false);
    const [useEarlyStopping, setUseEarlyStopping] = useState<boolean>(true);
    const [nEstimators, setNEstimators] = useState<number>(1000);
    const [trainingStrategy, setTrainingStrategy] = useState<"golden" | "grid_small" | "random" | "optuna">("golden");
    const [randomSearchIter, setRandomSearchIter] = useState<number>(5);
    const [optunaTrials, setOptunaTrials] = useState<number>(30);
    const [lastTrainingSummary, setLastTrainingSummary] = useState<{
        exchange: string;
        useEarlyStopping: boolean;
        nEstimators: number;
        trainingSamples: number;
        timestamp: string;
        numFeatures?: number;
        symbolsUsed?: number;
        rawRows?: number;
        bestIteration?: number;
        targetPct?: number;
        stopLossPct?: number;
        lookForwardDays?: number;
    } | null>(null);
    const [localModels, setLocalModels] = useState<LocalModel[]>([]);
    const [loadingLocalModels, setLoadingLocalModels] = useState<boolean>(false);
    const [deletingModels, setDeletingModels] = useState<Set<string>>(new Set());

    // Adaptive Learning State
    const [selectedAdaptiveModel, setSelectedAdaptiveModel] = useState<string | null>(null);
    const [adaptiveResults, setAdaptiveResults] = useState<any[]>([]);
    const [loadingAdaptiveResults, setLoadingAdaptiveResults] = useState<boolean>(false);
    const [adaptiveStats, setAdaptiveStats] = useState<{ total_logs: number; pending: number; } | null>(null);

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
        adaptive_starting: 5,
        adaptive_verifying: 40,
        adaptive_learning: 80,
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
        adaptive_starting: "Start",
        adaptive_verifying: "Verify",
        adaptive_learning: "Learning",
    };
    const overallProgressPct = (() => {
        const phase = trainingStatus?.phase ?? null;
        if (!phase) return null;
        if (phase === "adaptive_verifying" && symbolProgressPct !== null) {
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
    // Updated learning curve to support real metrics
    const [learningCurve, setLearningCurve] = useState<Array<{
        step: number;
        samples?: number | null;
        features?: number | null;
        iteration?: number;
        logloss?: number;
        error?: number;
    }>>([]);
    const [modelName, setModelName] = useState<string>("");
    const [featurePreset, setFeaturePreset] = useState<"core" | "extended" | "max">("extended");
    const [maxFeatures, setMaxFeatures] = useState<number | null>(null);
    const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
    const [pendingDeleteModel, setPendingDeleteModel] = useState<string | null>(null);
    // Sniper Strategy Parameters
    const [targetPct, setTargetPct] = useState<number>(0.03);
    const [stopLossPct, setStopLossPct] = useState<number>(0.06);
    const [lookForwardDays, setLookForwardDays] = useState<number>(20);
    const [learningRate, setLearningRate] = useState<number>(0.05);
    const [patience, setPatience] = useState<number>(50);

    // Meta-Labeling (default ON)
    const [useMetaLabeling, setUseMetaLabeling] = useState<boolean>(true);
    const [metaThreshold, setMetaThreshold] = useState<number>(0.33);

    // Adaptive Learning State
    const [retraining, setRetraining] = useState(false);
    const [updatingActuals, setUpdatingActuals] = useState(false);

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

    useEffect(() => {
        if (!trainingStatus?.running || !trainingStatus.stats) return;

        const stats = trainingStatus.stats;
        const now = Date.now();

        // Check if we have real training metrics (streaming)
        if (typeof stats.iteration === "number") {
            setLearningCurve((prev) => {
                const last = prev[prev.length - 1];
                if (last && last.iteration === stats.iteration) return prev;

                // Extract metrics loosely matching LightGBM output
                // usually: valid_0_logloss, valid_0_error, etc.
                const logloss = typeof stats["valid_0_logloss"] === "number" ? stats["valid_0_logloss"] : undefined;
                const error = typeof stats["valid_0_error"] === "number" ? stats["valid_0_error"] : undefined;

                const next = [...prev, {
                    step: now,
                    iteration: stats.iteration,
                    logloss,
                    error
                }];
                // Keep more history for training curves
                return next.length > 200 ? next.slice(next.length - 200) : next;
            });
            return;
        }

        // Fallback to legacy "samples/features" updates if not streaming metrics
        const samples = typeof stats.samples === "number" ? stats.samples : rowsLoaded ?? null;
        const features = typeof stats.features === "number" ? stats.features : null;
        if (samples === null && features === null) return;

        setLearningCurve((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.samples === samples && last.features === features) {
                return prev;
            }
            const next = [...prev, { step: now, samples, features }];
            return next.length > 80 ? next.slice(next.length - 80) : next;
        });
    }, [trainingStatus?.running, trainingStatus?.stats, rowsLoaded]);

    const estimatedImpact = useMemo(() => {
        // 1. Time per 100 trees depends on feature complexity
        let timePer100 = 20;
        if (featurePreset === "core") timePer100 = 10;
        if (featurePreset === "max") timePer100 = 40;

        // 2. Expected iterations
        let expectedTrees = nEstimators;
        if (useEarlyStopping) {
            // Heuristic for early stopping convergence:
            // Base convergence is roughly proportional to 1/LR
            const convergenceBase = 45 / learningRate;

            // Patience allows more exploration, slightly increasing expected trees
            // Base patience is around 50.
            const patienceFactor = 0.7 + (patience / 150); // e.g. 50 -> 1.03, 150 -> 1.7

            expectedTrees = Math.min(nEstimators, Math.round(convergenceBase * patienceFactor));
        }

        // Duration estimate
        const estSeconds = (expectedTrees / 100) * timePer100;

        let descriptor = "Balanced";
        let descriptorColor = "text-zinc-400";

        if (learningRate <= 0.02) {
            descriptor = "High Precision (Deep)";
            descriptorColor = "text-indigo-400";
        } else if (patience > 100 && learningRate < 0.06) {
            descriptor = "Exhaustive Search";
            descriptorColor = "text-purple-400";
        } else if (learningRate >= 0.1) {
            descriptor = "Fast Preview";
            descriptorColor = "text-amber-400";
        } else if (useEarlyStopping && patience < 30) {
            descriptor = "Early Exit (Rough)";
            descriptorColor = "text-rose-400";
        }

        return {
            trees: expectedTrees,
            time: estSeconds,
            descriptor,
            descriptorColor
        };
    }, [learningRate, nEstimators, patience, useEarlyStopping, featurePreset]);

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

    const fetchAdaptiveStats = async () => {
        if (!trainingExchange) return;
        try {
            let url = `/api/admin/train/adaptive/stats?exchange=${trainingExchange}`;
            if (selectedAdaptiveModel) {
                url += `&model_name=${selectedAdaptiveModel}`;
            }
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                setAdaptiveStats(data);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const fetchAdaptiveResults = async () => {
        if (!trainingExchange) return;
        setLoadingAdaptiveResults(true);
        try {
            let url = `/api/admin/train/adaptive/results?exchange=${trainingExchange}&limit=50`;
            if (selectedAdaptiveModel) {
                url += `&model_name=${selectedAdaptiveModel}`;
            }
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                setAdaptiveResults(Array.isArray(data) ? data : []);
            }
        } catch (e) {
            console.error("Error fetching adaptive results:", e);
        } finally {
            setLoadingAdaptiveResults(false);
        }
    };

    useEffect(() => {
        fetchAdaptiveStats();
        fetchAdaptiveResults();
    }, [trainingExchange, selectedAdaptiveModel]);

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

            <div className="grid grid-cols-1 gap-12">
                {/* Model Artifacts Row */}
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
                                try {
                                    await fetchTrainedModels();
                                } catch {
                                    // ignore errors
                                }
                                await refreshTrainingStatus();
                            }}
                            className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-500 hover:text-indigo-400 transition-all"
                        >
                            <History className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="flex-1 min-h-[400px]">
                        {loadingLocalModels && localModels.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-full gap-4 text-zinc-600 grayscale">
                                <Loader2 className="w-8 h-8 animate-spin" />
                                <p className="text-xs font-bold uppercase tracking-widest">Fetching models...</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 overflow-y-auto pr-2 custom-scrollbar">
                                {localModels.map(model => (
                                    <div
                                        key={model.name}
                                        className="p-6 rounded-3xl bg-black border border-zinc-800/50 hover:border-zinc-700 transition-all flex flex-col justify-between group h-full space-y-6"
                                    >
                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <div className="p-3 rounded-2xl bg-zinc-900 text-zinc-500 group-hover:bg-indigo-500/10 group-hover:text-indigo-400 transition-all">
                                                    <Database className="w-6 h-6" />
                                                </div>
                                                <button
                                                    onClick={() => deleteLocalModel(model.name)}
                                                    disabled={deletingModels.has(model.name)}
                                                    className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-500 hover:bg-red-600/20 hover:text-red-400 hover:border-red-600/50 transition-all disabled:opacity-50"
                                                >
                                                    {deletingModels.has(model.name) ? (
                                                        <Loader2 className="w-4 h-4 animate-spin" />
                                                    ) : (
                                                        <Trash2 className="w-4 h-4" />
                                                    )}
                                                </button>
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-base font-black text-zinc-100 truncate">{model.name}</div>
                                                {model.exchange && (
                                                    <div className="text-[10px] text-indigo-400 uppercase font-black tracking-widest mt-1">{model.exchange}</div>
                                                )}
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="p-2.5 rounded-xl bg-zinc-900/50 border border-zinc-800/50">
                                                    <div className="text-[10px] text-zinc-600 uppercase font-bold">Size</div>
                                                    <div className="text-xs font-mono text-zinc-400">{model.size_mb} MB</div>
                                                </div>
                                                <div className="p-2.5 rounded-xl bg-zinc-900/50 border border-zinc-800/50">
                                                    <div className="text-[10px] text-zinc-600 uppercase font-bold">Modified</div>
                                                    <div className="text-xs text-zinc-400">{new Date(model.modified_at).toLocaleDateString()}</div>
                                                </div>
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {model.num_features !== undefined && (
                                                    <span className="text-[10px] bg-indigo-600/20 text-indigo-300 px-2 py-1 rounded-lg font-bold">
                                                        {model.num_features} Features
                                                    </span>
                                                )}
                                                {model.num_parameters !== undefined && model.num_parameters > 0 && (
                                                    <span className="text-[10px] bg-amber-600/20 text-amber-300 px-2 py-1 rounded-lg font-bold">
                                                        {model.bestIteration ? `${model.bestIteration} Trees` : `${model.num_parameters} Trees`}
                                                    </span>
                                                )}
                                                {typeof model.trainingSamples === "number" && model.trainingSamples > 0 && (
                                                    <span className="text-[10px] bg-emerald-600/20 text-emerald-300 px-2 py-1 rounded-lg font-bold">
                                                        {model.trainingSamples} Samples
                                                    </span>
                                                )}
                                                {model.learning_rate !== undefined && (
                                                    <span className="text-[10px] bg-sky-600/20 text-sky-300 px-2 py-1 rounded-lg font-bold">
                                                        LR: {model.learning_rate}
                                                    </span>
                                                )}

                                                {model.uses_exchange_index_json && (
                                                    <span className="text-[10px] bg-purple-600/20 text-purple-300 px-2 py-1 rounded-lg font-bold">
                                                        Index JSON
                                                    </span>
                                                )}
                                                {model.uses_fundamentals && (
                                                    <span className="text-[10px] bg-emerald-600/20 text-emerald-300 px-2 py-1 rounded-lg font-bold">
                                                        Fundamentals
                                                    </span>
                                                )}
                                                {model.has_meta_labeling && (
                                                    <span className="text-[10px] bg-amber-600/20 text-amber-300 px-2 py-1 rounded-lg font-bold">
                                                        Meta-Labeling
                                                    </span>
                                                )}
                                            </div>
                                        </div>

                                        <div className="pt-4 border-t border-zinc-800/50 grid grid-cols-3 gap-2">
                                            {model.target_pct !== undefined && (
                                                <div className="text-center">
                                                    <div className="text-[9px] text-zinc-600 uppercase font-bold">Target</div>
                                                    <div className="text-[11px] text-emerald-400 font-black">{(model.target_pct * 100).toFixed(0)}%</div>
                                                </div>
                                            )}
                                            {model.stop_loss_pct !== undefined && (
                                                <div className="text-center">
                                                    <div className="text-[9px] text-zinc-600 uppercase font-bold">Stop</div>
                                                    <div className="text-[11px] text-rose-400 font-black">{(model.stop_loss_pct * 100).toFixed(0)}%</div>
                                                </div>
                                            )}
                                            {model.look_forward_days !== undefined && (
                                                <div className="text-center">
                                                    <div className="text-[9px] text-zinc-600 uppercase font-bold">Days</div>
                                                    <div className="text-[11px] text-sky-400 font-black">{model.look_forward_days}d</div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Training Control Cluster Row */}
                <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-8">
                    <div className="flex items-center gap-4">
                        <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                            <Zap className="w-6 h-6" />
                        </div>
                        <div>
                            <h2 className="text-xl font-black text-white">Local Training</h2>
                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Manual Server Processing</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                        {/* Left Column: Settings */}
                        <div className="space-y-6">
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
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${featurePreset === "core"
                                            ? "bg-zinc-100 text-black border-zinc-100"
                                            : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                                            }`}
                                    >
                                        Core
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setFeaturePreset("extended")}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${featurePreset === "extended"
                                            ? "bg-zinc-100 text-black border-zinc-100"
                                            : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                                            }`}
                                    >
                                        Extended
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setFeaturePreset("max")}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${featurePreset === "max"
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

                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1 flex items-center justify-between">
                                        <span>Learning Rate</span>
                                        {trainingStrategy === "optuna" && (
                                            <span className="text-[9px] text-indigo-400 animate-pulse flex items-center gap-1">
                                                <Sparkles className="w-3 h-3" />
                                                AUTO
                                            </span>
                                        )}
                                    </label>
                                    <input
                                        type="number"
                                        min={0.001}
                                        max={0.5}
                                        step={0.001}
                                        value={learningRate}
                                        onChange={(e) => setLearningRate(Number(e.target.value) || 0.05)}
                                        disabled={trainingStrategy === "optuna"}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                                        title={trainingStrategy === "optuna" ? "Automatically managed by Optuna" : ""}
                                    />
                                </div>

                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Patience</label>
                                    <input
                                        type="number"
                                        min={5}
                                        max={200}
                                        step={5}
                                        value={patience}
                                        onChange={(e) => setPatience(Number(e.target.value) || 50)}
                                        disabled={!useEarlyStopping}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 disabled:opacity-50"
                                    />
                                </div>

                                {/* Smart Estimation Card */}
                                <div className="sm:col-span-2 rounded-xl bg-zinc-900/50 border border-zinc-800 p-3 flex flex-col gap-2">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[10px] uppercase font-black text-zinc-500">Estimated Impact</span>
                                        <span className={`text-[10px] font-bold ${estimatedImpact.descriptorColor}`}>
                                            {estimatedImpact.descriptor}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-4 text-xs text-zinc-300">
                                        <div className="flex items-center gap-1.5">
                                            <Clock className="w-3 h-3 text-zinc-500" />
                                            <span>~{Math.ceil(estimatedImpact.time / 60)} min</span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <Database className="w-3 h-3 text-zinc-500" />
                                            <span>~{estimatedImpact.trees} Trees</span>
                                        </div>
                                    </div>
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
                                        <option value="optuna">Optuna Optimization (Bayesian - Recommended)</option>
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

                                {trainingStrategy === "optuna" && (
                                    <div className="space-y-1">
                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Optuna Trials</label>
                                        <input
                                            type="number"
                                            min={5}
                                            max={200}
                                            step={5}
                                            value={optunaTrials}
                                            onChange={(e) => setOptunaTrials(Number(e.target.value) || 30)}
                                            className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                        />
                                    </div>
                                )}

                                {/* Sniper Strategy Settings */}
                                <div className="col-span-2 pt-4 border-t border-zinc-800">
                                    <div className="flex items-center gap-2 mb-4">
                                        <span className="text-[10px] text-zinc-500 uppercase font-black">Sniper Strategy Settings</span>
                                        <span className="text-[10px] text-zinc-600">(Target Labels)</span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-3">
                                        <div className="space-y-1">
                                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Target %</label>
                                            <select
                                                value={targetPct}
                                                onChange={(e) => setTargetPct(Number(e.target.value))}
                                                className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                            >
                                                <option value={0.01}>1%</option>
                                                <option value={0.03}>3%</option>
                                                <option value={0.05}>5%</option>
                                                <option value={0.10}>10%</option>
                                                <option value={0.15}>15%</option>
                                                <option value={0.20}>20%</option>
                                                <option value={0.30}>30%</option>
                                            </select>
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Stop Loss %</label>
                                            <select
                                                value={stopLossPct}
                                                onChange={(e) => setStopLossPct(Number(e.target.value))}
                                                className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                            >
                                                <option value={0.01}>1%</option>
                                                <option value={0.03}>3%</option>
                                                <option value={0.05}>5%</option>
                                                <option value={0.06}>6%</option>
                                                <option value={0.07}>7%</option>
                                                <option value={0.10}>10%</option>
                                            </select>
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Days</label>
                                            <select
                                                value={lookForwardDays}
                                                onChange={(e) => setLookForwardDays(Number(e.target.value))}
                                                className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500"
                                            >
                                                <option value={10}>10 days</option>
                                                <option value={15}>15 days</option>
                                                <option value={20}>20 days</option>
                                                <option value={30}>30 days</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* Meta-Labeling Settings */}
                                <div className="col-span-2 pt-4 border-t border-zinc-800">
                                    <div className="flex items-center justify-between gap-3 mb-4">
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-zinc-500 uppercase font-black">Meta-Labeling</span>
                                            <span className="text-[10px] text-zinc-600">(XGBoost filter)</span>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => setUseMetaLabeling(v => !v)}
                                            className={`px-3 py-2 rounded-xl border text-[11px] font-black transition-all ${useMetaLabeling ? "bg-amber-500/20 border-amber-500/30 text-amber-200" : "bg-zinc-900 border-zinc-800 text-zinc-400"}`}
                                        >
                                            {useMetaLabeling ? "ON" : "OFF"}
                                        </button>
                                    </div>

                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="space-y-1 col-span-2">
                                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">
                                                Meta Confidence Threshold
                                            </label>
                                            <input
                                                type="range"
                                                min={0.1}
                                                max={0.95}
                                                step={0.01}
                                                value={metaThreshold}
                                                onChange={(e) => setMetaThreshold(Number(e.target.value) || 0.30)}
                                                disabled={!useMetaLabeling}
                                                className="w-full"
                                            />
                                            <div className="flex items-center justify-between text-[10px] text-zinc-500 px-1">
                                                <span>0.10</span>
                                                <span className="text-amber-300 font-black">{metaThreshold.toFixed(2)}</span>
                                                <span>0.95</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
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
                                        {lastTrainingSummary.targetPct !== undefined && (
                                            <span className="px-2 py-1 rounded-full bg-emerald-950/30 border border-emerald-500/20 text-[10px]">
                                                Target: <span className="text-emerald-400 font-bold">{(lastTrainingSummary.targetPct * 100).toFixed(0)}%</span>
                                            </span>
                                        )}
                                        {lastTrainingSummary.stopLossPct !== undefined && (
                                            <span className="px-2 py-1 rounded-full bg-rose-950/30 border border-rose-500/20 text-[10px]">
                                                SL: <span className="text-rose-400 font-bold">{(lastTrainingSummary.stopLossPct * 100).toFixed(0)}%</span>
                                            </span>
                                        )}
                                        {lastTrainingSummary.lookForwardDays !== undefined && (
                                            <span className="px-2 py-1 rounded-full bg-sky-950/30 border border-sky-500/20 text-[10px]">
                                                Days: <span className="text-sky-400 font-bold">{lastTrainingSummary.lookForwardDays}</span>
                                            </span>
                                        )}
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
                                            <div className="text-sm font-black text-zinc-100 mt-1">
                                                {lastTrainingSummary.bestIteration ? (
                                                    <span className="flex items-baseline gap-1">
                                                        <span className="text-emerald-400">{lastTrainingSummary.bestIteration}</span>
                                                        <span className="text-zinc-600 text-[10px] font-medium">/ {lastTrainingSummary.nEstimators} (Early Stop)</span>
                                                    </span>
                                                ) : (
                                                    lastTrainingSummary.nEstimators
                                                )}
                                            </div>
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
                                                optunaTrials: trainingStrategy === "optuna" ? optunaTrials : undefined,
                                                maxFeatures: maxFeatures ?? undefined,
                                                targetPct,
                                                stopLossPct,
                                                lookForwardDays,
                                                learningRate,
                                                patience,
                                                useMetaLabeling,
                                                metaThreshold,
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
                        </div>

                        {/* Right Column: Status & Logs */}
                        <div className="space-y-6 lg:border-l lg:border-zinc-800/50 lg:pl-12">
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
                                    <div className="rounded-2xl border border-zinc-800 bg-black/40 p-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <div className="text-[10px] text-zinc-500 uppercase font-black">Learning Curve</div>
                                            <div className="text-[10px] text-zinc-600">Samples / Features</div>
                                        </div>
                                        {learningCurve.length > 1 ? (
                                            <div className="h-40">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={learningCurve} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                                        <XAxis
                                                            dataKey={learningCurve.length > 0 && learningCurve[0].iteration !== undefined ? "iteration" : "step"}
                                                            hide
                                                        />
                                                        <YAxis
                                                            stroke="rgba(255,255,255,0.4)"
                                                            tick={{ fontSize: 10 }}
                                                            domain={['auto', 'auto']}
                                                        />
                                                        <Tooltip
                                                            labelFormatter={(v) => `Iter: ${v}`}
                                                            contentStyle={{ background: "#0a0a0a", border: "1px solid rgba(255,255,255,0.08)", color: "#fff" }}
                                                        />
                                                        {learningCurve.length > 0 && learningCurve[0].iteration !== undefined ? (
                                                            <>
                                                                <Line type="monotone" dataKey="logloss" name="Log Loss" stroke="#f43f5e" strokeWidth={2} dot={false} isAnimationActive={false} />
                                                                <Line type="monotone" dataKey="error" name="Error Rate" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Line type="monotone" dataKey="samples" stroke="#38bdf8" strokeWidth={2} dot={false} />
                                                                <Line type="monotone" dataKey="features" stroke="#a855f7" strokeWidth={2} dot={false} />
                                                            </>
                                                        )}
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        ) : (
                                            <div className="h-40 flex items-center justify-center text-[11px] text-zinc-500">
                                                Waiting for live training updates...
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
                        </div>
                    </div>

                    {/* 
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
                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Schedule (Cron)</label>
                                        <div className="flex gap-2">
                                            <input
                                                type="text"
                                                value={when}
                                                onChange={(e) => setWhen(e.target.value)}
                                                placeholder="Cron expression (e.g. 30 22 * * *)"
                                                className="flex-1 py-2 px-3 bg-black border border-zinc-800 rounded-lg text-xs text-zinc-300 focus:border-indigo-500"
                                            />
                                        </div>
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
                            */}

                    {/* ADAPTIVE LEARNING SECTION */}
                    <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-6">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="p-2.5 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
                                    <TrendingUp className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-xl font-black text-white">Adaptive Learning & Review</p>
                                    <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Manual Retraining</p>
                                </div>
                            </div>

                            {/* Model Selector */}
                            <div className="relative min-w-[200px]">
                                <select
                                    value={selectedAdaptiveModel || ""}
                                    onChange={(e) => setSelectedAdaptiveModel(e.target.value || null)}
                                    className="w-full px-4 py-2.5 rounded-xl bg-black border border-zinc-800 text-xs font-bold text-white outline-none focus:border-indigo-500 appearance-none cursor-pointer"
                                    disabled={!trainingExchange}
                                >
                                    <option value="">All Models (Exchange Wide)</option>
                                    {localModels
                                        .filter(m => !trainingExchange || m.exchange === trainingExchange)
                                        .map(m => (
                                            <option key={m.name} value={m.name}>{m.name}</option>
                                        ))}
                                </select>
                                <ChevronDown className="bg-black w-4 h-4 text-zinc-500 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                            <div className="space-y-6">
                                <div className="space-y-2 text-sm text-zinc-400 font-medium leading-relaxed">
                                    <p>Review recent predictions and retrain the model on mistakes to improve accuracy over time. This process helps the AI adapt to shifting market regimes.</p>
                                    {!trainingExchange && (
                                        <div className="p-4 mt-4 rounded-2xl bg-amber-900/10 border border-amber-500/20 text-amber-500 text-xs font-bold flex items-center gap-3">
                                            <Info className="w-5 h-5 shrink-0" />
                                            Select an exchange above to enable adaptive learning actions.
                                        </div>
                                    )}
                                </div>

                                {adaptiveStats && (
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-4 rounded-2xl bg-black border border-zinc-800 shadow-xl shadow-black/20">
                                            <p className="text-[10px] text-zinc-500 uppercase font-black tracking-widest mb-1">Total Logs</p>
                                            <p className="text-2xl font-black text-white">{adaptiveStats.total_logs}</p>
                                        </div>
                                        <div className="p-4 rounded-2xl bg-black border border-zinc-800 shadow-xl shadow-black/20">
                                            <p className="text-[10px] text-zinc-500 uppercase font-black tracking-widest mb-1">Pending Verify</p>
                                            <p className="text-2xl font-black text-amber-500">{adaptiveStats.pending}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Recent Results Table */}
                                <div className="rounded-2xl bg-black/50 border border-zinc-800/50 overflow-hidden ring-1 ring-zinc-800">
                                    <div className="px-4 py-3 border-b border-zinc-800/50 flex justify-between items-center bg-black/40">
                                        <h3 className="text-[10px] font-black text-zinc-400 uppercase tracking-wider">Recent Verified Predictions</h3>
                                        {loadingAdaptiveResults && <Loader2 className="w-3 h-3 animate-spin text-zinc-500" />}
                                    </div>
                                    <div className="max-h-[220px] overflow-y-auto custom-scrollbar">
                                        {adaptiveResults.length === 0 ? (
                                            <div className="p-8 text-center text-[10px] text-zinc-700 font-bold uppercase tracking-widest">No verified results found</div>
                                        ) : (
                                            <table className="w-full text-left border-collapse">
                                                <thead className="sticky top-0 bg-zinc-900/90 backdrop-blur-sm text-[9px] text-zinc-500 font-bold uppercase tracking-wider z-10">
                                                    <tr>
                                                        <th className="px-3 py-2">Date</th>
                                                        <th className="px-3 py-2">Sym</th>
                                                        <th className="px-3 py-2 text-center">Pred</th>
                                                        <th className="px-3 py-2 text-center">Res</th>
                                                        <th className="px-3 py-2 text-right">Entry</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="text-[10px] font-medium text-zinc-300 divide-y divide-zinc-800/30">
                                                    {adaptiveResults.map((row, i) => (
                                                        <tr key={i} className="hover:bg-zinc-800/30 transition-colors">
                                                            <td className="px-3 py-2 text-zinc-500 font-mono whitespace-nowrap">{new Date(row.date || row.created_at).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}</td>
                                                            <td className="px-3 py-2 text-white font-bold">{row.symbol}</td>
                                                            <td className="px-3 py-2 text-center">
                                                                <span className={`px-1.5 py-0.5 rounded-[4px] text-[9px] font-black ${row.prediction === 1 ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                                                                    {row.prediction === 1 ? "BUY" : "SELL"}
                                                                </span>
                                                            </td>
                                                            <td className="px-3 py-2 text-center">
                                                                {row.status === "win" && <span className="text-emerald-500 font-black">WIN</span>}
                                                                {row.status === "loss" && <span className="text-rose-500 font-black">LOSS</span>}
                                                                {row.status !== "win" && row.status !== "loss" && <span className="text-zinc-700">-</span>}
                                                            </td>
                                                            <td className="px-3 py-2 text-right font-mono text-zinc-400">{row.entry_price?.toFixed(2)}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <div className="flex flex-col justify-center space-y-4 lg:border-l lg:border-zinc-800/50 lg:pl-12">

                                <button
                                    onClick={async () => {
                                        console.log("[Adaptive] Verify Predictions clicked. Exchange:", trainingExchange);
                                        if (!trainingExchange) {
                                            toast.error("Please select an exchange first");
                                            return;
                                        }
                                        setUpdatingActuals(true);
                                        try {
                                            console.log("[Adaptive] Calling update-actuals...");
                                            await fetch("/api/admin/train/adaptive/update-actuals", {
                                                method: "POST",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({
                                                    exchange: trainingExchange,
                                                    look_forward_days: lookForwardDays,
                                                    model_name: selectedAdaptiveModel // Pass selected model if matched
                                                })
                                            });
                                            console.log("[Adaptive] Update actuals started.");
                                            // The global polling will pick up the status message, no toast needed as per request
                                        } catch (e) {
                                            console.error("[Adaptive] Update error:", e);
                                            toast.error("Failed to start update");
                                            setUpdatingActuals(false);
                                        }
                                        // Delay re-enabling locally to allow global state to pick up 'running=true'
                                        setTimeout(() => setUpdatingActuals(false), 3000);
                                    }}
                                    disabled={updatingActuals || (trainingStatus?.running === true)}
                                    className={`py-3 px-4 rounded-xl text-xs font-bold text-white flex items-center justify-center gap-2 disabled:opacity-50 transition-all ${!trainingExchange ? 'bg-zinc-800 border-zinc-700 text-zinc-500 cursor-not-allowed hover:bg-zinc-800' : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-700 border'}`}
                                >
                                    {updatingActuals ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                                    VERIFY PREDICTIONS
                                </button>

                                {trainingStatus?.running && trainingStatus.last_message && (
                                    <div className="space-y-4">
                                        <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 text-[10px] font-mono text-purple-400 text-center animate-pulse">
                                            {trainingStatus.last_message}
                                        </div>
                                        {overallProgressPct !== null && trainingStatus.phase?.startsWith('adaptive') && (
                                            <div className="space-y-1.5 px-1">
                                                <div className="flex items-center justify-between text-[9px] font-black uppercase tracking-widest text-zinc-500">
                                                    <span>{trainingStatus.phase.replace('adaptive_', '').replace('_', ' ')}</span>
                                                    <span>{overallProgressPct}%</span>
                                                </div>
                                                <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden border border-zinc-700/50">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-purple-600 to-indigo-500 transition-all duration-500"
                                                        style={{ width: `${overallProgressPct}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                <button
                                    onClick={async () => {
                                        console.log("[Adaptive] Retrain clicked. Exchange:", trainingExchange);
                                        if (!trainingExchange) {
                                            toast.error("Please select an exchange first");
                                            return;
                                        }
                                        setRetraining(true);
                                        try {
                                            console.log("[Adaptive] Calling retrain...");
                                            const res = await fetch("/api/admin/train/adaptive/retrain", {
                                                method: "POST",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({
                                                    exchange: trainingExchange,
                                                    lookback_days: 30,
                                                    model_name: selectedAdaptiveModel
                                                })
                                            });
                                            if (res.ok) {
                                                console.log("[Adaptive] Retraining started successfully.");
                                                // Removed toast as per request
                                            } else {
                                                const err = await res.text();
                                                console.error("[Adaptive] Retrain failed:", err);
                                                toast.error("Retraining failed: " + err);
                                                setRetraining(false);
                                            }
                                        } catch (e) {
                                            console.error("[Adaptive] Retrain connection error:", e);
                                            toast.error("Failed to start retraining");
                                            setRetraining(false);
                                        }
                                        // Delay re-enabling locally to allow global state to pick up
                                        setTimeout(() => setRetraining(false), 3000);
                                    }}
                                    disabled={retraining || (trainingStatus?.running === true)}
                                    className={`py-3 px-4 rounded-xl text-xs font-bold text-white flex items-center justify-center gap-2 disabled:opacity-50 shadow-[0_0_15px_rgba(168,85,247,0.3)] transition-all ${!trainingExchange ? 'bg-zinc-800 text-zinc-500 shadow-none cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-500'}`}
                                >
                                    {retraining ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                                    RETRAIN ON MISTAKES
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

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
        </div >
    );
}
