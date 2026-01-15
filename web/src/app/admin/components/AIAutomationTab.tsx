"use client";

import { Zap, ChevronDown, Check, Loader2, Download, Database, Info, History } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

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
                            <div className="space-y-2">
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

                            <button
                                onClick={async () => {
                                    if (!trainingExchange) return;
                                    setIsTraining(true);
                                    try {
                                        const res = await fetch("/api/admin/train/local", {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ exchange: trainingExchange })
                                        });
                                        const data = await res.json();
                                        if (res.ok) {
                                            toast.success("Local training started", {
                                                description: data.message
                                            });
                                        } else {
                                            toast.error(data.detail || "Failed to start local training");
                                        }
                                    } catch (e) {
                                        toast.error("Connection error");
                                    } finally {
                                        setIsTraining(false);
                                    }
                                }}
                                disabled={!trainingExchange || isTraining}
                                className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl text-xs font-black flex items-center justify-center gap-3 disabled:opacity-50 transition-all shadow-[0_0_20px_rgba(99,102,241,0.2)]"
                            >
                                {isTraining ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
                                RUN SERVER TRAINING
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
                                onClick={fetchTrainedModels}
                                className="p-2.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-500 hover:text-indigo-400 transition-all"
                            >
                                <History className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex-1 min-h-[400px] space-y-3 overflow-y-auto pr-2 custom-scrollbar">
                            {loadingModels ? (
                                <div className="flex flex-col items-center justify-center h-full gap-4 text-zinc-600 grayscale">
                                    <Loader2 className="w-8 h-8 animate-spin" />
                                    <p className="text-xs font-bold uppercase tracking-widest">Fetching models...</p>
                                </div>
                            ) : trainedModels.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-full gap-4 text-zinc-700 grayscale">
                                    <Database className="w-12 h-12" />
                                    <p className="text-xs font-bold uppercase tracking-widest">No models found in storage</p>
                                </div>
                            ) : (
                                trainedModels.map(model => (
                                    <div key={model.name} className="p-5 rounded-2xl bg-black border border-zinc-800/50 hover:border-zinc-700 transition-all flex items-center justify-between group">
                                        <div className="flex items-center gap-4 min-w-0">
                                            <div className="p-2.5 rounded-xl bg-zinc-900 text-zinc-500 group-hover:bg-indigo-500/10 group-hover:text-indigo-400 transition-all">
                                                <Database className="w-5 h-5" />
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-sm font-black text-zinc-100 truncate">{model.name}</div>
                                                <div className="flex items-center gap-3 mt-0.5">
                                                    <span className="text-[10px] text-zinc-600 font-mono">{(model.metadata?.size / 1024 / 1024).toFixed(2)} MB</span>
                                                    <span className="w-1 h-1 rounded-full bg-zinc-800" />
                                                    <span className="text-[10px] text-zinc-600 font-bold uppercase">{new Date(model.created_at).toLocaleDateString()}</span>
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleDownloadModel(model.name)}
                                            className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-500 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-all"
                                        >
                                            <Download className="w-5 h-5" />
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
