"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Zap, AlertTriangle, TrendingUp, TrendingDown, Info, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface AIDecisionSimulatorProps {
    modelName: string;
    predictors: string[];
    linkedSymbol?: string;
    linkedSymbolData?: Record<string, number>;
    onClearSymbol?: () => void;
}

export default function AIDecisionSimulator({
    modelName,
    predictors,
    linkedSymbol,
    linkedSymbolData,
    onClearSymbol
}: AIDecisionSimulatorProps) {
    const [featureValues, setFeatureValues] = useState<Record<string, number>>({});
    const [simulation, setSimulation] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    // Initialize/Sync features
    useEffect(() => {
        if (linkedSymbol && linkedSymbolData) {
            // Filter only predictors that the model expects
            const synced: Record<string, number> = {};
            predictors.forEach(p => {
                synced[p] = linkedSymbolData[p] ?? 0;
            });
            setFeatureValues(synced);
            return;
        }

        const defaults: Record<string, number> = {};
        predictors.slice(0, 10).forEach(p => {
            if (p === "RSI") defaults[p] = 50;
            else if (p.includes("Z_Score")) defaults[p] = 0;
            else if (p.includes("BB_PctB")) defaults[p] = 0.5;
            else defaults[p] = 0;
        });
        setFeatureValues(defaults);
    }, [predictors, linkedSymbol, linkedSymbolData]);

    const runSimulation = useCallback(async (vals: Record<string, number>) => {
        setLoading(true);
        try {
            const res = await fetch("/api/admin/ai-brain/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    modelName,
                    featureValues: vals
                })
            });
            if (res.ok) {
                setSimulation(await res.json());
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, [modelName]);

    // Debounced simulation
    useEffect(() => {
        if (Object.keys(featureValues).length === 0) return;
        const timer = setTimeout(() => {
            runSimulation(featureValues);
        }, 300);
        return () => clearTimeout(timer);
    }, [featureValues, runSimulation]);

    const handleSliderChange = (feature: string, val: number) => {
        setFeatureValues(prev => ({ ...prev, [feature]: val }));
    };

    const getConfidenceColor = (conf: number) => {
        if (conf > 0.75) return "text-emerald-400";
        if (conf > 0.5) return "text-amber-400";
        return "text-rose-400";
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-6 rounded-3xl bg-zinc-900 border border-zinc-800">
            {/* Left: Sliders */}
            <div className="space-y-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                            <TrendingUp className="w-5 h-5" />
                        </div>
                        <h3 className="text-lg font-black text-white">Feature Simulator</h3>
                    </div>
                    {linkedSymbol && (
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                            <Zap className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-[10px] font-black text-emerald-400 uppercase tracking-wider">Linked: {linkedSymbol}</span>
                            <button
                                onClick={onClearSymbol}
                                className="ml-2 p-1 hover:bg-emerald-500/20 rounded-lg text-emerald-400"
                                title="Unlock Sliders"
                            >
                                <Info className="w-3 h-3 hover:text-white" />
                            </button>
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-1 gap-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                    {Object.keys(featureValues).map(feature => (
                        <div key={feature} className={`space-y-2 p-3 rounded-2xl border transition-all ${linkedSymbol ? 'bg-zinc-800/20 border-zinc-800/50 opacity-60' : 'bg-black/40 border-zinc-800/50'}`}>
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-tighter">{feature}</label>
                                <span className="text-xs font-mono text-indigo-400 font-bold">{featureValues[feature].toFixed(2)}</span>
                            </div>
                            <input
                                type="range"
                                min={feature === "RSI" ? 0 : -5}
                                max={feature === "RSI" ? 100 : 5}
                                step={0.01}
                                value={featureValues[feature]}
                                onChange={(e) => handleSliderChange(feature, parseFloat(e.target.value))}
                                disabled={!!linkedSymbol}
                                className={`w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 ${linkedSymbol ? 'cursor-not-allowed hidden' : ''}`}
                            />
                            {linkedSymbol && (
                                <div className="h-1.5 w-full bg-indigo-500/20 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-indigo-500/50"
                                        style={{ width: `${Math.min(100, Math.max(0, feature === "RSI" ? featureValues[feature] : (featureValues[feature] + 5) * 10))}%` }}
                                    />
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Right: Output */}
            <div className="flex flex-col gap-6">
                <div className="flex-1 p-8 rounded-3xl bg-black border border-zinc-800 flex flex-col items-center justify-center text-center space-y-6 relative overflow-hidden">
                    {loading && (
                        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10">
                            <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
                        </div>
                    )}

                    <div className="p-4 rounded-3xl bg-zinc-900/50 border border-zinc-800/50 w-full">
                        <div className="text-[10px] uppercase font-black text-zinc-500 tracking-widest mb-2">Current Brain Response</div>
                        <div className={`text-6xl font-black tracking-tighter ${simulation?.decision === 'BUY' ? 'text-emerald-500' : simulation?.decision === 'WAIT' ? 'text-amber-500' : 'text-rose-500'}`}>
                            {simulation?.decision || '---'}
                        </div>
                    </div>

                    <div className="w-full space-y-2">
                        <div className="flex items-center justify-between text-xs font-bold uppercase text-zinc-500 px-1">
                            <span>Confidence</span>
                            <span className={getConfidenceColor(simulation?.confidence ?? 0)}>
                                {((simulation?.confidence ?? 0) * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="h-4 w-full bg-zinc-900 rounded-full overflow-hidden border border-zinc-800 p-0.5">
                            <div
                                className={`h-full rounded-full transition-all duration-500 ${simulation?.confidence > 0.7 ? 'bg-emerald-500' : simulation?.confidence > 0.4 ? 'bg-amber-500' : 'bg-rose-500'}`}
                                style={{ width: `${(simulation?.confidence ?? 0) * 100}%` }}
                            />
                        </div>
                    </div>

                    {simulation?.is_risky && (
                        <div className="flex items-center gap-2 p-3 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-bold w-full">
                            <AlertTriangle className="w-4 h-4 shrink-0" />
                            <span>High uncertainty area! Decisions may be unreliable here.</span>
                        </div>
                    )}
                </div>

                {/* Sensitivity Analysis */}
                <div className="p-6 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-4">
                    <div className="flex items-center justify-between">
                        <h4 className="text-xs font-black text-zinc-400 uppercase tracking-widest">Brain Sensitivity</h4>
                        <Info className="w-4 h-4 text-zinc-600" />
                    </div>
                    <div className="space-y-2">
                        {simulation?.sensitivity && Object.entries(simulation.sensitivity)
                            .sort((a: any, b: any) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 4)
                            .map(([key, val]: [string, any]) => (
                                <div key={key} className="flex items-center gap-4">
                                    <div className="text-[10px] font-bold text-zinc-500 w-24 truncate">{key}</div>
                                    <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden flex">
                                        <div
                                            className={`h-full ${val > 0 ? 'bg-emerald-500/50 ml-auto' : 'bg-rose-500/50 mr-auto'}`}
                                            style={{ width: `${Math.min(100, Math.abs(val) * 500)}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
