"use client";

import React, { useState, useEffect } from "react";
import { Brain, Map, Activity, RefreshCw, Crosshair, Layers } from "lucide-react";
import AIDecisionSimulator from "../ai-brain/AIDecisionSimulator";
import { toast } from "sonner";

interface AIBrainVisualizerSectionProps {
    selectedModel: string;
    targetSymbol?: string;
    lastRunTimestamp: number;
}

export default function AIBrainVisualizerSection({
    selectedModel,
    targetSymbol,
    lastRunTimestamp
}: AIBrainVisualizerSectionProps) {
    const [modelInfo, setModelInfo] = useState<any>(null);
    const [heatmapData, setHeatmapData] = useState<any>(null);
    const [loadingHeatmap, setLoadingHeatmap] = useState(false);

    const [featX, setFeatX] = useState("RSI");
    const [featY, setFeatY] = useState("Z_Score");

    const [activeSymbolData, setActiveSymbolData] = useState<Record<string, number> | null>(null);

    // Sync activeSymbolData and Heatmap to lastRunTimestamp
    useEffect(() => {
        if (lastRunTimestamp === 0 || !selectedModel) return;

        const refreshData = async () => {
            // 1. Fetch Model Info
            await fetchModelInfo(selectedModel);

            // 2. Fetch Symbol Data for the target symbol
            if (targetSymbol) {
                await fetchSymbolData(targetSymbol);
            }

            // 3. Generate Heatmap
            await generateHeatmap();
        };

        refreshData();
    }, [lastRunTimestamp, selectedModel]); // We watch model/timestamp to refresh

    const fetchSymbolData = async (symbol: string) => {
        try {
            const res = await fetch(`/api/admin/ai-brain/symbol-data/${encodeURIComponent(symbol)}`);
            if (res.ok) {
                const data = await res.json();
                setActiveSymbolData(data);
            }
        } catch (e) {
            console.error("Error fetching symbol data", e);
        }
    };

    const fetchModelInfo = async (name: string) => {
        try {
            const res = await fetch(`/api/admin/models/${encodeURIComponent(name)}/info`);
            const data = await res.json();
            setModelInfo(data);

            const features = data.features || [];
            if (features.length >= 2) {
                setFeatX(features[0]);
                setFeatY(features[1]);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const generateHeatmap = async () => {
        if (!selectedModel) return;
        setLoadingHeatmap(true);
        try {
            const res = await fetch("/api/admin/ai-brain/heatmap", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    modelName: selectedModel,
                    featureX: featX,
                    featureY: featY,
                    fixedValues: {},
                    gridRes: 40
                })
            });
            if (res.ok) {
                setHeatmapData(await res.json());
            }
        } catch (e) {
            console.error("Failed to generate heatmap", e);
        } finally {
            setLoadingHeatmap(false);
        }
    };

    const getColor = (conf: number) => {
        if (conf < 0.5) {
            const r = 255;
            const g = Math.floor(conf * 2 * 255);
            return `rgb(${r}, ${g}, 0)`;
        } else {
            const r = Math.floor((1 - conf) * 2 * 255);
            const g = 255;
            return `rgb(${r}, ${g}, 0)`;
        }
    };

    // If no analysis has been run yet, show a simpler placeholder or just skip
    if (lastRunTimestamp === 0) {
        return (
            <div className="mt-12 border-t border-zinc-800 pt-12 pb-24 text-center">
                <div className="inline-flex p-4 rounded-3xl bg-zinc-900 border border-zinc-800 mb-4 text-zinc-500">
                    <Brain className="w-8 h-8 opacity-20" />
                </div>
                <h3 className="text-xl font-black text-zinc-700">AI Brain Insights</h3>
                <p className="text-zinc-600 text-[10px] mt-2 uppercase tracking-widest font-bold">Run analysis to visualize the model's decision surface</p>
            </div>
        );
    }

    return (
        <div className="space-y-12 pb-24 mt-12 border-t border-zinc-800 pt-12">
            <header className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-black text-white tracking-tight flex items-center gap-4">
                        <Brain className="w-10 h-10 text-indigo-500" />
                        AI Brain Insights
                    </h2>
                    <p className="text-zinc-500 font-bold uppercase tracking-widest text-[10px] mt-2">
                        Intelligence Surface for {targetSymbol || selectedModel}
                    </p>
                </div>

                <div className="hidden md:flex items-center gap-3 px-6 py-2 rounded-2xl bg-zinc-900 border border-zinc-800">
                    <div className="flex flex-col items-end">
                        <span className="text-[8px] text-zinc-500 font-black uppercase tracking-widest">Active Model</span>
                        <span className="text-xs font-bold text-zinc-300">{selectedModel}</span>
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                <div className="xl:col-span-2 space-y-6">
                    <div className="p-8 rounded-[40px] bg-zinc-900 border border-zinc-800 relative overflow-hidden">
                        <div className="flex items-center justify-between mb-8 relative z-10">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                                    <Map className="w-5 h-5" />
                                </div>
                                <h3 className="text-xl font-black text-white">Confidence Surface</h3>
                            </div>

                            <div className="px-4 py-1.5 rounded-full bg-black/50 border border-zinc-800 text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                {featX} × {featY}
                            </div>
                        </div>

                        <div className="aspect-square w-full rounded-2xl bg-black border border-zinc-800 p-4 relative flex flex-col items-center justify-center">
                            {loadingHeatmap ? (
                                <div className="flex flex-col items-center gap-4 text-zinc-600">
                                    <RefreshCw className="w-8 h-8 animate-spin" />
                                    <span className="text-[10px] font-black uppercase">Scanning Surface...</span>
                                </div>
                            ) : heatmapData ? (
                                <div className="w-full h-full relative group">
                                    <div
                                        className="w-full h-full grid"
                                        style={{
                                            gridTemplateColumns: `repeat(${heatmapData.x_range.length}, 1fr)`,
                                            gridTemplateRows: `repeat(${heatmapData.y_range.length}, 1fr)`
                                        }}
                                    >
                                        {heatmapData.confidence_grid.flat().map((conf: number, i: number) => {
                                            const isRisky = heatmapData.risk_zones.flat()[i];
                                            return (
                                                <div
                                                    key={i}
                                                    className="w-full h-full relative"
                                                    style={{ backgroundColor: getColor(conf), opacity: 0.8 }}
                                                >
                                                    {isRisky && i % 4 === 0 && (
                                                        <div className="absolute inset-0 flex items-center justify-center text-[8px] text-black/20 font-black">×</div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>

                                    <div className="absolute bottom-4 right-4 p-4 rounded-2xl bg-black/80 backdrop-blur-md border border-zinc-800/50 flex flex-col gap-2">
                                        <div className="h-1.5 w-32 bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 rounded-full" />
                                        <div className="flex justify-between text-[7px] font-bold text-zinc-500 uppercase tracking-wider">
                                            <span>Caution</span>
                                            <span>Best Entry</span>
                                        </div>
                                    </div>

                                    <div className="absolute -left-8 top-1/2 -rotate-90 text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em]">{featY}</div>
                                    <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em]">{featX}</div>
                                </div>
                            ) : (
                                <div className="text-zinc-700 font-black uppercase text-xs">Awaiting Analysis Data</div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="space-y-8">
                    <div className="p-8 rounded-[40px] bg-indigo-600/10 border border-indigo-500/20 space-y-4">
                        <div className="flex items-center gap-3">
                            <Activity className="w-6 h-6 text-indigo-400" />
                            <h3 className="text-xl font-black text-white">Live Insights</h3>
                        </div>
                        <p className="text-xs text-zinc-400 leading-relaxed font-medium">
                            This surface maps the model's confidence across primary indicators.
                            <span className="text-emerald-400"> Green</span> zones represent high-probability patterns found in training.
                        </p>
                        <div className="pt-4 border-t border-indigo-500/20 grid grid-cols-2 gap-4">
                            <div className="text-center">
                                <div className="text-[10px] text-zinc-500 font-bold uppercase">Safe Path</div>
                                <div className="text-lg font-black text-white">Validated</div>
                            </div>
                            <div className="text-center">
                                <div className="text-[10px] text-zinc-500 font-bold uppercase">Risk Profile</div>
                                <div className="text-lg font-black text-rose-400">Dynamic</div>
                            </div>
                        </div>
                    </div>

                    <div className="p-8 rounded-[40px] bg-zinc-900 border border-zinc-800 space-y-6">
                        <div className="flex items-center gap-3">
                            <Layers className="w-5 h-5 text-zinc-500" />
                            <h3 className="text-lg font-black text-white">Technical Specs</h3>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 rounded-2xl bg-black/50 border border-zinc-800">
                                <span className="text-xs font-bold text-zinc-500">Feature Count</span>
                                <span className="text-xs font-mono text-zinc-300">{modelInfo?.num_features || '---'}</span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-2xl bg-black/50 border border-zinc-800">
                                <span className="text-xs font-bold text-zinc-500">Algorithm</span>
                                <span className="text-xs font-mono text-indigo-400 font-black">{modelInfo?.model_type || '---'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <Crosshair className="w-6 h-6 text-indigo-500" />
                    <h3 className="text-2xl font-black text-white tracking-tight">Real-time Decision Simulator</h3>
                </div>
                {selectedModel && modelInfo && (
                    <AIDecisionSimulator
                        modelName={selectedModel}
                        predictors={modelInfo.features || []}
                        linkedSymbol={targetSymbol}
                        linkedSymbolData={activeSymbolData || undefined}
                        onClearSymbol={() => setActiveSymbolData(null)}
                    />
                )}
            </div>
        </div>
    );
}
