"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Brain, Layers, Crosshair, Map, Activity, Search, RefreshCw, ChevronRight, X, Loader2 } from "lucide-react";
import AIDecisionSimulator from "./AIDecisionSimulator";
import { toast } from "sonner";
import { getCountries, searchSymbols } from "@/lib/api";

export default function AIBrainDashboard() {
    const [localModels, setLocalModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("");
    const [modelInfo, setModelInfo] = useState<any>(null);
    const [heatmapData, setHeatmapData] = useState<any>(null);
    const [loadingHeatmap, setLoadingHeatmap] = useState(false);

    const [featX, setFeatX] = useState("RSI");
    const [featY, setFeatY] = useState("Z_Score");

    const [activeSymbol, setActiveSymbol] = useState<string>("");
    const [activeSymbolData, setActiveSymbolData] = useState<Record<string, number> | null>(null);
    const [searching, setSearching] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [showResults, setShowResults] = useState(false);

    const [countries, setCountries] = useState<any[]>([]);
    const [selectedCountry, setSelectedCountry] = useState<string>("");

    useEffect(() => {
        fetchModels();
        fetchInventory();
    }, []);

    const fetchInventory = async () => {
        try {
            const res = await fetch("/api/admin/db-inventory");
            if (res.ok) {
                const data: any[] = await res.json();
                // Get unique countries and sort them
                const uniqueCountries = Array.from(new Set(data.map(item => item.country)))
                    .filter(c => c && c !== "Unknown")
                    .sort();

                setCountries(uniqueCountries);
                if (uniqueCountries.length > 0) {
                    // Try to default to USA or first one
                    const defaultC = uniqueCountries.find(c => c.toLowerCase() === "usa") || uniqueCountries[0];
                    setSelectedCountry(defaultC);
                }
            }
        } catch (e) {
            console.error("Failed to fetch inventory", e);
        }
    };

    const handleSearch = async (query: string) => {
        setSearchQuery(query);
        if (query.length < 2) {
            setSearchResults([]);
            return;
        }
        setSearching(true);
        try {
            // Updated to pass selected country
            const results = await searchSymbols(query, selectedCountry);
            setSearchResults(results.slice(0, 5));
            setShowResults(true);
        } catch (e) {
            console.error(e);
        } finally {
            setSearching(false);
        }
    };

    const handleSymbolSelect = async (symbol: string) => {
        setActiveSymbol(symbol);
        setShowResults(false);
        setSearchQuery(symbol);
        try {
            const res = await fetch(`/api/admin/ai-brain/symbol-data/${encodeURIComponent(symbol)}`);
            if (res.ok) {
                const data = await res.json();
                setActiveSymbolData(data);
                toast.success(`Linked to ${symbol} data`);
            } else {
                toast.error("Failed to fetch symbol indicators");
            }
        } catch (e) {
            toast.error("Error linking symbol");
        }
    };

    const fetchModels = async () => {
        try {
            const res = await fetch("/api/admin/train/models");
            const data = await res.json();
            setLocalModels(data.models || []);
            if (data.models && data.models.length > 0) {
                setSelectedModel(data.models[0].name);
            }
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        if (!selectedModel) return;
        fetchModelInfo(selectedModel);
    }, [selectedModel]);

    const fetchModelInfo = async (name: string) => {
        try {
            const res = await fetch(`/api/admin/models/${encodeURIComponent(name)}/info`);
            const data = await res.json();
            setModelInfo(data);

            // Auto-select features if available
            const features = data.features || [];
            if (features.length >= 2) {
                if (!features.includes(featX)) setFeatX(features[0]);
                if (!features.includes(featY)) setFeatY(features[1]);
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
                    fixedValues: {}, // Defaults used in backend
                    gridRes: 40
                })
            });
            if (res.ok) {
                setHeatmapData(await res.json());
            }
        } catch (e) {
            toast.error("Failed to generate heatmap");
        } finally {
            setLoadingHeatmap(false);
        }
    };

    useEffect(() => {
        if (modelInfo) {
            generateHeatmap();
        }
    }, [modelInfo, featX, featY]);

    const getColor = (conf: number) => {
        // 0 -> Red, 0.5 -> Yellow, 1 -> Green
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

    return (
        <div className="space-y-12 pb-24">
            <header className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div>
                    <h2 className="text-4xl font-black text-white tracking-tight flex items-center gap-4">
                        <Brain className="w-10 h-10 text-indigo-500" />
                        AI Brain Visualizer
                    </h2>
                    <p className="text-zinc-500 font-bold uppercase tracking-widest text-[10px] mt-2">
                        Confidence Heatmaps & Real-time Decision Simulation
                    </p>
                </div>

                <div className="flex flex-col md:flex-row items-center gap-4 bg-zinc-900 p-2 rounded-2xl border border-zinc-800">
                    {/* Country Selector */}
                    <div className="flex items-center gap-2 pl-2">
                        <Map className="w-4 h-4 text-zinc-500" />
                        <select
                            value={selectedCountry}
                            onChange={(e) => setSelectedCountry(e.target.value)}
                            className="bg-transparent text-xs font-bold text-zinc-300 outline-none cursor-pointer max-w-[120px] truncate"
                        >
                            <option value="">All Countries</option>
                            {countries.map(c => (
                                <option key={c} value={c}>{c}</option>
                            ))}
                        </select>
                    </div>

                    <div className="w-px h-6 bg-zinc-800 hidden md:block" />

                    {/* Symbol Selector */}
                    <div className="relative w-full md:w-64">
                        <div className="flex items-center gap-2 bg-black/50 p-2 rounded-xl border border-zinc-800 focus-within:border-indigo-500 transition-all">
                            <Search className="w-4 h-4 text-zinc-500" />
                            <input
                                type="text"
                                placeholder="Link Symbol (e.g. AAPL)..."
                                value={searchQuery}
                                onChange={(e) => handleSearch(e.target.value)}
                                onFocus={() => searchQuery.length >= 2 && setShowResults(true)}
                                className="bg-transparent text-xs font-bold text-zinc-300 outline-none w-full"
                            />
                            {searching ? (
                                <RefreshCw className="w-3.5 h-3.5 text-indigo-500 animate-spin" />
                            ) : activeSymbol && (
                                <button onClick={() => { setActiveSymbol(""); setActiveSymbolData(null); setSearchQuery(""); }}>
                                    <X className="w-3.5 h-3.5 text-zinc-500 hover:text-white" />
                                </button>
                            )}
                        </div>

                        {/* Search Results Dropdown */}
                        {showResults && searchResults.length > 0 && (
                            <div className="absolute top-full left-0 right-0 mt-2 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl z-50 overflow-hidden divide-y divide-zinc-800/50">
                                {searchResults.map((r) => (
                                    <button
                                        key={r.symbol}
                                        onClick={() => handleSymbolSelect(r.symbol)}
                                        className="w-full text-left p-3 hover:bg-indigo-500/10 flex items-center justify-between group transition-colors"
                                    >
                                        <div className="flex flex-col">
                                            <span className="text-xs font-black text-white group-hover:text-indigo-400">{r.symbol}</span>
                                            <span className="text-[9px] text-zinc-500 truncate w-32">{r.name}</span>
                                        </div>
                                        <ChevronRight className="w-3.5 h-3.5 text-zinc-700 group-hover:text-indigo-500" />
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    <div className="w-px h-6 bg-zinc-800 hidden md:block" />

                    <div className="flex items-center gap-2 pr-2">
                        <Layers className="w-4 h-4 text-zinc-500 ml-2" />
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="bg-transparent text-sm font-bold text-zinc-300 outline-none cursor-pointer"
                        >
                            {localModels.map(m => (
                                <option key={m.name} value={m.name}>{m.name}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                {/* Heatmap Section */}
                <div className="xl:col-span-2 space-y-6">
                    <div className="p-8 rounded-[40px] bg-zinc-900 border border-zinc-800 relative overflow-hidden">
                        <div className="flex items-center justify-between mb-8 relative z-10">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                                    <Map className="w-5 h-5" />
                                </div>
                                <h3 className="text-xl font-black text-white">Confidence Surface</h3>
                            </div>

                            <div className="flex items-center gap-2">
                                <select
                                    value={featX}
                                    onChange={(e) => setFeatX(e.target.value)}
                                    className="bg-black/50 border border-zinc-800 rounded-lg text-[10px] font-black uppercase p-2 outline-none text-zinc-400"
                                >
                                    {modelInfo?.features?.map((f: string) => <option key={f} value={f}>{f}</option>)}
                                </select>
                                <span className="text-zinc-700 font-black">×</span>
                                <select
                                    value={featY}
                                    onChange={(e) => setFeatY(e.target.value)}
                                    className="bg-black/50 border border-zinc-800 rounded-lg text-[10px] font-black uppercase p-2 outline-none text-zinc-400"
                                >
                                    {modelInfo?.features?.map((f: string) => <option key={f} value={f}>{f}</option>)}
                                </select>
                            </div>
                        </div>

                        {/* Heatmap Grid */}
                        <div className="aspect-square w-full rounded-2xl bg-black border border-zinc-800 p-4 relative flex flex-col items-center justify-center">
                            {loadingHeatmap ? (
                                <div className="flex flex-col items-center gap-4 text-zinc-600">
                                    <RefreshCw className="w-8 h-8 animate-spin" />
                                    <span className="text-[10px] font-black uppercase">Generating Brain Surface...</span>
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
                                                    title={`Confidence: ${(conf * 100).toFixed(1)}%`}
                                                >
                                                    {isRisky && i % 4 === 0 && (
                                                        <div className="absolute inset-0 flex items-center justify-center text-[8px] text-black/20 font-black">×</div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>

                                    {/* Legend */}
                                    <div className="absolute bottom-4 right-4 p-4 rounded-2xl bg-black/80 backdrop-blur-md border border-zinc-800/50 flex flex-col gap-2">
                                        <div className="flex items-center gap-2 text-[8px] font-black uppercase tracking-widest">
                                            <span className="w-20">Confidence Map</span>
                                        </div>
                                        <div className="h-1.5 w-full bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 rounded-full" />
                                        <div className="flex justify-between text-[7px] font-bold text-zinc-500">
                                            <span>Low</span>
                                            <span>High</span>
                                        </div>
                                    </div>

                                    {/* Axis Labels */}
                                    <div className="absolute -left-8 top-1/2 -rotate-90 text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em]">{featY}</div>
                                    <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em]">{featX}</div>
                                </div>
                            ) : (
                                <div className="text-zinc-700 font-black uppercase text-xs">Select features to view map</div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Sidebar Analytics */}
                <div className="space-y-8">
                    <div className="p-8 rounded-[40px] bg-indigo-600/10 border border-indigo-500/20 space-y-4">
                        <div className="flex items-center gap-3">
                            <Activity className="w-6 h-6 text-indigo-400" />
                            <h3 className="text-xl font-black text-white">Live Insights</h3>
                        </div>
                        <p className="text-xs text-zinc-400 leading-relaxed font-medium">
                            The visualization on the left shows the model's "probability surface".
                            <span className="text-emerald-400"> Green zones</span> indicate high-confidence entry points, while
                            <span className="text-rose-400"> red zones</span> signify extreme caution or short signals.
                        </p>
                        <div className="pt-4 border-t border-indigo-500/20 grid grid-cols-2 gap-4">
                            <div className="text-center">
                                <div className="text-[10px] text-zinc-500 font-bold uppercase">Safe Nodes</div>
                                <div className="text-lg font-black text-white">74%</div>
                            </div>
                            <div className="text-center">
                                <div className="text-[10px] text-zinc-500 font-bold uppercase">Risk Ratio</div>
                                <div className="text-lg font-black text-rose-400">12%</div>
                            </div>
                        </div>
                    </div>

                    <div className="p-8 rounded-[40px] bg-zinc-900 border border-zinc-800 space-y-6">
                        <div className="flex items-center gap-3">
                            <Layers className="w-5 h-5 text-zinc-500" />
                            <h3 className="text-lg font-black text-white">Model Specs</h3>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 rounded-2xl bg-black/50 border border-zinc-800">
                                <span className="text-xs font-bold text-zinc-500">Estimators</span>
                                <span className="text-xs font-mono text-zinc-300">{modelInfo?.num_parameters || '---'}</span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-2xl bg-black/50 border border-zinc-800">
                                <span className="text-xs font-bold text-zinc-500">Feature Count</span>
                                <span className="text-xs font-mono text-zinc-300">{modelInfo?.num_features || '---'}</span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-2xl bg-black/50 border border-zinc-800">
                                <span className="text-xs font-bold text-zinc-500">Model Type</span>
                                <span className="text-xs font-mono text-indigo-400 font-black">{modelInfo?.model_type || '---'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Simulator Section */}
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <Crosshair className="w-6 h-6 text-indigo-500" />
                    <h3 className="text-2xl font-black text-white tracking-tight">Real-time Decision Simulator</h3>
                </div>
                {selectedModel && modelInfo && (
                    <AIDecisionSimulator
                        modelName={selectedModel}
                        predictors={modelInfo.features || []}
                        linkedSymbol={activeSymbol}
                        linkedSymbolData={activeSymbolData || undefined}
                        onClearSymbol={() => {
                            setActiveSymbol("");
                            setActiveSymbolData(null);
                            setSearchQuery("");
                        }}
                    />
                )}
            </div>
        </div>
    );
}
