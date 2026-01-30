"use client";

import { Database, Globe, Loader2, Download, Check, ChevronLeft, ChevronRight, BarChart3, History, Zap, Cloud, RefreshCcw } from "lucide-react";
import { type SymbolResult } from "@/lib/api";

interface DataManagerTabProps {
    selectedCountry: string;
    setCountryDialogOpen: (open: boolean) => void;
    processing: boolean;
    dataSourcesTab: "prices" | "funds";
    setDataSourcesTab: (tab: "prices" | "funds") => void;
    config: { priceSource: string; fundSource: string; maxWorkers: number };
    setPriceSource: (source: string) => void;
    setFundSource: (source: string) => void;
    maxPriceDays: number;
    setMaxPriceDays: (days: number) => void;
    selectedSymbols: Set<string>;
    runUpdate: () => void;
    progress: { current: number, total: number, lastMsg: string } | null;
    logs: string[];
    setLogs: React.Dispatch<React.SetStateAction<string[]>>;
    filteredSymbols: SymbolResult[];
    loadingSymbols: boolean;
    toggleSelectAll: () => void;
    symbolsQuery: string;
    setSymbolsQuery: (query: string) => void;
    paginatedSymbols: SymbolResult[];
    toggleSelect: (s: SymbolResult) => void;
    pageSize: number;
    setPageSize: (size: number) => void;
    currentPage: number;
    setCurrentPage: (page: number | ((p: number) => number)) => void;
    totalPages: number;
    setRecalcDialogOpen: (open: boolean) => void;
    recalculatingIndicators: boolean;
    fetchRecentDbFunds: () => void;
    fetchInventory: () => void;
    loadingRecentFunds: boolean;
    dbInventory: any[];
    showEmptyExchanges: boolean;
    setShowEmptyExchanges: (show: boolean) => void;
    setSelectedDbEx: (ex: string | null) => void;
    setDrillDownMode: (mode: "prices" | "fundamentals" | null) => void;
    fetchDbSymbols: (ex: string, mode?: "prices" | "fundamentals") => void;
    setSelectedCountry: (country: string) => void;
    setAutoSelectPending: (ex: string | null) => void;
    setActiveMainTab: (tab: "data" | "ai") => void;
    loadingInventory: boolean;
    setMaxWorkers: (workers: number) => void;
    setConfig: React.Dispatch<React.SetStateAction<{ priceSource: string; fundSource: string; maxWorkers: number }>>;
    updatingInventory?: boolean;
    runInventoryUpdate?: (country?: string) => void;
}

export default function DataManagerTab({
    selectedCountry,
    setCountryDialogOpen,
    processing,
    dataSourcesTab,
    setDataSourcesTab,
    config,
    setPriceSource,
    setFundSource,
    maxPriceDays,
    setMaxPriceDays,
    selectedSymbols,
    runUpdate,
    progress,
    logs,
    setLogs,
    filteredSymbols,
    loadingSymbols,
    toggleSelectAll,
    symbolsQuery,
    setSymbolsQuery,
    paginatedSymbols,
    toggleSelect,
    pageSize,
    setPageSize,
    currentPage,
    setCurrentPage,
    totalPages,
    setRecalcDialogOpen,
    recalculatingIndicators,
    fetchRecentDbFunds,
    fetchInventory,
    loadingRecentFunds,
    dbInventory,
    showEmptyExchanges,
    setShowEmptyExchanges,
    setSelectedDbEx,
    setDrillDownMode,
    fetchDbSymbols,
    setSelectedCountry,
    setAutoSelectPending,
    setActiveMainTab,
    loadingInventory,
    setMaxWorkers,
    setConfig,
    updatingInventory,
    runInventoryUpdate
}: DataManagerTabProps) {
    return (
        <div className="p-8 max-w-7xl mx-auto w-full space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="flex flex-col gap-2">
                <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
                    <Database className="h-8 w-8 text-indigo-500" />
                    Data Manager
                </h1>
                <p className="text-sm text-zinc-500 font-medium">
                    Sync price and fundamental data to Supabase. Manage local symbols inventory.
                </p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Controls */}
                <div className="lg:col-span-1 space-y-4">
                    {/* Country Selector */}
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 space-y-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-medium text-zinc-100">Market Select</h3>
                            <Globe className="h-4 w-4 text-zinc-500" />
                        </div>
                        <button
                            onClick={() => setCountryDialogOpen(true)}
                            disabled={processing}
                            className="w-full h-11 rounded-xl border border-zinc-800 bg-zinc-900 px-4 flex items-center justify-between text-sm text-zinc-300 hover:border-indigo-500 hover:bg-zinc-800/50 transition-all cursor-pointer group disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <span className="font-medium">{selectedCountry}</span>
                            <Globe className="h-4 w-4 text-zinc-500 group-hover:text-indigo-400 transition-colors" />
                        </button>

                        <button
                            onClick={() => runInventoryUpdate?.()}
                            disabled={updatingInventory || processing}
                            className="w-full h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 px-4 flex items-center justify-center gap-2 text-[10px] font-bold text-indigo-400 hover:bg-indigo-600 hover:text-white transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {updatingInventory ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCcw className="h-3 w-3 group-hover:rotate-180 transition-transform duration-500" />}
                            {updatingInventory ? "Updating..." : "Update Global Inventory"}
                        </button>

                        <button
                            onClick={() => runInventoryUpdate?.(selectedCountry)}
                            disabled={updatingInventory || processing}
                            className="w-full h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-4 flex items-center justify-center gap-2 text-[10px] font-bold text-emerald-400 hover:bg-emerald-600 hover:text-white transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {updatingInventory ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCcw className="h-3 w-3 group-hover:rotate-180 transition-transform duration-500" />}
                            {updatingInventory ? "Updating..." : `Update ${selectedCountry} Only`}
                        </button>
                    </div>

                    {/* Data Source Switcher */}
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 space-y-4">
                        <h3 className="text-sm font-medium text-zinc-100">Data Sources</h3>

                        <div className="flex gap-2">
                            <button
                                onClick={() => setDataSourcesTab("prices")}
                                className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${dataSourcesTab === "prices"
                                    ? "bg-indigo-600 text-white border-indigo-600"
                                    : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                    }`}
                            >
                                Prices
                            </button>
                            <button
                                onClick={() => setDataSourcesTab("funds")}
                                className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${dataSourcesTab === "funds"
                                    ? "bg-indigo-600 text-white border-indigo-600"
                                    : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                    }`}
                            >
                                Funds
                            </button>
                        </div>


                        {dataSourcesTab === "prices" ? (
                            <div className="space-y-2">
                                <div className="text-xs text-zinc-500">Prices</div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setPriceSource("eodhd")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${config.priceSource === "eodhd"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        EODHD
                                    </button>
                                    <button
                                        onClick={() => setPriceSource("tradingview")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${config.priceSource === "tradingview"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        TradingView
                                    </button>
                                </div>
                                <div className="space-y-2 pt-2">
                                    <div className="flex flex-col gap-0.5">
                                        <div className="flex justify-between text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
                                            <span>Target History (Days)</span>
                                            <span className="text-indigo-400">{maxPriceDays} days</span>
                                        </div>
                                        <div className="text-[9px] text-zinc-600 leading-tight">
                                            Always updates to today + ensures total history length.
                                        </div>
                                    </div>
                                    <input
                                        type="number"
                                        min={30}
                                        max={5000}
                                        value={maxPriceDays}
                                        onChange={(e) => setMaxPriceDays(Number(e.target.value))}
                                        className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-indigo-500"
                                    />
                                    <div className="p-2 rounded-lg bg-indigo-500/5 border border-indigo-500/10 flex items-center justify-between">
                                        <span className="text-[10px] text-zinc-500 uppercase font-bold">Estimated Rows</span>
                                        <span className="text-[10px] text-indigo-400 font-mono font-bold">~{maxPriceDays * (selectedSymbols.size || 1)} bars</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <div className="text-xs text-zinc-500">Funds</div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setFundSource("tradingview")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${config.fundSource === "tradingview"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        TradingView
                                    </button>
                                    <button
                                        onClick={() => setFundSource("mubasher")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${config.fundSource === "mubasher"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        Mubasher
                                    </button>
                                </div>
                            </div>
                        )}


                        <div className="space-y-4 pt-4 border-t border-zinc-900">
                            <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Index Management</h3>
                            <div className="space-y-2">
                                <input
                                    placeholder="Symbol (e.g. EGX30.INDX)"
                                    className="w-full h-8 rounded-lg border border-zinc-800 bg-zinc-900 px-3 text-xs text-zinc-300 outline-none focus:border-indigo-500"
                                    id="index-symbol"
                                    defaultValue="EGX30.INDX"
                                />
                                <div className="flex gap-2">
                                    <select
                                        className="flex-1 h-8 rounded-lg border border-zinc-800 bg-zinc-900 px-2 text-xs text-zinc-300 outline-none"
                                        id="index-source"
                                        defaultValue="eodhd"
                                        onChange={(e) => {
                                            const exInput = document.getElementById('index-exchange') as HTMLInputElement;
                                            if (exInput) exInput.style.display = e.target.value === 'tradingview' ? 'block' : 'none';
                                        }}
                                    >
                                        <option value="eodhd">EODHD</option>
                                        <option value="tradingview">TradingView</option>
                                    </select>
                                    <input
                                        id="index-exchange"
                                        placeholder="Exchange (e.g. EGX)"
                                        className="flex-1 h-8 rounded-lg border border-zinc-800 bg-zinc-900 px-3 text-xs text-zinc-300 outline-none focus:border-indigo-500 hidden"
                                    />
                                </div>
                                <button
                                    onClick={async () => {
                                        const sym = (document.getElementById('index-symbol') as HTMLInputElement).value;
                                        const src = (document.getElementById('index-source') as HTMLSelectElement).value;
                                        const ex = (document.getElementById('index-exchange') as HTMLInputElement).value;

                                        if (!sym) return;

                                        try {
                                            setLogs(prev => [`[${new Date().toLocaleTimeString()}] Starting Index Sync for ${sym}...`, ...prev]);
                                            const res = await fetch('http://localhost:8000/admin/sync-index', {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    symbol: sym,
                                                    source: src,
                                                    exchange: src === 'tradingview' ? ex : undefined
                                                })
                                            });
                                            const data = await res.json();
                                            setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${data.message}`, ...prev]);
                                        } catch (e) {
                                            setLogs(prev => [`[${new Date().toLocaleTimeString()}] ERR: Index Sync Failed`, ...prev]);
                                        }
                                    }}
                                    className="w-full h-8 rounded-lg bg-zinc-800 border border-zinc-700 text-xs font-bold text-zinc-300 hover:bg-zinc-700 transition-all"
                                >
                                    Sync Index
                                </button>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <div className="text-xs text-zinc-500">Workers</div>
                            <input
                                type="number"
                                min={1}
                                max={64}
                                value={config.maxWorkers}
                                onChange={(e) => setConfig((prev) => ({ ...prev, maxWorkers: Number(e.target.value) }))}
                                onBlur={() => setMaxWorkers(config.maxWorkers)}
                                disabled={processing}
                                className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300"
                            />
                        </div>

                        <div className="space-y-4 pt-4">
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-zinc-400">Selected Symbols</span>
                                <span className="font-mono text-indigo-400">{selectedSymbols.size}</span>
                            </div>

                            <button
                                onClick={runUpdate}
                                disabled={processing || selectedSymbols.size === 0}
                                className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-3 text-sm font-bold text-white hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                {processing ? <Loader2 className="h-5 w-5 animate-spin" /> : <Download className="h-5 w-5" />}
                                {processing ? "Updating..." : "Update Cloud"}
                            </button>

                        </div>


                        {progress && processing && (
                            <div className="space-y-2 pt-2">
                                <div className="flex justify-between text-xs text-zinc-500">
                                    <span>Progress</span>
                                    <span>{Math.round((progress.current / progress.total) * 100)}%</span>
                                </div>
                                <div className="h-2 w-full rounded-full bg-zinc-900 overflow-hidden">
                                    <div
                                        className="h-full bg-indigo-500 transition-all duration-300"
                                        style={{ width: `${(progress.current / progress.total) * 100}%` }}
                                    />
                                </div>
                                <div className="text-xs text-zinc-600 truncate font-mono">
                                    {progress.lastMsg}
                                </div>
                            </div>
                        )}

                        {logs.length > 0 && (
                            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4 space-y-3">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Sync Operations Log</h3>
                                    <button
                                        onClick={() => setLogs([])}
                                        className="text-[10px] font-bold text-zinc-600 hover:text-red-400 transition-colors uppercase tracking-widest"
                                    >
                                        Clear
                                    </button>
                                </div>
                                <div className="space-y-1 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                                    {logs.map((log, i) => {
                                        const isErr = log.includes(": ERR");
                                        const isOk = log.includes(": OK");
                                        return (
                                            <div
                                                key={i}
                                                className={`text-[10px] font-mono break-all border-l-2 pl-2 py-1.5 leading-relaxed transition-all rounded-r-md ${isErr ? "bg-orange-500/10 text-orange-400 border-orange-500/50" :
                                                    isOk ? "bg-green-500/10 text-green-400 border-green-500/50" :
                                                        "text-zinc-400 border-zinc-800 hover:bg-white/5"
                                                    }`}
                                            >
                                                {log}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right: Symbol List */}
                <div className="lg:col-span-2 rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col h-[600px]">
                    <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/50">
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-zinc-200">Available Symbols</span>
                            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500">{filteredSymbols.length}</span>
                        </div>

                        <button
                            onClick={toggleSelectAll}
                            className="text-xs font-medium text-indigo-400 hover:text-indigo-300"
                        >
                            {(() => {
                                const filteredIds = filteredSymbols.map(s => s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol);
                                const allFilteredSelected = filteredIds.length > 0 && filteredIds.every((id) => selectedSymbols.has(id));
                                return allFilteredSelected ? "Deselect All" : "Select All";
                            })()}
                        </button>
                    </div>

                    <div className="px-6 py-3 border-b border-zinc-800">
                        <input
                            value={symbolsQuery}
                            onChange={(e) => setSymbolsQuery(e.target.value)}
                            disabled={processing}
                            placeholder="Search symbols..."
                            className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-zinc-500"
                        />
                    </div>

                    <div className="flex-1 overflow-y-auto p-2">
                        {loadingSymbols ? (
                            <div className="flex h-full items-center justify-center text-zinc-500 gap-2">
                                <Loader2 className="h-5 w-5 animate-spin" />
                                Loading symbols...
                            </div>
                        ) : filteredSymbols.length === 0 ? (
                            <div className="flex h-full items-center justify-center text-zinc-500">
                                No symbols found
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                                {paginatedSymbols.map(s => {
                                    const id = s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol;
                                    const isSelected = selectedSymbols.has(id);
                                    return (
                                        <div
                                            key={id}
                                            onClick={() => !processing && toggleSelect(s)}
                                            className={`
                                                cursor-pointer rounded-lg border px-3 py-2 transition-all relative
                                                ${isSelected
                                                    ? "border-indigo-500 bg-indigo-500/10 text-indigo-100"
                                                    : "border-zinc-800 bg-zinc-900/30 text-zinc-400 hover:bg-zinc-900 hover:border-zinc-700"
                                                }
                                            `}
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="font-mono font-bold text-sm">{s.symbol}</span>
                                                {isSelected && <Check className="h-3 w-3 text-indigo-400" />}
                                            </div>
                                            <div className="text-xs opacity-60 truncate mt-1">{s.name}</div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Pagination Controls */}
                    {!loadingSymbols && filteredSymbols.length > 0 && (
                        <div className="px-6 py-4 border-t border-zinc-800 bg-zinc-900/30 flex flex-wrap items-center justify-between gap-4">
                            <div className="flex items-center gap-3">
                                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Page Size</span>
                                <select
                                    value={pageSize}
                                    onChange={(e) => setPageSize(Number(e.target.value))}
                                    className="bg-zinc-900 border border-zinc-700 text-zinc-300 text-xs rounded-lg px-2 py-1 outline-none focus:border-indigo-500"
                                >
                                    {[50, 100, 200, 500, 1000].map(size => (
                                        <option key={size} value={size}>{size}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="flex items-center gap-4">
                                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                    Page {currentPage} of {totalPages || 1}
                                </span>
                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                        disabled={currentPage === 1}
                                        className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 transition-all"
                                    >
                                        <ChevronLeft className="h-4 w-4" />
                                    </button>
                                    <button
                                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                        disabled={currentPage === totalPages}
                                        className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 transition-all"
                                    >
                                        <ChevronRight className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Bottom Section: Modern Fundamentals Dashboard */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/50 backdrop-blur-sm overflow-hidden flex flex-col min-h-[400px]">
                        <div className="px-8 py-6 border-b border-zinc-800 bg-gradient-to-r from-zinc-900/50 to-indigo-900/10 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                    <BarChart3 className="w-6 h-6" />
                                </div>
                                <div>
                                    <h2 className="text-xl font-bold text-zinc-100">Data Center</h2>
                                    <p className="text-xs text-zinc-500">Global market fundamental insights from Cloud</p>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setRecalcDialogOpen(true)}
                                    disabled={recalculatingIndicators}
                                    className="p-2 px-4 rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-xs font-bold text-indigo-400 hover:bg-indigo-600 hover:text-white transition-all flex items-center gap-2 disabled:opacity-50"
                                >
                                    {recalculatingIndicators ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                                    {recalculatingIndicators ? "Recalculating..." : "Recalculate Technicals"}
                                </button>
                                <button
                                    onClick={() => { fetchRecentDbFunds(); fetchInventory(); }}
                                    className="p-2 px-4 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-bold text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-all flex items-center gap-2"
                                >
                                    <History className="w-4 h-4" />
                                    {loadingRecentFunds ? "Fetching..." : "Refresh All"}
                                </button>
                            </div>
                        </div>

                        <div className="flex-1 p-8 space-y-12">
                            {/* Inventory Summary Row */}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                                        <Database className="w-4 h-4 text-indigo-400" />
                                        Database Inventory Summary
                                    </h3>
                                    <div className="flex items-center gap-4">
                                        <button
                                            onClick={() => setShowEmptyExchanges(!showEmptyExchanges)}
                                            className={`text-[9px] font-bold px-2 py-1 rounded border transition-all ${showEmptyExchanges ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400' : 'bg-zinc-900 border-zinc-800 text-zinc-600 hover:text-zinc-400'}`}
                                        >
                                            {showEmptyExchanges ? 'HIDE EMPTY' : 'SHOW ALL EXCHANGES'}
                                        </button>
                                        <p className="text-[10px] text-zinc-600 font-medium whitespace-nowrap">Click an exchange to view details</p>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                                    {dbInventory
                                        .filter(item => showEmptyExchanges || item.priceCount > 0)
                                        .map((item) => (
                                            <div
                                                key={item.exchange}
                                                onClick={(e) => {
                                                    setSelectedDbEx(item.exchange);
                                                    setDrillDownMode('prices');
                                                    fetchDbSymbols(item.exchange, 'prices');
                                                    if (item.country && item.country !== 'N/A') {
                                                        setSelectedCountry(item.country);
                                                    }
                                                }}
                                                className={`p-4 rounded-2xl border transition-all cursor-pointer group flex flex-col h-full bg-zinc-900/30 border-zinc-800/50 hover:border-zinc-700/50`}
                                            >
                                                <div className="flex justify-between items-center text-[10px] font-bold text-zinc-500 uppercase group-hover:text-zinc-300">
                                                    {item.exchange} {item.country !== 'N/A' ? `(${item.country})` : ''}
                                                    <div className={`w-1.5 h-1.5 rounded-full ${item.status === 'healthy' ? 'bg-green-500' : 'bg-zinc-700'}`} />
                                                </div>
                                                <div className="flex items-baseline gap-1 mt-1">
                                                    <div className="text-2xl font-black text-zinc-100">{item.priceCount}</div>
                                                    <div className="text-xs text-zinc-600 font-bold">/ {item.expectedCount || item.priceCount}</div>
                                                </div>
                                                <div className="flex items-center justify-between mt-1 min-h-[16px]">
                                                    {item.expectedCount > item.priceCount && (
                                                        <span className="text-[8px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 font-bold">
                                                            {item.expectedCount - item.priceCount} Missing
                                                        </span>
                                                    )}
                                                </div>
                                                {item.expectedCount > item.priceCount && (
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setDrillDownMode('prices');
                                                            fetchDbSymbols(item.exchange, 'prices');
                                                            setSelectedCountry(item.country);
                                                            setAutoSelectPending(item.exchange);
                                                            setDataSourcesTab('prices');
                                                            setActiveMainTab("data");
                                                        }}
                                                        className="mt-auto w-full py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-[9px] font-bold text-indigo-400 hover:bg-indigo-600 hover:text-white transition-all flex items-center justify-center gap-2"
                                                    >
                                                        <Zap className="w-3 h-3" />
                                                        SELECT MISSINGS
                                                    </button>
                                                )}
                                            </div>
                                        ))}
                                </div>
                            </div>


                            <div className="space-y-8">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-[0.2em] flex items-center gap-2">
                                        <Cloud className="w-4 h-4 text-emerald-400" />
                                        Fundamentals Database
                                    </h3>
                                    <p className="text-[10px] text-zinc-600 font-medium">Synced fundamentals per exchange</p>
                                </div>

                                {loadingInventory ? (
                                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                                        {[1, 2, 3, 4, 5, 6].map(i => (
                                            <div key={i} className="h-24 rounded-2xl bg-zinc-900/50 border border-zinc-800 animate-pulse" />
                                        ))}
                                    </div>
                                ) : dbInventory.length === 0 ? (
                                    <div className="py-20 text-center border-2 border-dashed border-zinc-900 rounded-3xl">
                                        <Database className="w-12 h-12 text-zinc-800 mx-auto mb-4" />
                                        <p className="text-zinc-600 text-sm">No fundamental data found in Supabase.</p>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                                        {dbInventory
                                            .filter(item => showEmptyExchanges || item.fundCount > 0)
                                            .map((item) => (
                                                <div
                                                    key={`fund-${item.exchange}`}
                                                    onClick={(e) => {
                                                        setSelectedDbEx(item.exchange);
                                                        setDrillDownMode('fundamentals');
                                                        fetchDbSymbols(item.exchange, 'fundamentals');
                                                        if (item.country && item.country !== 'N/A') {
                                                            setSelectedCountry(item.country);
                                                        }
                                                    }}
                                                    className={`p-4 rounded-2xl border transition-all cursor-pointer group flex flex-col h-full hover:border-emerald-500/30 hover:bg-emerald-500/5 bg-zinc-900/30 border-zinc-800/50`}
                                                >
                                                    <div className="flex justify-between items-center text-[10px] font-bold text-zinc-500 uppercase group-hover:text-zinc-300">
                                                        {item.exchange} {item.country !== 'N/A' ? `(${item.country})` : ''}
                                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                                    </div>
                                                    <div className="flex items-baseline gap-1 mt-1">
                                                        <div className="text-2xl font-black text-zinc-100">{item.fundCount}</div>
                                                        <div className="text-xs text-zinc-600 font-bold">/ {item.expectedCount || item.fundCount}</div>
                                                    </div>
                                                    <div className="flex items-center justify-between mt-1 min-h-[16px]">
                                                        {item.expectedCount > item.fundCount && (
                                                            <span className="text-[8px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 font-bold">
                                                                {item.expectedCount - item.fundCount} Missing
                                                            </span>
                                                        )}
                                                    </div>
                                                    {item.expectedCount > item.fundCount && (
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setDrillDownMode('fundamentals');
                                                                fetchDbSymbols(item.exchange, 'fundamentals');
                                                                setSelectedCountry(item.country);
                                                                setAutoSelectPending(item.exchange);
                                                                setDataSourcesTab('funds');
                                                                setActiveMainTab("data");
                                                            }}
                                                            className="mt-auto w-full py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-[9px] font-bold text-emerald-400 hover:bg-emerald-600 hover:text-white transition-all flex items-center justify-center gap-2"
                                                        >
                                                            <Zap className="w-3 h-3" />
                                                            SELECT MISSINGS
                                                        </button>
                                                    )}
                                                </div>
                                            ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
