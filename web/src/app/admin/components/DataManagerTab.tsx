"use client";

import { Database, Globe, Loader2, Download, Check, ChevronLeft, ChevronRight, BarChart3, History, Zap, Cloud, RefreshCcw } from "lucide-react";
import {
    type SymbolResult,
    getAlpacaSupabaseStats,
    getAlpacaAssets,
    syncAlpacaPrices,
    getCryptoSymbolStats,
    deleteCryptoBars,
    type AlpacaAsset,
    type AlpacaSupabaseStats,
    type CryptoSymbolStat,
} from "@/lib/api";
import { useEffect, useState } from "react";

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
    const [cryptoStats, setCryptoStats] = useState<AlpacaSupabaseStats | null>(null);
    const [marketMode, setMarketMode] = useState<"global" | "crypto">("global");
    const [cryptoAssets, setCryptoAssets] = useState<AlpacaAsset[]>([]);
    const [cryptoQuery, setCryptoQuery] = useState("");
    const [selectedCryptoSymbols, setSelectedCryptoSymbols] = useState<Set<string>>(new Set());
    const [cryptoTimeframe, setCryptoTimeframe] = useState<"1m" | "1h" | "1d">("1h");
    const [cryptoSyncing, setCryptoSyncing] = useState(false);
    const [loadingCrypto, setLoadingCrypto] = useState(false);
    const isCryptoMode = marketMode === "crypto";
    const [cryptoDialogOpen, setCryptoDialogOpen] = useState(false);
    const [cryptoDialogLoading, setCryptoDialogLoading] = useState(false);
    const [cryptoDialogRows, setCryptoDialogRows] = useState<CryptoSymbolStat[]>([]);
    const [cryptoDialogQuery, setCryptoDialogQuery] = useState("");
    const [cryptoDialogTimeframe, setCryptoDialogTimeframe] = useState<"1m" | "1h" | "1d">("1h");
    const [cryptoDialogSelected, setCryptoDialogSelected] = useState<Set<string>>(new Set());
    const [cryptoDialogSort, setCryptoDialogSort] = useState<{
        key: "symbol" | "bars" | "start" | "end" | "days";
        dir: "asc" | "desc";
    }>({ key: "bars", dir: "desc" });

    const fetchCryptoSupabaseStats = async () => {
        try {
            const sb = await getAlpacaSupabaseStats("crypto");
            setCryptoStats(sb);
        } catch {
            // ignore (Supabase may not be configured)
        }
    };

    useEffect(() => {
        fetchCryptoSupabaseStats();
    }, []);

    useEffect(() => {
        if (!cryptoDialogOpen) return;
        setCryptoDialogLoading(true);
        getCryptoSymbolStats(cryptoDialogTimeframe)
            .then(setCryptoDialogRows)
            .catch(() => setCryptoDialogRows([]))
            .finally(() => setCryptoDialogLoading(false));
    }, [cryptoDialogOpen, cryptoDialogTimeframe]);

    useEffect(() => {
        setCurrentPage(1);
        if (marketMode !== "crypto") return;
        refreshCryptoAssets();
    }, [marketMode, setCurrentPage]);

    const refreshCryptoAssets = () => {
        setLoadingCrypto(true);
        return getAlpacaAssets(undefined, "crypto")
            .then(setCryptoAssets)
            .catch(() => setCryptoAssets([]))
            .finally(() => setLoadingCrypto(false));
    };

    const toggleCryptoSelect = (symbol: string) => {
        const next = new Set(selectedCryptoSymbols);
        if (next.has(symbol)) next.delete(symbol);
        else next.add(symbol);
        setSelectedCryptoSymbols(next);
    };

    const filteredCryptoAssets = cryptoAssets.filter(a =>
        a.symbol.toLowerCase().includes(cryptoQuery.toLowerCase()) ||
        (a.name || "").toLowerCase().includes(cryptoQuery.toLowerCase())
    );

    const cryptoTotalPages = Math.max(1, Math.ceil(filteredCryptoAssets.length / pageSize));
    const cryptoPageItems = filteredCryptoAssets.slice((currentPage - 1) * pageSize, currentPage * pageSize);

    const toggleSelectAllCrypto = () => {
        const allSelected = filteredCryptoAssets.length > 0 && filteredCryptoAssets.every(a => selectedCryptoSymbols.has(a.symbol));
        const next = new Set(selectedCryptoSymbols);
        if (allSelected) {
            filteredCryptoAssets.forEach(a => next.delete(a.symbol));
        } else {
            filteredCryptoAssets.forEach(a => next.add(a.symbol));
        }
        setSelectedCryptoSymbols(next);
    };

    const handleUpdateCryptoPrices = async () => {
        setCryptoSyncing(true);
        try {
            const symbols = selectedCryptoSymbols.size > 0
                ? Array.from(selectedCryptoSymbols)
                : filteredCryptoAssets.map(a => a.symbol);
            const res = await syncAlpacaPrices(symbols, {
                assetClass: "crypto",
                exchange: "BINANCE",
                days: maxPriceDays,
                source: "binance",
                timeframe: cryptoTimeframe,
            });
            setLogs(prev => [
                `[${new Date().toLocaleTimeString()}] Crypto update: ${res.rows_upserted} bars (${res.timeframe || cryptoTimeframe}, ${res.days}d)`,
                ...prev
            ]);
            fetchCryptoSupabaseStats();
        } catch (e: any) {
            setLogs(prev => [`[${new Date().toLocaleTimeString()}] ERR: Crypto update`, ...prev]);
        } finally {
            setCryptoSyncing(false);
        }
    };

    const filteredCryptoDialogRows = cryptoDialogRows.filter(r =>
        r.symbol.toLowerCase().includes(cryptoDialogQuery.toLowerCase())
    );
    const cryptoDialogRowDays = (row: CryptoSymbolStat) => {
        if (!row.first_ts || !row.last_ts) return 0;
        const start = new Date(row.first_ts).getTime();
        const end = new Date(row.last_ts).getTime();
        if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return 0;
        const diffDays = (end - start) / (1000 * 60 * 60 * 24);
        return Math.max(1, Math.round(diffDays + 1));
    };
    const sortedCryptoDialogRows = [...filteredCryptoDialogRows].sort((a, b) => {
        const dir = cryptoDialogSort.dir === "asc" ? 1 : -1;
        if (cryptoDialogSort.key === "symbol") {
            return a.symbol.localeCompare(b.symbol) * dir;
        }
        if (cryptoDialogSort.key === "bars") {
            return (a.rows_count - b.rows_count) * dir;
        }
        if (cryptoDialogSort.key === "days") {
            return (cryptoDialogRowDays(a) - cryptoDialogRowDays(b)) * dir;
        }
        if (cryptoDialogSort.key === "start") {
            const av = a.first_ts ? new Date(a.first_ts).getTime() : 0;
            const bv = b.first_ts ? new Date(b.first_ts).getTime() : 0;
            return (av - bv) * dir;
        }
        const av = a.last_ts ? new Date(a.last_ts).getTime() : 0;
        const bv = b.last_ts ? new Date(b.last_ts).getTime() : 0;
        return (av - bv) * dir;
    });

    const toggleCryptoDialogSelect = (symbol: string) => {
        const next = new Set(cryptoDialogSelected);
        if (next.has(symbol)) next.delete(symbol);
        else next.add(symbol);
        setCryptoDialogSelected(next);
    };
    const toggleCryptoDialogSelectAll = () => {
        const allSelected = sortedCryptoDialogRows.length > 0 && sortedCryptoDialogRows.every(r => cryptoDialogSelected.has(r.symbol));
        const next = new Set(cryptoDialogSelected);
        if (allSelected) {
            sortedCryptoDialogRows.forEach(r => next.delete(r.symbol));
        } else {
            sortedCryptoDialogRows.forEach(r => next.add(r.symbol));
        }
        setCryptoDialogSelected(next);
    };
    const handleCryptoDialogDelete = async () => {
        if (cryptoDialogSelected.size === 0) return;
        if (!confirm(`Delete ${cryptoDialogSelected.size} symbols from ${cryptoDialogTimeframe}?`)) return;
        try {
            await deleteCryptoBars(Array.from(cryptoDialogSelected), cryptoDialogTimeframe);
            setCryptoDialogSelected(new Set());
            setCryptoDialogLoading(true);
            const rows = await getCryptoSymbolStats(cryptoDialogTimeframe);
            setCryptoDialogRows(rows);
            fetchCryptoSupabaseStats();
        } catch {
            // ignore
        } finally {
            setCryptoDialogLoading(false);
        }
    };
    return (
        <div className="p-4 md:p-8 max-w-full mx-auto w-full space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 overflow-x-hidden">
            <header className="flex flex-col gap-2 max-w-full">
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
                            <h3 className="text-sm font-medium text-zinc-100 flex gap-2">
                                <span>Market Select</span>
                            </h3>
                            <Globe className="h-4 w-4 text-zinc-500" />
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setMarketMode("global")}
                                className={`flex-1 py-2 text-[10px] font-bold rounded-lg border transition-all ${marketMode === "global"
                                    ? "bg-indigo-600 text-white border-indigo-600 shadow-[0_4px_12px_-4px_rgba(79,70,229,0.5)]"
                                    : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                    }`}
                            >
                                <div className="flex flex-col">
                                    <span className="text-[10px] font-black uppercase tracking-wider">GLOBAL</span>
                                </div>
                            </button>
                            <button
                                onClick={() => setMarketMode("crypto")}
                                className={`flex-1 py-2 text-[10px] font-bold rounded-lg border transition-all ${marketMode === "crypto"
                                    ? "bg-indigo-600 text-white border-indigo-600 shadow-[0_4px_12px_-4px_rgba(79,70,229,0.5)]"
                                    : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                    }`}
                            >
                                <div className="flex flex-col">
                                    <span className="text-[10px] font-black uppercase tracking-wider">CRYPTO</span>
                                </div>
                            </button>
                        </div>

                        {marketMode === "global" ? (
                            <>
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
                            </>
                        ) : (
                            <>
                                <button
                                    onClick={refreshCryptoAssets}
                                    disabled={loadingCrypto}
                                    className="w-full h-10 rounded-xl bg-zinc-900 border border-zinc-800 px-4 flex items-center justify-center gap-2 text-[10px] font-bold text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-50"
                                >
                                    {loadingCrypto ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
                                    Refresh Crypto Symbols
                                </button>

                                <div className="text-[10px] text-zinc-500">
                                    Symbols loaded: <span className="text-zinc-300 font-bold">{cryptoAssets.length}</span>
                                </div>

                                <div className="space-y-2 pt-1">
                                    <div className="flex justify-between text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
                                        <span>Target History (Days)</span>
                                        <span className="text-indigo-400">{maxPriceDays} days</span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2">
                                        {(["1m", "1h", "1d"] as const).map((tf) => (
                                            <button
                                                key={tf}
                                                onClick={() => setCryptoTimeframe(tf)}
                                                disabled={cryptoSyncing}
                                                className={`h-9 rounded-xl border text-[10px] font-bold transition-all ${cryptoTimeframe === tf
                                                    ? "bg-indigo-600 text-white border-indigo-600"
                                                    : "bg-zinc-900 text-zinc-500 border-zinc-800 hover:bg-zinc-800 hover:text-zinc-300"
                                                    }`}
                                            >
                                                {tf.toUpperCase()}
                                            </button>
                                        ))}
                                    </div>
                                    <input
                                        type="number"
                                        min={5}
                                        max={5000}
                                        value={maxPriceDays}
                                        onChange={(e) => setMaxPriceDays(Number(e.target.value))}
                                        className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-indigo-500"
                                    />
                                    <button
                                        onClick={handleUpdateCryptoPrices}
                                        disabled={cryptoSyncing}
                                        className="w-full h-10 rounded-xl bg-indigo-600 px-4 flex items-center justify-center gap-2 text-[10px] font-bold text-white hover:bg-indigo-500 transition-all disabled:opacity-50"
                                    >
                                        {cryptoSyncing ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
                                        Update Crypto Prices {selectedCryptoSymbols.size > 0 ? `(Selected: ${selectedCryptoSymbols.size})` : "(All Filtered)"}
                                    </button>
                                </div>
                            </>
                        )}
                    </div>

                    {/* Data Source Switcher */}
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 space-y-4">
                        <h3 className="text-sm font-medium text-zinc-100 flex gap-2">
                            <span>Data Sources</span>
                        </h3>

                        {marketMode === "global" && (
                            <>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setDataSourcesTab("prices")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${dataSourcesTab === "prices"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        <div className="flex flex-col">
                                            <span className="text-[10px] font-black uppercase tracking-wider">Prices</span>
                                        </div>
                                    </button>
                                    <button
                                        onClick={() => setDataSourcesTab("funds")}
                                        className={`flex-1 py-2 text-xs font-bold rounded-lg border transition-all ${dataSourcesTab === "funds"
                                            ? "bg-indigo-600 text-white border-indigo-600"
                                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:bg-zinc-800"
                                            }`}
                                    >
                                        <div className="flex flex-col">
                                            <span className="text-[10px] font-black uppercase tracking-wider">Funds</span>
                                        </div>
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
                            </>
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
                            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500">
                                {isCryptoMode ? filteredCryptoAssets.length : filteredSymbols.length}
                            </span>
                        </div>

                        <button
                            onClick={isCryptoMode ? toggleSelectAllCrypto : toggleSelectAll}
                            className="text-xs font-medium text-indigo-400 hover:text-indigo-300"
                        >
                            {isCryptoMode
                                ? (() => {
                                    const allFilteredSelected = filteredCryptoAssets.length > 0 && filteredCryptoAssets.every((a) => selectedCryptoSymbols.has(a.symbol));
                                    return allFilteredSelected ? "Deselect All" : "Select All";
                                })()
                                : (() => {
                                    const filteredIds = filteredSymbols.map(s => s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol);
                                    const allFilteredSelected = filteredIds.length > 0 && filteredIds.every((id) => selectedSymbols.has(id));
                                    return allFilteredSelected ? "Deselect All" : "Select All";
                                })()
                            }
                        </button>
                    </div>

                    <div className="px-6 py-3 border-b border-zinc-800">
                        <input
                            value={isCryptoMode ? cryptoQuery : symbolsQuery}
                            onChange={(e) => isCryptoMode ? setCryptoQuery(e.target.value) : setSymbolsQuery(e.target.value)}
                            disabled={processing || (isCryptoMode && cryptoSyncing)}
                            placeholder="Search symbols..."
                            className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-zinc-500"
                        />
                    </div>

                    <div className="flex-1 overflow-y-auto p-2">
                        {isCryptoMode ? (
                            loadingCrypto ? (
                                <div className="flex h-full items-center justify-center text-zinc-500 gap-2">
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                    Loading symbols...
                                </div>
                            ) : filteredCryptoAssets.length === 0 ? (
                                <div className="flex h-full items-center justify-center text-zinc-500">
                                    No symbols found
                                </div>
                            ) : (
                                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                                    {cryptoPageItems.map(a => {
                                        const isSelected = selectedCryptoSymbols.has(a.symbol);
                                        return (
                                            <div
                                                key={a.symbol}
                                                onClick={() => !cryptoSyncing && toggleCryptoSelect(a.symbol)}
                                                className={`
                                                cursor-pointer rounded-lg border px-3 py-2 transition-all relative
                                                ${isSelected
                                                        ? "border-indigo-500 bg-indigo-500/10 text-indigo-100"
                                                        : "border-zinc-800 bg-zinc-900/30 text-zinc-400 hover:bg-zinc-900 hover:border-zinc-700"
                                                    }
                                            `}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <span className="font-mono font-bold text-sm">{a.symbol}</span>
                                                    {isSelected && <Check className="h-3 w-3 text-indigo-400" />}
                                                </div>
                                                <div className="text-xs opacity-60 truncate mt-1">{a.name}</div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )
                        ) : loadingSymbols ? (
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
                    {(!isCryptoMode && !loadingSymbols && filteredSymbols.length > 0) || (isCryptoMode && !loadingCrypto && filteredCryptoAssets.length > 0) ? (
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
                                    Page {currentPage} of {isCryptoMode ? cryptoTotalPages : (totalPages || 1)}
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
                                        onClick={() => setCurrentPage(p => Math.min(isCryptoMode ? cryptoTotalPages : totalPages, p + 1))}
                                        disabled={currentPage === (isCryptoMode ? cryptoTotalPages : totalPages)}
                                        className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 transition-all"
                                    >
                                        <ChevronRight className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : null}
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
                                    onClick={() => { fetchRecentDbFunds(); fetchInventory(); fetchCryptoSupabaseStats(); }}
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


                            {/* Crypto Section */}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                                        <Globe className="w-4 h-4 text-indigo-400" />
                                        Crypto Database Summary
                                    </h3>
                                    <p className="text-[10px] text-zinc-600 font-medium">Supabase crypto inventory</p>
                                </div>

                                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                                    <div
                                        onClick={() => setCryptoDialogOpen(true)}
                                        className="p-4 rounded-2xl border transition-all cursor-pointer flex flex-col h-full bg-zinc-900/30 border-zinc-800/50 hover:border-indigo-500/40 hover:bg-indigo-500/5"
                                    >
                                        <div className="flex justify-between items-center text-[10px] font-bold text-zinc-500 uppercase">
                                            CRYPTO
                                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                        </div>
                                        <div className="flex items-baseline gap-1 mt-1">
                                            <div className="text-2xl font-black text-zinc-100">{cryptoStats?.stock_bars_intraday?.rows ?? 0}</div>
                                            <div className="text-xs text-zinc-600 font-bold">bars</div>
                                        </div>
                                        <div className="mt-2 text-[10px] font-mono text-zinc-500 truncate">
                                            1m {cryptoStats?.stock_bars_intraday?.by_timeframe?.["1m"] ?? 0} | 1h {cryptoStats?.stock_bars_intraday?.by_timeframe?.["1h"] ?? 0} | 1d {cryptoStats?.stock_bars_intraday?.by_timeframe?.["1d"] ?? 0}
                                        </div>
                                        <div className="mt-2 text-[10px] text-zinc-600">
                                            Daily: {cryptoStats?.stock_prices?.rows ?? 0} | Last: {cryptoStats?.stock_prices?.last_date ?? "n/a"}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {cryptoDialogOpen && (
                                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6">
                                    <div className="w-full max-w-4xl rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl">
                                        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                                <div>
                                                    <div className="text-sm font-bold text-zinc-100">Crypto Symbols Detail</div>
                                                    <div className="text-[10px] text-zinc-500">Counts and date ranges by symbol</div>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => setCryptoDialogOpen(false)}
                                                className="text-xs font-bold text-zinc-500 hover:text-white"
                                            >
                                                Close
                                            </button>
                                        </div>

                                        <div className="px-6 py-4 flex flex-col gap-3">
                                            <div className="flex flex-wrap items-center gap-3">
                                                <input
                                                    value={cryptoDialogQuery}
                                                    onChange={(e) => setCryptoDialogQuery(e.target.value)}
                                                    placeholder="Search symbol..."
                                                    className="flex-1 min-w-[200px] h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-indigo-500"
                                                />
                                                <div className="flex gap-2">
                                                    {(["1m", "1h", "1d"] as const).map(tf => (
                                                        <button
                                                            key={tf}
                                                            onClick={() => setCryptoDialogTimeframe(tf)}
                                                            className={`h-10 px-3 rounded-lg border text-xs font-bold transition-all ${cryptoDialogTimeframe === tf ? "bg-indigo-600 text-white border-indigo-600" : "bg-zinc-900 text-zinc-500 border-zinc-800 hover:bg-zinc-800"}`}
                                                        >
                                                            {tf.toUpperCase()}
                                                        </button>
                                                    ))}
                                                </div>
                                                <div className="text-[10px] text-zinc-500">
                                                    {cryptoDialogLoading ? "Loading..." : `${filteredCryptoDialogRows.length} symbols`}
                                                </div>
                                                <button
                                                    onClick={toggleCryptoDialogSelectAll}
                                                    className="text-xs font-bold text-indigo-400 hover:text-indigo-300"
                                                >
                                                    {sortedCryptoDialogRows.length > 0 && sortedCryptoDialogRows.every(r => cryptoDialogSelected.has(r.symbol)) ? "Deselect All" : "Select All"}
                                                </button>
                                                <button
                                                    onClick={handleCryptoDialogDelete}
                                                    disabled={cryptoDialogSelected.size === 0}
                                                    className="text-xs font-bold text-red-400 hover:text-red-300 disabled:opacity-40"
                                                >
                                                    Delete Selected
                                                </button>
                                            </div>

                                            <div className="max-h-[420px] overflow-y-auto border border-zinc-900 rounded-xl">
                                                <div className="grid grid-cols-6 gap-2 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-zinc-500 bg-zinc-900/50 border-b border-zinc-800">
                                                    <div className="flex items-center gap-2">
                                                        <input
                                                            type="checkbox"
                                                            checked={sortedCryptoDialogRows.length > 0 && sortedCryptoDialogRows.every(r => cryptoDialogSelected.has(r.symbol))}
                                                            onChange={toggleCryptoDialogSelectAll}
                                                        />
                                                        <button
                                                            onClick={() => setCryptoDialogSort(s => ({ key: "symbol", dir: s.key === "symbol" && s.dir === "asc" ? "desc" : "asc" }))}
                                                            className="hover:text-zinc-200"
                                                        >
                                                            Symbol
                                                        </button>
                                                    </div>
                                                    <button
                                                        onClick={() => setCryptoDialogSort(s => ({ key: "bars", dir: s.key === "bars" && s.dir === "asc" ? "desc" : "asc" }))}
                                                        className="hover:text-zinc-200"
                                                    >
                                                        Bars
                                                    </button>
                                                    <button
                                                        onClick={() => setCryptoDialogSort(s => ({ key: "days", dir: s.key === "days" && s.dir === "asc" ? "desc" : "asc" }))}
                                                        className="hover:text-zinc-200"
                                                    >
                                                        Days
                                                    </button>
                                                    <button
                                                        onClick={() => setCryptoDialogSort(s => ({ key: "start", dir: s.key === "start" && s.dir === "asc" ? "desc" : "asc" }))}
                                                        className="hover:text-zinc-200"
                                                    >
                                                        Start
                                                    </button>
                                                    <button
                                                        onClick={() => setCryptoDialogSort(s => ({ key: "end", dir: s.key === "end" && s.dir === "asc" ? "desc" : "asc" }))}
                                                        className="hover:text-zinc-200"
                                                    >
                                                        End
                                                    </button>
                                                    <div className="text-right pr-2">Select</div>
                                                </div>
                                                {cryptoDialogLoading ? (
                                                    <div className="p-6 text-center text-zinc-500 text-sm">Loading...</div>
                                                ) : filteredCryptoDialogRows.length === 0 ? (
                                                    <div className="p-6 text-center text-zinc-500 text-sm">No data</div>
                                                ) : (
                                                    sortedCryptoDialogRows.map((row) => (
                                                        <div key={row.symbol} className="grid grid-cols-6 gap-2 px-4 py-2 text-xs text-zinc-300 border-b border-zinc-900">
                                                            <div className="font-mono font-bold">{row.symbol}</div>
                                                            <div className="text-zinc-400">{row.rows_count.toLocaleString()}</div>
                                                            <div className="text-zinc-400">{cryptoDialogRowDays(row)}</div>
                                                            <div className="text-zinc-500">{row.first_ts ? new Date(row.first_ts).toLocaleString() : "n/a"}</div>
                                                            <div className="text-zinc-500">{row.last_ts ? new Date(row.last_ts).toLocaleString() : "n/a"}</div>
                                                            <div className="text-right pr-2">
                                                                <input
                                                                    type="checkbox"
                                                                    checked={cryptoDialogSelected.has(row.symbol)}
                                                                    onChange={() => toggleCryptoDialogSelect(row.symbol)}
                                                                />
                                                            </div>
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}


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
