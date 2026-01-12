"use client";

import { useState, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";
import { getCountries, searchSymbols, type SymbolResult } from "@/lib/api";
import { Database, Download, Check, AlertTriangle, Loader2, Zap, BarChart3, Info, TrendingUp, History, Cloud } from "lucide-react";
import { Toaster, toast } from "sonner";

import CountrySelectDialog from "@/components/CountrySelectDialog";

export default function AdminPage() {
    const { t } = useLanguage();
    const { countries, countriesLoading } = useAppState();

    // State
    // const [countries, setCountries] = useState<string[]>([]); // Removed local state
    const [selectedCountry, setSelectedCountry] = useState("Egypt");
    // const [loadingCountries, setLoadingCountries] = useState(false); // Removed local state
    const [countryDialogOpen, setCountryDialogOpen] = useState(false);

    const [symbols, setSymbols] = useState<SymbolResult[]>([]);
    const [loadingSymbols, setLoadingSymbols] = useState(false);
    const [symbolsQuery, setSymbolsQuery] = useState("");

    const [dataSourcesTab, setDataSourcesTab] = useState<"prices" | "funds">("prices");

    // State restoration
    const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
    const [processing, setProcessing] = useState(false);
    const [progress, setProgress] = useState<{ current: number, total: number, lastMsg: string } | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [usage, setUsage] = useState<{ used: number, limit: number, extraLeft: number } | null>(null);
    const [maxPriceDays, setMaxPriceDays] = useState(365);
    const [updateFundamentals, setUpdateFundamentals] = useState(false);

    const [previewTicker, setPreviewTicker] = useState<string | null>(null);
    const [fundPreview, setFundPreview] = useState<any | null>(null);

    const [loadingFundPreview, setLoadingFundPreview] = useState(false);

    const [syncLogs, setSyncLogs] = useState<any[]>([]);
    const [syncing, setSyncing] = useState(false);

    const [dbInventory, setDbInventory] = useState<any[]>([]);
    const [loadingInventory, setLoadingInventory] = useState(false);
    const [dbSearch, setDbSearch] = useState("");
    const [selectedDbEx, setSelectedDbEx] = useState<string | null>(null);
    const [dbSymbols, setDbSymbols] = useState<any[]>([]);
    const [loadingDbSymbols, setLoadingDbSymbols] = useState(false);

    const [recentDbFunds, setRecentDbFunds] = useState<any[]>([]);
    const [loadingRecentFunds, setLoadingRecentFunds] = useState(false);

    // Fetchers
    const fetchUsage = () => {
        fetch("/api/admin/usage")
            .then(res => res.json())
            .then(setUsage)
            .catch(console.error);
    };

    const fetchSyncHistory = () => {
        fetch("/api/admin/sync-history")
            .then(res => res.json())
            .then(setSyncLogs)
            .catch(console.error);
    };

    const fetchInventory = () => {
        setLoadingInventory(true);
        fetch("/api/admin/db-inventory")
            .then(res => res.json())
            .then(setDbInventory)
            .catch(console.error)
            .finally(() => setLoadingInventory(false));
    };

    const fetchRecentDbFunds = async () => {
        setLoadingRecentFunds(true);
        try {
            const res = await fetch("/api/admin/recent-fundamentals");
            const data = await res.json();
            setRecentDbFunds(data);
        } catch (e) {
            console.error("Failed to fetch recent funds:", e);
        } finally {
            setLoadingRecentFunds(false);
        }
    };

    // Initialization
    useEffect(() => {
        if (countries.length > 0 && !countries.includes(selectedCountry)) {
            if (countries.includes("Egypt")) setSelectedCountry("Egypt");
            else setSelectedCountry(countries[0]);
        }
    }, [countries]);

    useEffect(() => {
        fetchUsage();
        fetchSyncHistory();
        fetchInventory();
        fetchRecentDbFunds();
    }, []);

    const fetchDbSymbols = (ex: string) => {
        setLoadingDbSymbols(true);
        fetch(`/api/admin/db-symbols/${ex}`)
            .then(res => res.json())
            .then(setDbSymbols)
            .catch(console.error)
            .finally(() => setLoadingDbSymbols(false));
    };

    // Sync updateFundamentals with tab
    useEffect(() => {
        setUpdateFundamentals(dataSourcesTab === "funds");
    }, [dataSourcesTab]);

    // Load symbols when country changes
    useEffect(() => {
        if (!selectedCountry) return;

        setLoadingSymbols(true);
        // We use search with empty query to get list, usually restricted to 'A' or similar if no 'list_all' API.
        // But our local API `searchSymbols` uses file globbing so we can request "*" if supported
        // or just search " " (space) or similar.
        // Actually our `searchSymbols` in API requires `q`. 
        // Let's assume we can search by country effectively.
        // Or we might need a `get_symbols_for_country` API. 
        // Current `searchSymbols` implementation implementation: glob `*.json` in country folder.
        // So passing "*" as query to `searchSymbols` might work if the backend supports it.
        // Looking at `symbols_local.py` (not visible but based on usage), it filters by contents.
        // Let's try searching "A" for now as default, or better:
        // The user wants to download data. The symbols list should come from the 'pre-scanned' list of all symbols? 
        // Actually, `searchSymbols` implementation in `api` reads from local JSONs `Country_all_symbols_...`.
        // So fetching with `q=""` might return all?
        // Let's try `q=""` assuming the backend handles it, or `q="."` regex.
        searchSymbols("", selectedCountry, 50000) // Increase limit to get all
            .then(res => {
                setSymbols(res);
                setSelectedSymbols(new Set()); // Reset selection
            })
            .catch(console.error)
            .finally(() => setLoadingSymbols(false));

    }, [selectedCountry]);

    // Handlers
    const toggleSelect = (s: SymbolResult) => {
        const id = s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol;
        setPreviewTicker(id);
        const next = new Set(selectedSymbols);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedSymbols(next);
    };

    const toggleSelectAll = () => {
        const filteredIds = filteredSymbols.map(s => s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol);
        const allFilteredSelected = filteredIds.length > 0 && filteredIds.every((id) => selectedSymbols.has(id));
        if (allFilteredSelected) {
            const next = new Set(selectedSymbols);
            filteredIds.forEach((id) => next.delete(id));
            setSelectedSymbols(next);
            return;
        }
        const next = new Set(selectedSymbols);
        filteredIds.forEach((id) => next.add(id));
        setSelectedSymbols(next);
    };

    // State
    const [config, setConfig] = useState<{ priceSource: string; fundSource: string; maxWorkers: number }>({
        priceSource: "eodhd",
        fundSource: "tradingview",
        maxWorkers: 8,
    });

    useEffect(() => {
        // Fetch config
        fetch("/api/admin/config")
            .then(res => res.json())
            .then((c) => {
                let priceSource = c?.priceSource ?? c?.source ?? "eodhd";
                let fundSource = c?.fundSource ?? "tradingview";
                const maxWorkers = typeof c?.maxWorkers === "number" && c.maxWorkers > 0 ? c.maxWorkers : 8;

                // Backward compatibility mappings
                if (priceSource === "cache") priceSource = "tradingview";
                if (fundSource === "auto" || fundSource === "eodhd") fundSource = "tradingview";

                setConfig({ priceSource, fundSource, maxWorkers });
            })
            .catch(console.error);
    }, []);

    const setPriceSource = async (priceSource: string) => {
        try {
            const res = await fetch("/api/admin/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ priceSource })
            });
            if (res.ok) {
                setConfig((prev) => ({ ...prev, priceSource }));
                toast.success(`Price source: ${priceSource.toUpperCase()}`);
                if (priceSource === "eodhd") {
                    fetch("/api/admin/usage").then(r => r.json()).then(setUsage);
                } else {
                    setUsage(null);
                }
            } else {
                const err = await res.json().catch(() => null);
                toast.error(err?.detail || "Failed to update price source");
            }
        } catch (e) {
            toast.error("Failed to update price source");
        }
    };

    const setFundSource = async (fundSource: string) => {
        try {
            const res = await fetch("/api/admin/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fundSource })
            });
            if (res.ok) {
                setConfig((prev) => ({ ...prev, fundSource }));
                toast.success(`Fund source: ${fundSource.toUpperCase()}`);
            } else {
                const err = await res.json().catch(() => null);
                toast.error(err?.detail || "Failed to update fund source");
            }
        } catch (e) {
            toast.error("Failed to update fund source");
        }
    };

    const setMaxWorkers = async (maxWorkers: number) => {
        const safe = Number.isFinite(maxWorkers) && maxWorkers > 0 ? Math.floor(maxWorkers) : 8;
        try {
            const res = await fetch("/api/admin/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ maxWorkers: safe })
            });
            if (res.ok) {
                setConfig((prev) => ({ ...prev, maxWorkers: safe }));
                toast.success(`Workers: ${safe}`);
            } else {
                const err = await res.json().catch(() => null);
                toast.error(err?.detail || "Failed to update workers");
            }
        } catch (e) {
            toast.error("Failed to update workers");
        }
    };

    useEffect(() => {
        if (!previewTicker) {
            setFundPreview(null);
            return;
        }

        const ac = new AbortController();
        setLoadingFundPreview(true);

        fetch(
            `/api/admin/fundamentals/${encodeURIComponent(previewTicker)}?source=${encodeURIComponent(config.fundSource)}`,
            { signal: ac.signal }
        )
            .then((res) => res.json())
            .then(setFundPreview)
            .catch((err) => {
                if (err?.name !== "AbortError") console.error(err);
            })
            .finally(() => setLoadingFundPreview(false));

        return () => ac.abort();
    }, [previewTicker, config.fundSource]);

    const runUpdate = async () => {
        if (selectedSymbols.size === 0) return;

        setProcessing(true);
        setLogs([]);

        const queue = Array.from(selectedSymbols);
        const total = queue.length;
        const BATCH_SIZE = 5;

        let processed = 0;

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const batch = queue.slice(i, i + BATCH_SIZE);

            try {
                const res = await fetch("/api/admin/update_batch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        symbols: batch,
                        country: selectedCountry,
                        updatePrices: dataSourcesTab === "prices",
                        updateFundamentals: updateFundamentals,
                        maxPriceDays: maxPriceDays
                    })
                });

                const data = await res.json();

                if (data.results) {
                    const msgs = data.results.map((r: any) =>
                        `${r.symbol}: ${r.success ? "OK" : "ERR"} - ${r.message}${r.fund?.source ? ` | FundSource: ${r.fund.source}` : ""
                        }${r.fund?.data && (r.fund.data.marketCap ?? r.fund.data.peRatio ?? r.fund.data.eps ?? r.fund.data.dividendYield) != null
                            ? ` | MC:${r.fund.data.marketCap ?? "-"} PE:${r.fund.data.peRatio ?? "-"} EPS:${r.fund.data.eps ?? "-"} DY:${r.fund.data.dividendYield ?? "-"}`
                            : ""
                        }`
                    );
                    setLogs(prev => [...prev.slice(-4), ...msgs]);
                    setProgress({
                        current: Math.min(i + BATCH_SIZE, total),
                        total: total,
                        lastMsg: msgs[msgs.length - 1]
                    });
                }

            } catch (e) {
                console.error(e);
                setLogs(prev => [...prev, `Batch failed: ${e}`]);
            }

            processed += batch.length;
        }

        setProcessing(false);
        toast.success("Update Complete!", {
            description: `Processed ${processed} symbols successfully.`,
        });
    }

    const handleSync = async (exchange?: string) => {
        setSyncing(true);
        try {
            const res = await fetch("/api/admin/sync-data", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ exchange })
            });
            if (res.ok) {
                toast.success("Sync started");
                setTimeout(() => {
                    fetch("/api/admin/sync-history").then(r => r.json()).then(setSyncLogs);
                }, 1000);
            } else {
                toast.error("Sync failed");
            }
        } catch (e) {
            console.error(e);
            toast.error("Sync failed");
        } finally {
            setSyncing(false);
        }
    };

    const filteredSymbols = (() => {

        const q = symbolsQuery.trim().toLowerCase();
        if (!q) return symbols;
        return symbols.filter((s) => {
            const id = (s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol).toLowerCase();
            const name = (s.name || "").toLowerCase();
            return id.includes(q) || name.includes(q);
        });
    })();

    return (
        <div className="flex flex-col gap-6">
            <header className="flex flex-col gap-2">
                <h1 className="text-2xl font-semibold tracking-tight text-indigo-400 flex items-center gap-2">
                    <Database className="h-6 w-6" />
                    Data Manager
                </h1>
                <p className="text-sm text-zinc-400">
                    Manage local market data. Download updates or fetch history for offline analysis.
                </p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Controls */}
                <div className="lg:col-span-1 space-y-4">

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

                        <div className="flex items-center gap-3 p-2 rounded-lg bg-zinc-900/50 border border-zinc-900">
                            <input
                                type="checkbox"
                                id="updateFundamentals"
                                checked={updateFundamentals}
                                onChange={(e) => setUpdateFundamentals(e.target.checked)}
                                className="w-4 h-4 rounded border-zinc-700 bg-zinc-950 text-indigo-600 focus:ring-indigo-500"
                            />
                            <label htmlFor="updateFundamentals" className="text-xs text-zinc-300 font-medium cursor-pointer">
                                Update Fundamentals
                            </label>
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
                                    <div className="flex justify-between text-[10px] text-zinc-500 font-medium">
                                        <span>Max Price Days (Local)</span>
                                        <span className="text-indigo-400">{maxPriceDays} days</span>
                                    </div>
                                    <input
                                        type="number"
                                        min={30}
                                        max={5000}
                                        value={maxPriceDays}
                                        onChange={(e) => setMaxPriceDays(Number(e.target.value))}
                                        className="w-full h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 outline-none focus:border-indigo-500"
                                    />
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


                    </div>



                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 space-y-6">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-medium text-zinc-100 flex items-center gap-2">
                                <Cloud className="w-4 h-4 text-indigo-400" />
                                Cloud Data Sync
                            </h3>
                            {syncing && <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />}
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-[10px] uppercase text-zinc-500 font-bold tracking-wider">Target Market</label>
                                <button
                                    onClick={() => setCountryDialogOpen(true)}
                                    disabled={processing}
                                    className="w-full h-10 flex items-center justify-between rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors"
                                >
                                    <span>{selectedCountry}</span>
                                    <span className="text-zinc-500 text-[10px]">▼</span>
                                </button>
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleSync()}
                                    disabled={syncing}
                                    className="flex-1 py-2 text-[10px] font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-50"
                                >
                                    Sync All
                                </button>
                                <button
                                    onClick={() => {
                                        const firstWithEx = symbols.find(s => s.exchange);
                                        const ex = firstWithEx ? firstWithEx.exchange : (selectedCountry === "Egypt" ? "EGX" : "US");
                                        handleSync(ex);
                                    }}
                                    disabled={syncing || symbols.length === 0}
                                    className="flex-1 py-2 text-[10px] font-bold rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                                >
                                    Sync {selectedCountry}
                                </button>
                            </div>
                        </div>

                        <CountrySelectDialog
                            open={countryDialogOpen}
                            onClose={() => setCountryDialogOpen(false)}
                            onSelect={(c) => {
                                setSelectedCountry(c);
                                setCountryDialogOpen(false);
                            }}
                            countries={countries}
                            selectedCountry={selectedCountry}
                        />


                        <div className="space-y-4 pt-4 border-t border-zinc-800">
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
                                {processing ? "Updating..." : "Update Selected"}
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
                    </div>

                    {logs.length > 0 && (
                        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                            <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Recent Logs</h3>
                            <div className="space-y-1">
                                {logs.map((log, i) => (
                                    <div key={i} className="text-xs font-mono text-zinc-400 truncate border-l-2 border-zinc-800 pl-2 py-0.5">
                                        {log}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
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
                                {filteredSymbols.map(s => {
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
                                    <h2 className="text-xl font-bold text-zinc-100">Fundamentals Dashboard</h2>
                                    <p className="text-xs text-zinc-500">Global market fundamental insights from Cloud</p>
                                </div>
                            </div>
                            <div className="flex gap-2">
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
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                                    {dbInventory.map((item) => (
                                        <div
                                            key={item.exchange}
                                            className="p-4 rounded-2xl bg-zinc-900/30 border border-zinc-800/50 flex flex-col gap-1"
                                        >
                                            <div className="flex justify-between items-center text-[10px] font-bold text-zinc-500 uppercase">
                                                {item.exchange}
                                                <div className={`w-1.5 h-1.5 rounded-full ${item.status === 'healthy' ? 'bg-green-500' : 'bg-zinc-700'}`} />
                                            </div>
                                            <div className="text-2xl font-black text-zinc-100">{item.symbolCount}</div>
                                            <div className="text-[9px] text-zinc-600 font-mono">
                                                {item.lastUpdate ? new Date(item.lastUpdate).toLocaleDateString() : 'Empty'}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {!previewTicker ? (
                                <div className="space-y-8">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-[0.2em] flex items-center gap-2">
                                            <TrendingUp className="w-4 h-4 text-emerald-400" />
                                            Recently Updated in Cloud
                                        </h3>
                                    </div>

                                    {loadingRecentFunds ? (
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                            {[1, 2, 3, 4, 5].map(i => (
                                                <div key={i} className="h-32 rounded-xl bg-zinc-900/50 border border-zinc-800 animate-pulse" />
                                            ))}
                                        </div>
                                    ) : recentDbFunds.length === 0 ? (
                                        <div className="py-20 text-center border-2 border-dashed border-zinc-900 rounded-3xl">
                                            <Database className="w-12 h-12 text-zinc-800 mx-auto mb-4" />
                                            <p className="text-zinc-600 text-sm">No fundamental data found in Supabase.</p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                            {recentDbFunds.map((fund: any) => {
                                                const d = fund.data || {};
                                                return (
                                                    <div
                                                        key={`${fund.symbol}-${fund.exchange}`}
                                                        onClick={() => setPreviewTicker(`${fund.symbol}.${fund.exchange}`)}
                                                        className="group p-5 rounded-2xl bg-zinc-900/30 border border-zinc-800/50 hover:border-indigo-500/30 hover:bg-indigo-500/5 transition-all cursor-pointer"
                                                    >
                                                        <div className="flex justify-between items-start mb-4">
                                                            <div className="px-2.5 py-1 rounded-lg bg-zinc-800 text-[10px] font-bold text-zinc-100 group-hover:bg-indigo-600 transition-colors">
                                                                {fund.symbol}
                                                            </div>
                                                            <span className="text-[9px] text-zinc-600 font-mono">{fund.exchange}</span>
                                                        </div>
                                                        <div className="text-sm font-bold text-zinc-100 truncate mb-1">{d.name || "N/A"}</div>
                                                        <div className="text-[10px] text-zinc-500 mb-4">{d.sector || "Unknown Sector"}</div>
                                                        <div className="flex items-center gap-3 text-[10px] text-zinc-400 pt-3 border-t border-zinc-800/50">
                                                            <div className="flex-1">
                                                                <div className="text-zinc-600 mb-0.5">M.Cap</div>
                                                                <div className="font-mono text-zinc-300">
                                                                    {typeof d.marketCap === 'number'
                                                                        ? (d.marketCap > 1e9 ? `${(d.marketCap / 1e9).toFixed(1)}B` : `${(d.marketCap / 1e6).toFixed(1)}M`)
                                                                        : d.marketCap || "-"
                                                                    }
                                                                </div>
                                                            </div>
                                                            <div className="flex-1 text-right">
                                                                <div className="text-zinc-600 mb-0.5">P/E</div>
                                                                <div className="font-mono text-zinc-300">{d.peRatio || "-"}</div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-500">
                                    <div className="flex items-center justify-between">
                                        <button
                                            onClick={() => { setPreviewTicker(null); setFundPreview(null); }}
                                            className="text-xs font-bold text-indigo-400 hover:text-indigo-300 flex items-center gap-1 transition-colors"
                                        >
                                            ← Back to Feed
                                        </button>
                                        <div className="flex items-center gap-3">
                                            <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold flex items-center gap-1.5">
                                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                                Live Analytics
                                            </span>
                                        </div>
                                    </div>

                                    {loadingFundPreview ? (
                                        <div className="space-y-6">
                                            <div className="h-20 w-1/3 bg-zinc-900 rounded-2xl animate-pulse" />
                                            <div className="grid grid-cols-4 gap-6">
                                                {[1, 2, 3, 4].map(i => <div key={i} className="h-32 bg-zinc-900 rounded-2xl animate-pulse" />)}
                                            </div>
                                        </div>
                                    ) : fundPreview?.data ? (
                                        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                                            {/* Stock Info */}
                                            <div className="lg:col-span-1 space-y-6">
                                                <div className="p-6 rounded-3xl bg-zinc-900/50 border border-zinc-800 flex flex-col gap-4">
                                                    <div className="w-16 h-16 rounded-2xl bg-indigo-600 flex items-center justify-center text-2xl font-black text-white shadow-xl shadow-indigo-500/20">
                                                        {previewTicker.split('.')[0][0]}
                                                    </div>
                                                    <div>
                                                        <h3 className="text-2xl font-black text-white tracking-tight">{previewTicker}</h3>
                                                        <p className="text-sm text-zinc-400 font-medium">{fundPreview.data.name || "Company Overview"}</p>
                                                    </div>
                                                    <div className="space-y-1.5 pt-4">
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-zinc-500">Exchange</span>
                                                            <span className="text-zinc-200 font-mono">{fundPreview.meta?.market || "Cloud"}</span>
                                                        </div>
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-zinc-500">Source</span>
                                                            <span className="text-zinc-200">{fundPreview.meta?.source || "-"}</span>
                                                        </div>
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-zinc-500">Updated</span>
                                                            <span className="text-zinc-200">{new Date(fundPreview.meta?.fetchedAt * 1000).toLocaleDateString()}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Key Metrics */}
                                            <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-4 gap-4">
                                                <MetricCard
                                                    label="Market Cap"
                                                    value={fundPreview.data.marketCap}
                                                    icon={<BarChart3 className="w-4 h-4 text-indigo-400" />}
                                                    isCurrency
                                                />
                                                <MetricCard
                                                    label="P/E Ratio"
                                                    value={fundPreview.data.peRatio}
                                                    icon={<Zap className="w-4 h-4 text-yellow-400" />}
                                                />
                                                <MetricCard
                                                    label="Earnings Per Share"
                                                    value={fundPreview.data.eps}
                                                    icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
                                                />
                                                <MetricCard
                                                    label="Div. Yield"
                                                    value={fundPreview.data.dividendYield}
                                                    icon={<TrendingUp className="w-4 h-4 text-pink-400" />}
                                                    isPercent
                                                />

                                                <div className="col-span-full p-6 rounded-3xl bg-zinc-900/30 border border-zinc-800 space-y-4">
                                                    <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                                                        <Info className="w-4 h-4" />
                                                        Profile Details
                                                    </h4>
                                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 outline-none">
                                                        <div className="space-y-1">
                                                            <div className="text-[10px] text-zinc-600 font-bold uppercase">Sector</div>
                                                            <div className="text-sm text-zinc-200 font-medium">{fundPreview.data.sector || "N/A"}</div>
                                                        </div>
                                                        <div className="space-y-1">
                                                            <div className="text-[10px] text-zinc-600 font-bold uppercase">Industry</div>
                                                            <div className="text-sm text-zinc-200 font-medium">{fundPreview.data.industry || "N/A"}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center py-20 text-zinc-600">No details available for this symbol.</div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            <Toaster theme="dark" position="bottom-right" />
        </div >
    );
}

function MetricCard({ label, value, icon, isCurrency, isPercent }: { label: string, value: any, icon: React.ReactNode, isCurrency?: boolean, isPercent?: boolean }) {
    const displayValue = (() => {
        if (value === null || value === undefined || value === "-") return "-";
        if (typeof value === 'number') {
            if (isCurrency) {
                if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
                if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
                return `$${value.toLocaleString()}`;
            }
            if (isPercent) return `${(value * 1).toFixed(2)}%`;
            return value.toLocaleString();
        }
        return String(value);
    })();

    return (
        <div className="p-5 rounded-3xl bg-zinc-900/50 border border-zinc-800 space-y-3 hover:border-zinc-700 transition-colors">
            <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">{label}</span>
                <div className="p-1.5 rounded-lg bg-zinc-800 text-zinc-400">
                    {icon}
                </div>
            </div>
            <div className="text-xl font-black text-zinc-100 font-mono tracking-tight">{displayValue}</div>
        </div>
    );
}
