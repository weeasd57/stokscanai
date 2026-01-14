"use client";

import { useState, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";
import { getCountries, searchSymbols, type SymbolResult } from "@/lib/api";
import { Database, Download, Check, AlertTriangle, Loader2, Zap, BarChart3, Info, TrendingUp, History, Cloud, Globe, ChevronLeft, ChevronRight, ChevronDown, FileText, X, Search } from "lucide-react";
import { Toaster, toast } from "sonner";

import CountrySelectDialog from "@/components/CountrySelectDialog";

export default function AdminPage() {
    const { t } = useLanguage();
    // Admin specifically needs local symbols list to manage syncing
    const [countries, setCountries] = useState<string[]>([]);
    const [countriesLoading, setCountriesLoading] = useState(false);

    useEffect(() => {
        setCountriesLoading(true);
        getCountries("local")
            .then(setCountries)
            .finally(() => setCountriesLoading(false));
    }, []);

    // State
    // const [countries, setCountries] = useState<string[]>([]); // Removed local state
    const [selectedCountry, setSelectedCountry] = useState("Egypt");
    // const [loadingCountries, setLoadingCountries] = useState(false); // Removed local state
    const [countryDialogOpen, setCountryDialogOpen] = useState(false);

    const [symbols, setSymbols] = useState<SymbolResult[]>([]);
    const [loadingSymbols, setLoadingSymbols] = useState(false);
    const [symbolsQuery, setSymbolsQuery] = useState("");
    const [currentPage, setCurrentPage] = useState(1);
    const [pageSize, setPageSize] = useState(100);

    const [activeMainTab, setActiveMainTab] = useState<"data" | "ai">("data");
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

    const [dbInventory, setDbInventory] = useState<any[]>([]);
    const [loadingInventory, setLoadingInventory] = useState(false);
    const [dbSearch, setDbSearch] = useState("");
    const [selectedDbEx, setSelectedDbEx] = useState<string | null>(null);
    const [drillDownMode, setDrillDownMode] = useState<'prices' | 'fundamentals' | null>(null);
    const [autoSelectPending, setAutoSelectPending] = useState<string | null>(null);
    const [dbSymbols, setDbSymbols] = useState<any[]>([]);
    const [selectedDrillSymbols, setSelectedDrillSymbols] = useState<Set<string>>(new Set());
    const [loadingDbSymbols, setLoadingDbSymbols] = useState(false);
    const [dbSymbolsSort, setDbSymbolsSort] = useState<{ key: string, dir: 'asc' | 'desc' }>({ key: 'symbol', dir: 'asc' });

    const [recentDbFunds, setRecentDbFunds] = useState<any[]>([]);
    const [loadingRecentFunds, setLoadingRecentFunds] = useState(false);
    const [showEmptyExchanges, setShowEmptyExchanges] = useState(false);
    const [recalculatingIndicators, setRecalculatingIndicators] = useState(false);
    const [recalcDialogOpen, setRecalcDialogOpen] = useState(false);
    const [recalcExchange, setRecalcExchange] = useState<string | null>(null);
    const [recalcSearch, setRecalcSearch] = useState("");
    const [recalcResults, setRecalcResults] = useState<any[]>([]);
    const [selectedRecalcSymbols, setSelectedRecalcSymbols] = useState<Set<string>>(new Set());
    const [loadingRecalcResults, setLoadingRecalcResults] = useState(false);
    const [trainedModels, setTrainedModels] = useState<any[]>([]);
    const [loadingModels, setLoadingModels] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingExchange, setTrainingExchange] = useState("");
    const [isExchangeDropdownOpen, setIsExchangeDropdownOpen] = useState(false);

    // Smart Sync & Scheduling State
    const [smartSyncExchange, setSmartSyncExchange] = useState("");
    const [isSmartSyncDropdownOpen, setIsSmartSyncDropdownOpen] = useState(false);
    const [smartSyncSearch, setSmartSyncSearch] = useState("");
    const [smartSyncDays, setSmartSyncDays] = useState(365);
    const [smartSyncPrices, setSmartSyncPrices] = useState(true);
    const [smartSyncFunds, setSmartSyncFunds] = useState(false);
    const [smartSyncUnified, setSmartSyncUnified] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncCron, setSyncCron] = useState("30 22 * * *");

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
            .then(data => {
                if (Array.isArray(data)) setSyncLogs(data);
                else setSyncLogs([]);
            })
            .catch(console.error);
    };

    const fetchInventory = () => {
        setLoadingInventory(true);
        fetch("/api/admin/db-inventory")
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data)) setDbInventory(data);
                else setDbInventory([]);
            })
            .catch(console.error)
            .finally(() => setLoadingInventory(false));
    };

    const fetchRecentDbFunds = async () => {
        setLoadingRecentFunds(true);
        try {
            const res = await fetch("/api/admin/recent-fundamentals");
            const data = await res.json();
            if (Array.isArray(data)) setRecentDbFunds(data);
            else setRecentDbFunds([]);
        } catch (e) {
            console.error("Failed to fetch recent funds:", e);
        } finally {
            setLoadingRecentFunds(false);
        }
    };

    const fetchTrainedModels = async () => {
        setLoadingModels(true);
        try {
            const res = await fetch("/api/admin/train/models");
            const data = await res.json();
            if (data.models && Array.isArray(data.models)) {
                setTrainedModels(data.models);
            }
        } catch (e) {
            console.error("Failed to fetch models:", e);
        } finally {
            setLoadingModels(false);
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
        fetchTrainedModels();
    }, []);

    const fetchDbSymbols = (ex: string, mode: 'prices' | 'fundamentals' = 'prices') => {
        setLoadingDbSymbols(true);
        setSelectedDrillSymbols(new Set());
        fetch(`/api/admin/db-symbols/${ex}?mode=${mode}`)
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data)) setDbSymbols(data);
                else setDbSymbols([]);
            })
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
        searchSymbols("", selectedCountry, 100000, undefined, "local") // Increase limit further to get all (100k)
            .then(res => {
                setSymbols(res);
                setSelectedSymbols(new Set()); // Reset selection
            })
            .catch(console.error)
            .finally(() => setLoadingSymbols(false));

    }, [selectedCountry]);

    // Reset page on search or country change
    useEffect(() => {
        setCurrentPage(1);
    }, [selectedCountry, symbolsQuery, pageSize]);

    // Auto-select remaining symbols when both lists are ready
    useEffect(() => {
        if (!autoSelectPending || loadingSymbols || loadingDbSymbols) return;

        // Find symbols in the current 'symbols' list that match the exchange but aren't in 'dbSymbols'
        const inDbSet = new Set(dbSymbols.map(s => s.symbol));
        const missing = symbols.filter(s => {
            if (s.exchange !== autoSelectPending) return false;
            return !inDbSet.has(s.symbol);
        });

        if (missing.length > 0) {
            const newSelected = new Set(selectedSymbols);
            missing.forEach(s => {
                const id = s.exchange ? `${s.symbol}.${s.exchange}` : s.symbol;
                newSelected.add(id);
            });
            setSelectedSymbols(newSelected);
            toast.info(`Auto-selected ${missing.length} missing symbols for ${autoSelectPending}`);
        } else {
            toast.success(`All symbols for ${autoSelectPending} are already in database`);
        }

        setAutoSelectPending(null);
    }, [symbols, dbSymbols, autoSelectPending, loadingSymbols, loadingDbSymbols]);

    const handleDownloadCsv = (exchange: string, symbol?: string) => {
        const mode = drillDownMode === 'fundamentals' ? 'export-fundamentals' : 'export-prices';
        const url = `/api/admin/${mode}/${exchange}${symbol ? `?symbol=${symbol}` : ""}`;
        window.open(url, "_blank");
    };

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
                    setLogs(prev => [...prev, ...msgs]);
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

    const handleRecalculateIndicators = async (exchange: string, symbolsOverride?: string[]) => {
        const finalSymbols = symbolsOverride || [];
        const isExchangeWide = !symbolsOverride || symbolsOverride.length === 0;

        setRecalculatingIndicators(true);
        try {
            const body: any = {
                symbols: finalSymbols,
                exchange: exchange
            };

            const res = await fetch("/api/admin/recalculate-indicators", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });

            if (res.ok) {
                const data = await res.json();
                toast.success(isExchangeWide ? "Exchange Recalculation Started" : "Bulk Recalculation Started", {
                    description: data.message || "Indicators are being recalculated in the background."
                });
            } else {
                const err = await res.json().catch(() => null);
                toast.error(err?.detail || "Failed to start recalculation");
            }
        } catch (e) {
            toast.error("Connection error");
        } finally {
            setRecalculatingIndicators(false);
        }
    }

    const handleTriggerSmartSync = async () => {
        if (!smartSyncExchange) return;
        setIsSyncing(true);
        try {
            const res = await fetch("/api/admin/sync/trigger", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    exchange: smartSyncExchange,
                    days: smartSyncDays,
                    updatePrices: smartSyncPrices,
                    updateFunds: smartSyncFunds,
                    unified: smartSyncUnified
                })
            });
            if (res.ok) {
                toast.success("Smart Sync Pipeline Triggered", {
                    description: `Data sync started for ${smartSyncExchange} via GitHub Actions.`
                });
            } else {
                const err = await res.json().catch(() => ({}));
                toast.error(err.detail || "Failed to trigger sync");
            }
        } catch (e) {
            toast.error("Network error triggering sync");
        } finally {
            setIsSyncing(false);
        }
    };

    const handleSetCron = async () => {
        try {
            const res = await fetch("/api/admin/sync/schedule", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    cron: syncCron,
                    startTime: "22:30", // Placeholders for now
                    endTime: "04:00"
                })
            });
            if (res.ok) {
                toast.success("Schedule Preference Updated", {
                    description: "The new cron schedule has been recorded."
                });
            } else {
                toast.error("Failed to update schedule");
            }
        } catch (e) {
            toast.error("Network error updating schedule");
        }
    };

    const handleTriggerTraining = async (exchange: string) => {
        setIsTraining(true);
        try {
            const res = await fetch("/api/admin/train/trigger", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ exchange })
            });

            if (res.ok) {
                toast.success("AI Training Triggered", {
                    description: `GitHub Action has been started for ${exchange}. This may take several minutes.`
                });
            } else {
                const err = await res.json().catch(() => null);
                toast.error(err?.detail || "Failed to trigger training");
            }
        } catch (e) {
            toast.error("Connection error");
        } finally {
            setIsTraining(false);
        }
    };

    const handleDownloadModel = async (filename: string) => {
        try {
            const res = await fetch(`/api/admin/train/download/${filename}`);
            const data = await res.json();
            if (data.url) {
                window.open(data.url, "_blank");
            } else {
                toast.error("Failed to get download URL");
            }
        } catch (e) {
            toast.error("Failed to download model");
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

    const paginatedSymbols = (() => {
        const start = (currentPage - 1) * pageSize;
        return filteredSymbols.slice(start, start + pageSize);
    })();

    const totalPages = Math.ceil(filteredSymbols.length / pageSize);

    return (
        <div className="min-h-screen bg-black text-zinc-100 flex flex-col selection:bg-indigo-500/30">
            {/* Top Navigation Header */}
            <header className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-black/80 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-4 md:px-8 h-20 flex items-center justify-between gap-8">
                    {/* Brand */}
                    <div className="flex flex-col">
                        <h2 className="text-xl font-black tracking-tighter text-white flex items-center gap-2">
                            <Database className="w-6 h-6 text-indigo-500" />
                            STOKSCAN <span className="text-indigo-500">AI</span>
                        </h2>
                        <p className="text-[9px] text-zinc-500 font-bold uppercase tracking-[0.2em] mt-0.5">Admin Control Room</p>
                    </div>

                    {/* Navigation Tabs */}
                    <nav className="flex items-center bg-zinc-900/50 p-1 rounded-xl border border-zinc-800/50">
                        <button
                            onClick={() => setActiveMainTab("data")}
                            className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-bold transition-all ${activeMainTab === "data"
                                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20"
                                : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
                                }`}
                        >
                            <Database className="w-4 h-4" />
                            Data Manager
                        </button>
                        <button
                            onClick={() => setActiveMainTab("ai")}
                            className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-bold transition-all ${activeMainTab === "ai"
                                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20"
                                : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
                                }`}
                        >
                            <Zap className="w-4 h-4" />
                            AI & Automation
                        </button>
                    </nav>

                    {/* Usage Stats (Condensed) */}
                    <div className="hidden lg:flex items-center gap-4 px-4 py-2 rounded-xl bg-zinc-900/30 border border-zinc-800/50">
                        <div className="flex flex-col gap-1 min-w-[120px]">
                            <div className="flex justify-between text-[9px] font-bold text-zinc-500 uppercase">
                                <span>API Usage</span>
                                <span className="text-indigo-400 font-mono">{usage?.used || 0} / {usage?.limit || 1000}</span>
                            </div>
                            <div className="h-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-indigo-500 transition-all"
                                    style={{ width: `${Math.min(100, ((usage?.used || 0) / (usage?.limit || 1)) * 100)}%` }}
                                />
                            </div>
                        </div>
                        <BarChart3 className="w-4 h-4 text-zinc-600" />
                    </div>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="flex-1 w-full overflow-y-auto relative">
                {activeMainTab === "data" ? (
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
                ) : (
                    <div className="p-8 max-w-7xl mx-auto w-full space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <header className="flex flex-col gap-2">
                            <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
                                <Zap className="h-8 w-8 text-indigo-500" />
                                AI & Automation
                            </h1>
                            <p className="text-sm text-zinc-500 font-medium">
                                Control AI training pipelines and GitHub Actions cron job settings.
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
                                            <h2 className="text-xl font-black text-white">Trigger Training</h2>
                                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Manual GH Dispatch</p>
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

                                        <div className="grid grid-cols-2 gap-3">
                                            <button
                                                onClick={() => handleTriggerTraining(trainingExchange)}
                                                disabled={!trainingExchange || isTraining}
                                                className="w-full py-4 bg-zinc-800 hover:bg-zinc-700 text-white rounded-2xl text-[10px] font-black flex flex-col items-center justify-center gap-1 disabled:opacity-50 transition-all border border-zinc-700"
                                            >
                                                {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cloud className="w-4 h-4" />}
                                                GITHUB ACTION
                                            </button>
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
                                                        if (res.ok) toast.success(data.message);
                                                        else toast.error(data.detail);
                                                    } catch (e) {
                                                        toast.error("Failed to start local training");
                                                    } finally {
                                                        setIsTraining(false);
                                                    }
                                                }}
                                                disabled={!trainingExchange || isTraining}
                                                className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl text-[10px] font-black flex flex-col items-center justify-center gap-1 disabled:opacity-50 transition-all shadow-[0_0_20px_rgba(99,102,241,0.2)]"
                                            >
                                                {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                                                TRAIN LOCALLY
                                            </button>
                                        </div>

                                        <div className="flex items-start gap-3 p-4 rounded-xl bg-zinc-950 border border-zinc-800/50">
                                            <Info className="w-4 h-4 text-zinc-500 shrink-0 mt-0.5" />
                                            <div className="space-y-1">
                                                <p className="text-[10px] text-zinc-400 font-bold uppercase">Training Options</p>
                                                <p className="text-[10px] text-zinc-600 font-medium leading-relaxed">
                                                    Use <span className="text-zinc-300">GitHub Action</span> to train in the cloud (no local load). <br />
                                                    Use <span className="text-zinc-300">Train Locally</span> to run on this server (faster debug).
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Unified Data Pipeline Control */}
                                <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-8">
                                    <div className="flex items-center gap-4 border-b border-zinc-800 pb-6">
                                        <div className="p-3 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-indigo-500/20 border border-indigo-500/20 text-white">
                                            <Cloud className="w-6 h-6" />
                                        </div>
                                        <div>
                                            <h2 className="text-xl font-black text-white">Data Pipeline Control</h2>
                                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Manage & Schedule Sync Tasks</p>
                                        </div>
                                    </div>

                                    {/* 1. Configuration (Shared) */}
                                    <div className="space-y-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                                            <span className="text-[10px] uppercase font-black text-zinc-400">1. Configure Task Scope</span>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Target Exchange</label>
                                                <div className="relative">
                                                    <button
                                                        onClick={() => setIsSmartSyncDropdownOpen(!isSmartSyncDropdownOpen)}
                                                        className={`w-full bg-black border ${isSmartSyncDropdownOpen ? 'border-indigo-500 ring-1 ring-indigo-500/50' : 'border-zinc-800'} rounded-2xl p-4 text-sm text-left transition-all flex items-center justify-between group hover:border-zinc-700`}
                                                    >
                                                        <span className={`${smartSyncExchange ? 'text-white font-medium' : 'text-zinc-500'}`}>
                                                            {smartSyncExchange ? (
                                                                <span className="flex items-center gap-2">
                                                                    <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"></span>
                                                                    {smartSyncExchange}
                                                                    <span className="text-zinc-600 text-xs ml-1">
                                                                        ({dbInventory.find(i => i.exchange === smartSyncExchange)?.priceCount || 0} symbols)
                                                                    </span>
                                                                </span>
                                                            ) : (
                                                                "Select Target..."
                                                            )}
                                                        </span>
                                                        <ChevronDown className={`w-4 h-4 text-zinc-500 transition-transform duration-300 ${isSmartSyncDropdownOpen ? 'rotate-180 text-indigo-500' : 'group-hover:text-zinc-300'}`} />
                                                    </button>

                                                    {/* Dropdown Menu */}
                                                    <div className={`absolute z-30 top-full left-0 right-0 mt-2 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden transition-all duration-200 origin-top transform ${isSmartSyncDropdownOpen ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 -translate-y-2 pointer-events-none'}`}>

                                                        {/* Search Input */}
                                                        <div className="p-2 border-b border-zinc-800">
                                                            <div className="relative">
                                                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-500" />
                                                                <input
                                                                    type="text"
                                                                    placeholder="Search markets..."
                                                                    value={smartSyncSearch}
                                                                    onChange={(e) => setSmartSyncSearch(e.target.value)}
                                                                    className="w-full bg-black border border-zinc-800 rounded-xl py-2 pl-8 pr-3 text-[10px] text-zinc-300 focus:outline-none focus:border-indigo-500 transition-all placeholder:text-zinc-600"
                                                                    autoFocus
                                                                />
                                                            </div>
                                                        </div>

                                                        <div className="p-2 space-y-1 max-h-[300px] overflow-y-auto custom-scrollbar">
                                                            {dbInventory
                                                                .filter(i => i.exchange.toLowerCase().includes(smartSyncSearch.toLowerCase()))
                                                                .length === 0 && (
                                                                    <div className="p-4 text-center text-xs text-zinc-600 italic">No markets found.</div>
                                                                )}
                                                            {dbInventory
                                                                .filter(i => i.exchange.toLowerCase().includes(smartSyncSearch.toLowerCase()))
                                                                .map(i => (
                                                                    <button
                                                                        key={i.exchange}
                                                                        onClick={() => {
                                                                            setSmartSyncExchange(i.exchange);
                                                                            setIsSmartSyncDropdownOpen(false);
                                                                        }}
                                                                        className={`w-full p-3 rounded-xl text-left text-sm font-medium transition-all flex items-center justify-between group ${smartSyncExchange === i.exchange ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' : 'hover:bg-zinc-800 text-zinc-400 hover:text-white'}`}
                                                                    >
                                                                        <span className="flex items-center gap-3">
                                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-black uppercase tracking-wider ${smartSyncExchange === i.exchange ? 'bg-white/20 text-white' : 'bg-black border border-zinc-800 text-zinc-500 group-hover:border-zinc-700'}`}>
                                                                                {i.exchange.substring(0, 2)}
                                                                            </div>
                                                                            <div>
                                                                                <div className="">{i.exchange}</div>
                                                                                <div className={`text-[10px] ${smartSyncExchange === i.exchange ? 'text-indigo-200' : 'text-zinc-600 group-hover:text-zinc-500'}`}>
                                                                                    {i.count || i.priceCount || 0} DB Records
                                                                                </div>
                                                                            </div>
                                                                        </span>
                                                                        {smartSyncExchange === i.exchange && <Check className="w-4 h-4" />}
                                                                    </button>
                                                                ))}
                                                        </div>
                                                    </div>
                                                </div>
                                                {/* Backdrop to close */}
                                                {isSmartSyncDropdownOpen && (
                                                    <div className="fixed inset-0 z-20 bg-transparent" onClick={() => setIsSmartSyncDropdownOpen(false)} />
                                                )}
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Data Depth (Days)</label>
                                                <input
                                                    type="number"
                                                    value={smartSyncDays}
                                                    onChange={(e) => setSmartSyncDays(Number(e.target.value))}
                                                    className="w-full bg-black border border-zinc-800 rounded-2xl p-4 text-xs text-zinc-300 focus:border-indigo-500 transition-all"
                                                />
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-3 gap-3">
                                            <button
                                                onClick={() => setSmartSyncPrices(!smartSyncPrices)}
                                                className={`py-3 rounded-xl border text-[10px] font-bold transition-all flex flex-col items-center gap-1 ${smartSyncPrices ? 'bg-indigo-600/10 border-indigo-500/40 text-indigo-400' : 'bg-black border-zinc-800 text-zinc-500'}`}
                                            >
                                                <TrendingUp className="w-4 h-4" />
                                                UPDATE PRICES
                                            </button>
                                            <button
                                                onClick={() => setSmartSyncFunds(!smartSyncFunds)}
                                                className={`py-3 rounded-xl border text-[10px] font-bold transition-all flex flex-col items-center gap-1 ${smartSyncFunds ? 'bg-indigo-600/10 border-indigo-500/40 text-indigo-400' : 'bg-black border-zinc-800 text-zinc-500'}`}
                                            >
                                                <FileText className="w-4 h-4" />
                                                UPDATE FUNDS
                                            </button>
                                            <button
                                                onClick={() => setSmartSyncUnified(!smartSyncUnified)}
                                                className={`py-3 rounded-xl border text-[10px] font-bold transition-all flex flex-col items-center gap-1 ${smartSyncUnified ? 'bg-indigo-600/10 border-indigo-500/40 text-indigo-400' : 'bg-black border-zinc-800 text-zinc-500'}`}
                                            >
                                                <Zap className="w-4 h-4" />
                                                FORCE UNIFIED
                                            </button>
                                        </div>
                                    </div>

                                    {/* Divider */}
                                    <div className="h-px w-full bg-zinc-800/50"></div>

                                    {/* 2. Execution Methods */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                        {/* A. Manual Trigger */}
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                                                <span className="text-[10px] uppercase font-black text-zinc-400">Method A: Manual Run</span>
                                            </div>
                                            <div className="p-4 rounded-2xl bg-black border border-zinc-800/50 space-y-3 h-full flex flex-col justify-between">
                                                <p className="text-[10px] text-zinc-500 leading-relaxed">
                                                    Immediately triggers the sync process for the selected scope via GitHub Actions.
                                                </p>
                                                <button
                                                    onClick={handleTriggerSmartSync}
                                                    disabled={!smartSyncExchange || isSyncing}
                                                    className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl text-sm font-black flex items-center justify-center gap-3 disabled:opacity-50 transition-all shadow-lg shadow-emerald-500/10"
                                                >
                                                    {isSyncing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Cloud className="w-5 h-5" />}
                                                    RUN NOW
                                                </button>
                                            </div>
                                        </div>

                                        {/* B. Schedule (Cron) */}
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="w-1.5 h-1.5 rounded-full bg-amber-500"></div>
                                                <span className="text-[10px] uppercase font-black text-zinc-400">Method B: Automation</span>
                                            </div>
                                            <div className="p-4 rounded-2xl bg-black border border-zinc-800/50 space-y-4">
                                                <div className="space-y-2">
                                                    <div className="flex justify-between items-center">
                                                        <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Cron Schedule</label>
                                                        <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">ACTIVE</span>
                                                    </div>
                                                    <div className="flex flex-col xl:flex-row gap-2">
                                                        <input
                                                            type="text"
                                                            value={syncCron}
                                                            onChange={(e) => setSyncCron(e.target.value)}
                                                            className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-xs font-mono text-zinc-300 focus:border-indigo-500 transition-all w-full"
                                                            placeholder="0 0 * * *"
                                                        />
                                                        <button
                                                            onClick={handleSetCron}
                                                            className="px-6 py-3 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-[10px] font-black transition-all whitespace-nowrap"
                                                        >
                                                            SAVE
                                                        </button>
                                                    </div>
                                                </div>
                                                <button
                                                    onClick={() => window.open('https://github.com/weeasd57/stokscanai/actions', '_blank')}
                                                    className="w-full py-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-white rounded-xl text-[10px] font-bold transition-all flex items-center justify-center gap-2 border border-zinc-800"
                                                >
                                                    <History className="w-4 h-4" />
                                                    VIEW LOGS
                                                </button>
                                            </div>
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
                                                <FileText className="w-6 h-6" />
                                            </div>
                                            <div>
                                                <h2 className="text-xl font-black text-white">Model Artifacts</h2>
                                                <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">Trained .pkl Modules</p>
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
                                                <FileText className="w-12 h-12" />
                                                <p className="text-xs font-bold uppercase tracking-widest">No models found in cloud storage</p>
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
                )}
            </main>

            <CountrySelectDialog
                open={countryDialogOpen}
                onClose={() => setCountryDialogOpen(false)}
                onSelect={setSelectedCountry}
                countries={countries}
                selectedCountry={selectedCountry}
                forcedAdmin={true}
            />

            {selectedDbEx && (
                <div className="fixed inset-0 z-[150] flex items-center justify-center p-4 md:p-8 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
                    <div className="w-full max-w-6xl max-h-[90vh] bg-zinc-950 border border-zinc-800 rounded-3xl shadow-2xl flex flex-col overflow-hidden">
                        <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-900/50">
                            <div className="flex items-center gap-4">
                                <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                                    {drillDownMode === 'prices' ? <TrendingUp className="w-5 h-5 text-indigo-400" /> : <FileText className="w-5 h-5 text-emerald-400" />}
                                </div>
                                <div>
                                    <h3 className="text-xl font-black text-white tracking-tight">
                                        {selectedDbEx} {drillDownMode === 'prices' ? 'Stock Prices' : 'Fundamentals Data'}
                                    </h3>
                                    <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5">
                                        {dbSymbols.length} Total Symbols in database
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => handleDownloadCsv(selectedDbEx)}
                                    className="px-4 py-2 rounded-xl bg-zinc-900 border border-zinc-800 text-[10px] font-bold text-zinc-400 hover:text-white hover:border-zinc-700 transition-all flex items-center gap-2"
                                >
                                    <Download className="w-4 h-4" />
                                    DOWNLOAD CSV
                                </button>
                                <button
                                    onClick={() => { setSelectedDbEx(null); setDbSymbols([]); setDrillDownMode(null); }}
                                    className="p-2 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        </div>

                        <div className="flex-1 overflow-auto p-6 bg-zinc-950/50">
                            {loadingDbSymbols ? (
                                <div className="flex flex-col items-center justify-center py-20 gap-4">
                                    <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
                                    <p className="text-xs font-bold text-zinc-600 uppercase tracking-widest">Loading database records...</p>
                                </div>
                            ) : dbSymbols.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-20 grayscale opacity-50">
                                    <Database className="w-16 h-16 text-zinc-700 mb-4" />
                                    <p className="text-sm font-bold text-zinc-500 uppercase tracking-widest">No matching records found</p>
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-zinc-800/50 overflow-hidden bg-zinc-900/20 backdrop-blur-md">
                                    <table className="w-full text-left text-[11px]">
                                        <thead className="bg-zinc-900/80 text-zinc-500 font-bold uppercase tracking-wider border-b border-zinc-800">
                                            <tr>
                                                <th className="px-6 py-4">
                                                    <input
                                                        type="checkbox"
                                                        checked={dbSymbols.length > 0 && selectedDrillSymbols.size === dbSymbols.length}
                                                        onChange={(e) => {
                                                            if (e.target.checked) setSelectedDrillSymbols(new Set(dbSymbols.map(s => s.symbol)));
                                                            else setSelectedDrillSymbols(new Set());
                                                        }}
                                                        className="w-4 h-4 rounded border-zinc-700 bg-zinc-950 text-indigo-600 focus:ring-indigo-500"
                                                    />
                                                </th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-indigo-400 transition-colors" onClick={() => setDbSymbolsSort(p => ({ key: 'symbol', dir: p.key === 'symbol' && p.dir === 'asc' ? 'desc' : 'asc' }))}>
                                                    Ticker {dbSymbolsSort.key === 'symbol' ? (dbSymbolsSort.dir === 'asc' ? '↑' : '↓') : ''}
                                                </th>
                                                <th className="px-6 py-4">Name</th>
                                                <th className="px-6 py-4">Sector</th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-indigo-400 transition-colors" onClick={() => setDbSymbolsSort(p => ({ key: 'row_count', dir: p.key === 'row_count' && p.dir === 'asc' ? 'desc' : 'asc' }))}>
                                                    Count {dbSymbolsSort.key === 'row_count' ? (dbSymbolsSort.dir === 'asc' ? '↑' : '↓') : ''}
                                                </th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-indigo-400 transition-colors" onClick={() => setDbSymbolsSort(p => ({ key: 'last_sync', dir: p.key === 'last_sync' && p.dir === 'asc' ? 'desc' : 'asc' }))}>
                                                    Last Sync {dbSymbolsSort.key === 'last_sync' ? (dbSymbolsSort.dir === 'asc' ? '↑' : '↓') : ''}
                                                </th>
                                                <th className="px-6 py-4 cursor-pointer hover:text-indigo-400 transition-colors" onClick={() => setDbSymbolsSort(p => ({ key: 'last_price_date', dir: p.key === 'last_price_date' && p.dir === 'asc' ? 'desc' : 'asc' }))}>
                                                    Last Price {dbSymbolsSort.key === 'last_price_date' ? (dbSymbolsSort.dir === 'asc' ? '↑' : '↓') : ''}
                                                </th>
                                                <th className="px-6 py-4 text-right">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-zinc-800/50">
                                            {[...dbSymbols].sort((a, b) => {
                                                const valA = a[dbSymbolsSort.key] || '';
                                                const valB = b[dbSymbolsSort.key] || '';
                                                if (typeof valA === 'number' && typeof valB === 'number') {
                                                    return dbSymbolsSort.dir === 'asc' ? valA - valB : valB - valA;
                                                }
                                                return dbSymbolsSort.dir === 'asc' ? valA.toString().localeCompare(valB.toString()) : valB.toString().localeCompare(valA.toString());
                                            }).map((s) => (
                                                <tr key={s.symbol} className={`hover:bg-zinc-800/40 transition-colors group/row ${selectedDrillSymbols.has(s.symbol) ? 'bg-indigo-600/5' : ''}`}>
                                                    <td className="px-6 py-4">
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedDrillSymbols.has(s.symbol)}
                                                            onChange={(e) => {
                                                                const next = new Set(selectedDrillSymbols);
                                                                if (e.target.checked) next.add(s.symbol);
                                                                else next.delete(s.symbol);
                                                                setSelectedDrillSymbols(next);
                                                            }}
                                                            className="w-4 h-4 rounded border-zinc-700 bg-zinc-950 text-indigo-600 focus:ring-indigo-500"
                                                        />
                                                    </td>
                                                    <td className="px-6 py-4 font-mono font-black text-indigo-400 group-hover/row:text-indigo-300">{s.symbol}</td>
                                                    <td className="px-6 py-4 text-zinc-300 font-medium">{s.name}</td>
                                                    <td className="px-6 py-4">
                                                        <span className="px-2 py-0.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-500 font-bold uppercase tracking-tighter text-[9px]">
                                                            {s.sector}
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 text-indigo-400 font-mono font-bold">
                                                        {s.row_count || 0}
                                                    </td>
                                                    <td className="px-6 py-4 text-zinc-400 font-mono">
                                                        {s.last_sync ? new Date(s.last_sync).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : '—'}
                                                    </td>
                                                    <td className="px-6 py-4 text-zinc-500 font-mono">
                                                        {s.last_price_date || '—'}
                                                    </td>
                                                    <td className="px-6 py-4 text-right">
                                                        <button
                                                            onClick={() => handleDownloadCsv(selectedDbEx, s.symbol)}
                                                            className="p-2 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-500 hover:text-indigo-400 hover:border-indigo-500/30 transition-all shadow-sm"
                                                            title="Download Prices CSV"
                                                        >
                                                            <Download className="w-3.5 h-3.5" />
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                        <div className="p-4 bg-zinc-900/30 border-t border-zinc-800 flex items-center gap-3 backdrop-blur-sm">
                            <div className="mr-auto flex items-center gap-4">
                                <p className="text-[10px] text-zinc-600 font-bold uppercase tracking-widest flex items-center gap-2">
                                    <Info className="w-3.5 h-3.5" />
                                    {selectedDrillSymbols.size} Selected
                                </p>
                            </div>

                            {selectedDrillSymbols.size > 0 && (
                                <>
                                    <button
                                        onClick={() => {
                                            if (!selectedDbEx) return;
                                            handleRecalculateIndicators(selectedDbEx, Array.from(selectedDrillSymbols));
                                        }}
                                        className="px-4 py-2 rounded-xl bg-amber-500/10 border border-amber-500/30 text-[10px] font-bold text-amber-500 hover:bg-amber-600 hover:text-white transition-all flex items-center gap-2"
                                    >
                                        <Zap className="w-3.5 h-3.5" />
                                        RECALC TECHNICALS
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (!selectedDbEx) return;
                                            const newSelection = new Set(selectedSymbols);
                                            selectedDrillSymbols.forEach(sym => {
                                                const id = `${sym}.${selectedDbEx}`;
                                                newSelection.add(id);
                                            });
                                            setSelectedSymbols(newSelection);
                                            setActiveMainTab("data");
                                            setSelectedDbEx(null);
                                            toast.success(`Added ${selectedDrillSymbols.size} symbols to Data Manager queue`);
                                        }}
                                        className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-[10px] font-bold hover:bg-indigo-500 transition-all flex items-center gap-2"
                                    >
                                        <Database className="w-3.5 h-3.5" />
                                        USE IN DATA MANAGER
                                    </button>
                                </>
                            )}

                            <button
                                onClick={() => { setSelectedDbEx(null); setDbSymbols([]); setDrillDownMode(null); }}
                                className="px-6 py-2 rounded-xl bg-zinc-900 border border-zinc-800 text-[10px] font-bold text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
                            >
                                CLOSE VIEW
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <RecalculateDialog
                open={recalcDialogOpen}
                onClose={() => setRecalcDialogOpen(false)}
                exchanges={dbInventory.filter(i => i.priceCount > 0).map(i => ({ exchange: i.exchange, country: i.country, count: i.priceCount }))}
                onRun={(exchange) => {
                    handleRecalculateIndicators(exchange);
                    setRecalcDialogOpen(false);
                }}
                recalculating={recalculatingIndicators}
            />

            <Toaster theme="dark" position="bottom-right" />
        </div>
    );
}

function RecalculateDialog({ open, onClose, exchanges, onRun, recalculating }: {
    open: boolean,
    onClose: () => void,
    exchanges: { exchange: string, country: string, count: number }[],
    onRun: (exchange: string) => void,
    recalculating: boolean
}) {
    const [selectedExchange, setSelectedExchange] = useState<string | null>(null);

    useEffect(() => {
        if (!open) {
            setSelectedExchange(null);
        }
    }, [open]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-in fade-in duration-200">
            <div className="w-full max-w-2xl bg-zinc-950 border border-zinc-800 rounded-3xl shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
                <div className="p-6 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                            <Zap className="w-5 h-5" />
                        </div>
                        <div>
                            <h3 className="text-xl font-black text-white">Recalculate Technicals</h3>
                            <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Select an exchange to recalculate all indicators</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 rounded-xl hover:bg-zinc-800 transition-colors">
                        <X className="w-5 h-5 text-zinc-500" />
                    </button>
                </div>

                <div className="flex-1 overflow-auto p-6 space-y-4 min-h-0">
                    {exchanges.length === 0 ? (
                        <div className="py-20 text-center opacity-30 italic text-sm">No exchanges with data found</div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                            {exchanges.map(ex => (
                                <button
                                    key={ex.exchange}
                                    onClick={() => setSelectedExchange(ex.exchange)}
                                    className={`p-4 rounded-xl border text-left transition-all ${selectedExchange === ex.exchange
                                        ? 'bg-indigo-600/20 border-indigo-500/50 ring-2 ring-indigo-500/30'
                                        : 'bg-zinc-900/40 border-zinc-800/50 hover:border-zinc-700'
                                        }`}
                                >
                                    <div className="font-bold text-sm text-zinc-100 flex justify-between items-center">
                                        {ex.exchange}
                                        {selectedExchange === ex.exchange && <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />}
                                    </div>
                                    <div className="text-[9px] text-zinc-500 font-semibold uppercase">{ex.country}</div>
                                    <div className="mt-2 text-xs text-indigo-400 font-mono">{ex.count} symbols</div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="p-6 border-t border-zinc-800 bg-zinc-900/50 flex items-center justify-between">
                    <div className="text-[10px] font-bold text-zinc-500 uppercase">
                        {selectedExchange ? `Selected: ${selectedExchange}` : 'No exchange selected'}
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-6 py-2.5 rounded-xl border border-zinc-800 text-xs font-bold text-zinc-500 hover:bg-zinc-900 transition-all"
                        >
                            CANCEL
                        </button>
                        <button
                            onClick={() => selectedExchange && onRun(selectedExchange)}
                            disabled={!selectedExchange || recalculating}
                            className="px-8 py-2.5 rounded-xl bg-indigo-600 text-xs font-black text-white hover:bg-indigo-500 disabled:opacity-50 transition-all flex items-center gap-2"
                        >
                            {recalculating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                            RECALCULATE ALL
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
