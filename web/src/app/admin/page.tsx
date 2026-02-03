"use client";

import { useState, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";
import { getCountries, searchSymbols, type SymbolResult } from "@/lib/api";
import { Database, Download, Check, AlertTriangle, Loader2, Zap, Info, TrendingUp, History, Cloud, Globe, ChevronLeft, ChevronRight, ChevronDown, FileText, X, Search } from "lucide-react";
import { Toaster, toast } from "sonner";

import CountrySelectDialog from "@/components/CountrySelectDialog";
import AdminHeader from "./components/AdminHeader";
import DataManagerTab from "./components/DataManagerTab";
import AIAutomationTab from "./components/AIAutomationTab";
import TestModelTab from "./components/TestModelTab";
import FastScannerTab from "./components/FastScannerTab";
import BacktestTab from "./components/BacktestTab";
import SymbolDrillDownModal from "./components/SymbolDrillDownModal";
import RecalculateDialog from "./components/RecalculateDialog";

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

    const [activeMainTab, setActiveMainTab] = useState<"data" | "ai" | "test" | "scan" | "backtest">("data");
    const [dataSourcesTab, setDataSourcesTab] = useState<"prices" | "funds">("prices");

    // State restoration
    const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
    const [processing, setProcessing] = useState(false);
    const [progress, setProgress] = useState<{ current: number, total: number, lastMsg: string } | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
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
    const [updatingInventory, setUpdatingInventory] = useState(false);

    // Fetchers
    const fetchSyncHistory = () => {
        fetch("/api/admin/sync-history")
            .then(async (res) => {
                if (!res.ok) {
                    console.error("Failed to fetch sync history, status", res.status);
                    return [];
                }
                try {
                    return await res.json();
                } catch (e) {
                    console.error("Failed to parse sync history JSON:", e);
                    return [];
                }
            })
            .then((data) => {
                if (Array.isArray(data)) setSyncLogs(data);
                else setSyncLogs([]);
            })
            .catch((e) => {
                console.error("Failed to fetch sync history:", e);
                setSyncLogs([]);
            });
    };

    const fetchInventory = () => {
        setLoadingInventory(true);
        fetch("/api/admin/db-inventory")
            .then(async (res) => {
                if (!res.ok) {
                    console.error("Failed to fetch DB inventory, status", res.status);
                    return [];
                }
                try {
                    return await res.json();
                } catch (e) {
                    console.error("Failed to parse DB inventory JSON:", e);
                    return [];
                }
            })
            .then((data) => {
                if (Array.isArray(data)) setDbInventory(data);
                else setDbInventory([]);
            })
            .catch((e) => {
                console.error("Failed to fetch DB inventory:", e);
                setDbInventory([]);
            })
            .finally(() => setLoadingInventory(false));
    };

    const fetchRecentDbFunds = async () => {
        setLoadingRecentFunds(true);
        try {
            const res = await fetch("/api/admin/recent-fundamentals");
            if (!res.ok) {
                console.error("Failed to fetch recent funds, status", res.status);
                setRecentDbFunds([]);
                return;
            }
            let data: any = [];
            try {
                data = await res.json();
            } catch (e) {
                console.error("Failed to parse recent funds JSON:", e);
                data = [];
            }
            if (Array.isArray(data)) setRecentDbFunds(data);
            else setRecentDbFunds([]);
        } catch (e) {
            console.error("Failed to fetch recent funds:", e);
            setRecentDbFunds([]);
        } finally {
            setLoadingRecentFunds(false);
        }
    };

    const fetchTrainedModels = async () => {
        setLoadingModels(true);
        try {
            const res = await fetch("/api/admin/train/models");
            if (!res.ok) {
                console.error("Failed to fetch models, status", res.status);
                setTrainedModels([]);
                return;
            }
            let data: any = null;
            try {
                data = await res.json();
            } catch (e) {
                console.error("Failed to parse models JSON:", e);
                data = null;
            }
            if (data?.models && Array.isArray(data.models)) {
                setTrainedModels(data.models);
            } else {
                setTrainedModels([]);
            }
        } catch (e) {
            console.error("Failed to fetch models:", e);
            setTrainedModels([]);
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
        fetchSyncHistory();
        fetchInventory();
        fetchRecentDbFunds();
        fetchTrainedModels();
    }, []);

    const fetchDbSymbols = (ex: string, mode: 'prices' | 'fundamentals' = 'prices') => {
        setLoadingDbSymbols(true);
        setSelectedDrillSymbols(new Set());
        fetch(`/api/admin/db-symbols/${ex}?mode=${mode}`)
            .then(async (res) => {
                if (!res.ok) {
                    console.error("Failed to fetch DB symbols, status", res.status);
                    return [];
                }
                try {
                    return await res.json();
                } catch (e) {
                    console.error("Failed to parse DB symbols JSON:", e);
                    return [];
                }
            })
            .then((data) => {
                if (Array.isArray(data)) setDbSymbols(data);
                else setDbSymbols([]);
            })
            .catch((e) => {
                console.error("Failed to fetch DB symbols:", e);
                setDbSymbols([]);
            })
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
            .then(async (res) => {
                if (!res.ok) {
                    console.error("Failed to fetch admin config, status", res.status);
                    return null;
                }
                try {
                    return await res.json();
                } catch (e) {
                    console.error("Failed to parse admin config JSON:", e);
                    return null;
                }
            })
            .then((c) => {
                if (!c) return;
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

    const runInventoryUpdate = async (country?: string) => {
        setUpdatingInventory(true);
        const tid = toast.loading(country ? `Updating ${country} symbols...` : "Updating global symbols inventory...");
        try {
            const url = country
                ? `/api/admin/update-symbols-inventory?country=${encodeURIComponent(country)}`
                : "/api/admin/update-symbols-inventory";
            const res = await fetch(url, { method: "POST" });
            if (!res.ok) throw new Error("API error");
            toast.success("Inventory update started in background", { id: tid });
            // Refresh countries list after some time or immediately to see new files if they are fast
            setTimeout(() => {
                getCountries("local").then(setCountries);
            }, 5000);
        } catch (e) {
            toast.error("Failed to start inventory update", { id: tid });
        } finally {
            setUpdatingInventory(false);
        }
    };

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
            <AdminHeader
                activeMainTab={activeMainTab}
                setActiveMainTab={setActiveMainTab}
            />

            {/* Main Content Area */}
            <main className="flex-1 w-full overflow-y-auto relative">
                {activeMainTab === "data" ? (
                    <DataManagerTab
                        selectedCountry={selectedCountry}
                        setCountryDialogOpen={setCountryDialogOpen}
                        processing={processing}
                        updatingInventory={updatingInventory}
                        runInventoryUpdate={runInventoryUpdate}
                        dataSourcesTab={dataSourcesTab}
                        setDataSourcesTab={setDataSourcesTab}
                        config={config}
                        setPriceSource={setPriceSource}
                        setFundSource={setFundSource}
                        maxPriceDays={maxPriceDays}
                        setMaxPriceDays={setMaxPriceDays}
                        selectedSymbols={selectedSymbols}
                        runUpdate={runUpdate}
                        progress={progress}
                        logs={logs}
                        setLogs={setLogs}
                        filteredSymbols={filteredSymbols}
                        loadingSymbols={loadingSymbols}
                        toggleSelectAll={toggleSelectAll}
                        symbolsQuery={symbolsQuery}
                        setSymbolsQuery={setSymbolsQuery}
                        paginatedSymbols={paginatedSymbols}
                        toggleSelect={toggleSelect}
                        pageSize={pageSize}
                        setPageSize={setPageSize}
                        currentPage={currentPage}
                        setCurrentPage={setCurrentPage}
                        totalPages={totalPages}
                        setRecalcDialogOpen={setRecalcDialogOpen}
                        recalculatingIndicators={recalculatingIndicators}
                        fetchRecentDbFunds={fetchRecentDbFunds}
                        fetchInventory={fetchInventory}
                        loadingRecentFunds={loadingRecentFunds}
                        dbInventory={dbInventory}
                        showEmptyExchanges={showEmptyExchanges}
                        setShowEmptyExchanges={setShowEmptyExchanges}
                        setSelectedDbEx={setSelectedDbEx}
                        setDrillDownMode={setDrillDownMode}
                        fetchDbSymbols={fetchDbSymbols}
                        setSelectedCountry={setSelectedCountry}
                        setAutoSelectPending={setAutoSelectPending}
                        setActiveMainTab={setActiveMainTab}
                        loadingInventory={loadingInventory}
                        setMaxWorkers={setMaxWorkers}
                        setConfig={setConfig}
                    />
                ) : activeMainTab === "ai" ? (
                    <AIAutomationTab
                        dbInventory={dbInventory}
                        trainingExchange={trainingExchange}
                        setTrainingExchange={setTrainingExchange}
                        isExchangeDropdownOpen={isExchangeDropdownOpen}
                        setIsExchangeDropdownOpen={setIsExchangeDropdownOpen}
                        isTraining={isTraining}
                        fetchTrainedModels={fetchTrainedModels}
                        loadingModels={loadingModels}
                        trainedModels={trainedModels}
                        handleDownloadModel={handleDownloadModel}
                        setIsTraining={setIsTraining}
                    />
                ) : activeMainTab === "scan" ? (
                    <FastScannerTab />
                ) : activeMainTab === "backtest" ? (
                    <BacktestTab />
                ) : (
                    <TestModelTab />
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
                <SymbolDrillDownModal
                    selectedDbEx={selectedDbEx}
                    drillDownMode={drillDownMode}
                    dbSymbols={dbSymbols}
                    loadingDbSymbols={loadingDbSymbols}
                    dbSymbolsSort={dbSymbolsSort}
                    setDbSymbolsSort={setDbSymbolsSort}
                    selectedDrillSymbols={selectedDrillSymbols}
                    setSelectedDrillSymbols={setSelectedDrillSymbols}
                    handleDownloadCsv={handleDownloadCsv}
                    setSelectedDbEx={setSelectedDbEx}
                    setDbSymbols={setDbSymbols}
                    setDrillDownMode={setDrillDownMode}
                    handleRecalculateIndicators={handleRecalculateIndicators}
                    selectedSymbols={selectedSymbols}
                    setSelectedSymbols={setSelectedSymbols}
                    setActiveMainTab={setActiveMainTab}
                />
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

