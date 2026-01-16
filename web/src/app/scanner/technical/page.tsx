"use client";

import { useState, useEffect, useMemo } from "react";
import { Sliders, Search, Loader2, Globe, Database, TrendingUp, X, Filter, Bookmark, BookmarkCheck, ArrowLeftRight, ChevronLeft, ChevronRight, BarChart3, PieChart, Landmark, Coins, Scale, Percent, Minus, Plus, Info, LayoutTemplate, Settings2 } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import type { TechResult } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";
import StockLogo from "@/components/StockLogo";
import ScannerTemplates, { type ScannerTemplateId } from "@/components/ScannerTemplates";

export default function TechnicalScannerPage() {
    const { t } = useLanguage();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();
    const { state, countries, setTechScanner, runTechScan, stopTechScan, techScanLoading: loading, techScanError: error, clearTechScannerView, restoreLastTechScan, addSymbolToCompare } = useAppState();

    const {
        country,
        results,
        hasScanned,
        scannedCount,
        searchTerm,
        rsiMin,
        rsiMax,
        aboveEma50,
        aboveEma200,
        adxMin,
        adxMax,
        atrMin,
        atrMax,
        stochKMin,
        stochKMax,
        rocMin,
        rocMax,
        aboveVwap20,
        volumeAboveSma20,
        goldenCross,
        selectedStock,
        currentTab,
        marketCapMin,
        marketCapMax,
        sector,
        industry,
    } = state.techScanner;

    // Pagination State
    const [currentPage, setCurrentPage] = useState(1);
    const [showCountryDialog, setShowCountryDialog] = useState(false);
    const pageSize = 15;

    // Filtered Results memo
    const filteredResults = useMemo(() => {
        let res = [...results];
        if (searchTerm) {
            const low = searchTerm.toLowerCase();
            res = res.filter(r => r.symbol.toLowerCase().includes(low) || r.name.toLowerCase().includes(low));
        }
        return res;
    }, [searchTerm, results]);

    // Reset page on search or new results
    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, results.length]);

    const totalPages = Math.ceil(filteredResults.length / pageSize);
    const pagedResults = useMemo(() => {
        const start = (currentPage - 1) * pageSize;
        return filteredResults.slice(start, start + pageSize);
    }, [filteredResults, currentPage]);

    const TABS = [
        { id: 'overview', label: 'Overview', icon: LayoutTemplate },
        { id: 'performance', label: 'Performance', icon: BarChart3 },
        { id: 'valuation', label: 'Valuation', icon: PieChart },
        { id: 'dividends', label: 'Dividends', icon: Coins },
        { id: 'financials', label: 'Financials', icon: Scale },
    ] as const;

    // Auto-fetch data on mount and country change
    useEffect(() => {
        runScan();
    }, [country]);

    async function runScan() {
        await runTechScan();
    }

    function applyTemplate(id: ScannerTemplateId) {
        const baseUpdate = {
            searchTerm: "",
            rsiMin: "",
            rsiMax: "",
            aboveEma50: false,
            aboveEma200: false,
            adxMin: "",
            adxMax: "",
            atrMin: "",
            atrMax: "",
            stochKMin: "",
            stochKMax: "",
            rocMin: "",
            rocMax: "",
            aboveVwap20: false,
            volumeAboveSma20: false,
            goldenCross: false,
            marketCapMin: "",
            marketCapMax: "",
            sector: "",
            industry: "",
        };

        const presets: Record<ScannerTemplateId, Partial<typeof baseUpdate>> = {
            ai_growth: { aboveEma50: true, rsiMin: "45", rsiMax: "70" },
            macd_cross: { goldenCross: true, aboveEma50: true },
            rsi_oversold: { rsiMax: "30" },
            volume_breakout: { volumeAboveSma20: true },
            sma_200_breakout: { aboveEma200: true },
        };

        setTechScanner(prev => ({ ...prev, ...baseUpdate, ...presets[id] }));
        setTimeout(() => void runTechScan({ force: true }), 0);
    }

    const formatNum = (val: number | undefined | null, decimals = 2) => {
        if (val === undefined || val === null || isNaN(val)) return "N/A";
        return val.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    };

    const formatCompact = (val: number | undefined | null) => {
        if (val === undefined || val === null || isNaN(val)) return "-";
        if (val >= 1e12) return (val / 1e12).toFixed(2) + "T";
        if (val >= 1e9) return (val / 1e9).toFixed(2) + "B";
        if (val >= 1e6) return (val / 1e6).toFixed(2) + "M";
        return val.toLocaleString();
    };

    // Filter Dropdown Component
    function FilterDropdown({ label, value, active, onClick }: { label: string; value: string; active?: boolean; onClick: () => void }) {
        return (
            <button
                onClick={onClick}
                className={`
                    h-10 px-4 rounded-xl border flex items-center gap-2 text-xs font-bold transition-all shrink-0
                    ${active
                        ? "bg-blue-600/10 border-blue-500/30 text-blue-400"
                        : "bg-zinc-900/50 border-white/10 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 hover:border-white/20"}
                `}
            >
                <span className="opacity-70">{label}</span>
                <span className="font-black">{value}</span>
            </button>
        );
    }

    return (
        <div className="flex flex-col gap-0 relative min-h-[calc(100vh-100px)] bg-black/20 rounded-3xl overflow-hidden border border-white/5">
            {/* --- TradingView-Style Header --- */}
            <div className="flex flex-col gap-1 p-6 pb-4 border-b border-white/5 bg-zinc-950/40 backdrop-blur-xl">
                <div className="flex items-center gap-3">
                    <h1 className="text-2xl font-black text-white tracking-tight">Stock Screener</h1>
                    {loading && <Loader2 className="h-5 w-5 animate-spin text-blue-500" />}
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <span className="text-zinc-600 font-bold">All stocks</span>
                    {scannedCount > 0 && (
                        <span className="text-zinc-500 text-xs">
                            · {filteredResults.length} of {scannedCount} scanned
                        </span>
                    )}
                </div>
            </div>

            <div className="px-6 pt-6">
                <ScannerTemplates onSelect={applyTemplate} />
            </div>

            {/* --- Inline Filter Bar --- */}
            <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5 bg-zinc-950/20 overflow-x-auto no-scrollbar">
                {/* Country Selector */}
                <button
                    onClick={() => setShowCountryDialog(true)}
                    className="h-10 flex items-center gap-2 rounded-xl border border-white/10 bg-zinc-900/50 px-4 text-sm font-bold text-zinc-200 hover:bg-zinc-800 hover:border-white/20 transition-all shrink-0"
                >
                    <Globe className="h-4 w-4 text-blue-500" />
                    <span className="tracking-wide">{country}</span>
                </button>

                {/* Inline Filter Dropdowns */}
                <FilterDropdown
                    label="Index"
                    value="All"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Price"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Change %"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="RSI"
                    value={rsiMin || rsiMax ? `${rsiMin || 0}-${rsiMax || 100}` : "Any"}
                    active={!!(rsiMin || rsiMax)}
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Market cap"
                    value={marketCapMin || marketCapMax ? `${marketCapMin ? formatCompact(Number(marketCapMin)) : '0'}-${marketCapMax ? formatCompact(Number(marketCapMax)) : '∞'}` : "Any"}
                    active={!!(marketCapMin || marketCapMax)}
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Sector"
                    value={sector || "All"}
                    active={!!sector}
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="P/E"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="EPS dil growth"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Div yield %"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="ROE"
                    value="Any"
                    onClick={() => { }}
                />
                <FilterDropdown
                    label="Beta"
                    value="Any"
                    onClick={() => { }}
                />

                {/* Add More Filter Button */}
                <button className="h-10 w-10 flex items-center justify-center rounded-xl border border-white/10 bg-zinc-900/50 text-zinc-400 hover:bg-zinc-800 hover:text-white hover:border-white/20 transition-all shrink-0">
                    <Plus className="h-4 w-4" />
                </button>
            </div>

            {/* --- Tab Navigation --- */}
            <div className="flex items-center gap-2 px-6 py-3 border-b border-white/5 bg-zinc-950/10 overflow-x-auto no-scrollbar">
                <div className="flex items-center gap-1.5 p-1 rounded-xl bg-zinc-900/40 border border-white/5 shrink-0">
                    {TABS.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setTechScanner(prev => ({ ...prev, currentTab: tab.id as any }))}
                            className={`
                                h-9 px-4 flex items-center gap-2 rounded-lg text-xs font-black uppercase tracking-widest transition-all
                                ${currentTab === tab.id
                                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                                    : "text-zinc-500 hover:text-white hover:bg-white/5"}
                            `}
                        >
                            <tab.icon className="w-3.5 h-3.5" />
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* --- Results Content --- */}
            <div className="flex-1 flex flex-col min-h-0 relative">
                <div className="flex-1 overflow-auto custom-scrollbar">
                    {loading && results.length === 0 ? (
                        <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-6">
                            <div className="relative">
                                <div className="absolute inset-0 blur-3xl bg-blue-500/20 animate-pulse" />
                                <Loader2 className="h-12 w-12 animate-spin text-blue-500 relative z-10" />
                            </div>
                            <div className="flex flex-col items-center gap-1">
                                <p className="text-sm font-black text-white uppercase tracking-[0.3em] animate-pulse">Scanning {country} Market</p>
                                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Applying technical & fundamental filters...</p>
                            </div>
                        </div>
                    ) : results.length === 0 ? (
                        <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-6 text-zinc-700">
                            <Database className="h-16 w-16 opacity-10" />
                            <div className="text-center space-y-2">
                                <p className="text-lg font-black text-white uppercase tracking-widest opacity-20">No Stocks Found</p>
                                <p className="text-xs font-bold text-zinc-600 uppercase tracking-widest">Try adjusting your filters or changing market</p>
                            </div>
                        </div>
                    ) : (
                        <table className="w-full text-left text-sm whitespace-nowrap table-fixed border-collapse">
                            <thead className="sticky top-0 z-20 bg-zinc-950/90 backdrop-blur-md text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500 border-b border-white/10 shadow-xl">
                                <tr>
                                    <th className="w-64 px-8 py-5 text-left border-r border-white/5">Symbol</th>
                                    <th className="w-32 px-6 py-5 text-right">Price</th>
                                    <th className="w-32 px-6 py-5 text-right">Change %</th>
                                    {currentTab === 'overview' && (
                                        <>
                                            <th className="w-32 px-6 py-5 text-right">Volume</th>
                                            <th className="w-32 px-6 py-5 text-right">Mkt Cap</th>
                                            <th className="w-28 px-6 py-5 text-right">P/E</th>
                                            <th className="w-32 px-6 py-5 text-right">EPS</th>
                                            <th className="w-48 px-6 py-5 text-left">Sector</th>
                                        </>
                                    )}
                                    {currentTab === 'performance' && (
                                        <>
                                            <th className="w-32 px-6 py-5 text-center">RSI</th>
                                            <th className="w-32 px-6 py-5 text-right">EMA 50</th>
                                            <th className="w-32 px-6 py-5 text-right">EMA 200</th>
                                            <th className="w-32 px-6 py-5 text-right">Momentum</th>
                                            <th className="w-32 px-6 py-5 text-right">ADX</th>
                                            <th className="w-32 px-6 py-5 text-right">ROC (12)</th>
                                        </>
                                    )}
                                    {currentTab === 'dividends' && (
                                        <>
                                            <th className="w-32 px-6 py-5 text-right">Yield %</th>
                                            <th className="w-48 px-6 py-5 text-left font-mono">Industry</th>
                                        </>
                                    )}
                                    {currentTab === 'valuation' && (
                                        <>
                                            <th className="w-32 px-6 py-5 text-right">Mkt Cap</th>
                                            <th className="w-32 px-6 py-5 text-right">P/E</th>
                                            <th className="w-32 px-6 py-5 text-right">EPS</th>
                                            <th className="w-32 px-6 py-5 text-right">Yield %</th>
                                        </>
                                    )}
                                    {currentTab === 'financials' && (
                                        <>
                                            <th className="w-48 px-6 py-5 text-left">Sector</th>
                                            <th className="w-48 px-6 py-5 text-left">Industry</th>
                                            <th className="w-32 px-6 py-5 text-right">Mkt Cap</th>
                                        </>
                                    )}
                                    <th className="w-20 px-8 py-5 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/[0.03]">
                                {pagedResults.map((r) => (
                                    <tr
                                        key={r.symbol}
                                        onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: r }))}
                                        className={`
                                            group transition-all
                                            ${selectedStock?.symbol === r.symbol ? "bg-blue-600/10" : "hover:bg-white/[0.03]"}
                                        `}
                                    >
                                        <td className="px-8 py-4 border-r border-white/5">
                                            <div className="flex items-center gap-4">
                                                <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="md" />
                                                <div className="flex flex-col min-w-0">
                                                    <span className="font-black text-white text-sm group-hover:text-blue-400 transition-colors uppercase tracking-tight">{r.symbol}</span>
                                                    <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest truncate">{r.name || 'Unknown'}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <span className="font-mono font-black text-zinc-100">{formatNum(r.last_close)}</span>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <span className={`font-mono font-black ${r.change_p >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                                {r.change_p >= 0 ? "+" : ""}{r.change_p.toFixed(2)}%
                                            </span>
                                        </td>
                                        {currentTab === 'overview' && (
                                            <>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-400 text-xs">{formatCompact(r.volume)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-100 font-bold text-xs">{formatCompact(r.market_cap)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-400 text-xs">{formatNum(r.pe_ratio, 1)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-400 text-xs">{formatNum(r.eps, 2)}</td>
                                                <td className="px-6 py-4 text-left">
                                                    <span className="inline-flex px-2 py-0.5 rounded-md bg-white/5 border border-white/5 text-[9px] font-black uppercase font-mono text-zinc-500">{r.sector || '-'}</span>
                                                </td>
                                            </>
                                        )}
                                        {currentTab === 'performance' && (
                                            <>
                                                <td className="px-6 py-4 text-center">
                                                    <div className={`
                                                        inline-flex px-2 py-0.5 rounded text-[10px] font-black
                                                        ${r.rsi < 35 ? "bg-emerald-500/10 text-emerald-400" : r.rsi > 65 ? "bg-red-500/10 text-red-400" : "bg-zinc-900 text-zinc-500"}
                                                    `}>{r.rsi.toFixed(0)}</div>
                                                </td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-500 text-xs">{formatNum(r.ema50)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-500 text-xs">{formatNum(r.ema200)}</td>
                                                <td className="px-6 py-4 text-right">
                                                    <span className={`font-mono font-black text-xs ${r.momentum >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                                        {r.momentum >= 0 ? "+" : ""}{(r.momentum * 100).toFixed(2)}%
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-500 text-xs">{formatNum(r.adx14, 1)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-500 text-xs">{formatNum(r.roc12, 1)}%</td>
                                            </>
                                        )}
                                        {currentTab === 'dividends' && (
                                            <>
                                                <td className="px-6 py-4 text-right font-mono text-blue-400 font-black">{r.dividend_yield ? `${formatNum(r.dividend_yield * 100, 2)}%` : "-"}</td>
                                                <td className="px-6 py-4 text-left font-mono text-zinc-500 text-[10px] uppercase truncate">{r.industry || "-"}</td>
                                            </>
                                        )}
                                        {currentTab === 'valuation' && (
                                            <>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-100 font-bold">{formatCompact(r.market_cap)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-400">{formatNum(r.pe_ratio, 1)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-400">{formatNum(r.eps, 2)}</td>
                                                <td className="px-6 py-4 text-right font-mono text-blue-400">{r.dividend_yield ? `${formatNum(r.dividend_yield * 100, 2)}%` : "-"}</td>
                                            </>
                                        )}
                                        {currentTab === 'financials' && (
                                            <>
                                                <td className="px-6 py-4 text-left">
                                                    <span className="inline-flex px-2 py-0.5 rounded-md bg-white/5 border border-white/5 text-[9px] font-black uppercase font-mono text-zinc-400">{r.sector || '-'}</span>
                                                </td>
                                                <td className="px-6 py-4 text-left">
                                                    <span className="inline-flex px-2 py-0.5 rounded-md bg-white/5 border border-white/5 text-[9px] font-black uppercase font-mono text-zinc-500">{r.industry || '-'}</span>
                                                </td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-100 font-bold">{formatCompact(r.market_cap)}</td>
                                            </>
                                        )}
                                        <td className="px-8 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                            <button
                                                onClick={() => {
                                                    if (isSaved(r.symbol)) removeSymbolBySymbol(r.symbol);
                                                    else saveSymbol({
                                                        symbol: r.symbol,
                                                        name: r.name,
                                                        source: "tech_scanner",
                                                        metadata: { logo_url: r.logo_url }
                                                    });
                                                }}
                                                className={`p-2 rounded-xl transition-all ${isSaved(r.symbol) ? "text-blue-400 bg-blue-500/10" : "text-zinc-600 hover:text-white hover:bg-zinc-800"}`}
                                            >
                                                {isSaved(r.symbol) ? <BookmarkCheck className="h-4.5 w-4.5" /> : <Bookmark className="h-4.5 w-4.5" />}
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* --- Pagination Footer --- */}
                {totalPages > 1 && (
                    <div className="px-8 py-4 border-t border-white/5 bg-zinc-950/80 backdrop-blur-md flex items-center justify-between z-30">
                        <div className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">
                            Page <span className="text-white">{currentPage}</span> / <span className="text-white">{totalPages}</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <button
                                disabled={currentPage === 1}
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                className="h-10 px-6 rounded-xl border border-white/5 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
                            >
                                <ChevronLeft className="w-4 h-4" /> Prev
                            </button>
                            <button
                                disabled={currentPage === totalPages}
                                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                className="h-10 px-6 rounded-xl border border-white/5 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
                            >
                                Next <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                )}

                {/* --- Detail Slide-over (Unchanged but styled higher) --- */}
                {selectedStock && (
                    <>
                        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-[200] animate-in fade-in duration-300" onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: null }))} />
                        <div className="fixed inset-y-0 right-0 w-[450px] bg-zinc-950 border-l border-white/10 z-[201] animate-in slide-in-from-right duration-500 flex flex-col shadow-[0_0_100px_rgba(0,0,0,0.8)]">
                            <div className="p-8 pb-4 flex items-center justify-between border-b border-white/5 bg-white/[0.02]">
                                <div className="flex items-center gap-5">
                                    <StockLogo symbol={selectedStock.symbol} logoUrl={selectedStock.logo_url} size="xl" />
                                    <div className="flex flex-col gap-1">
                                        <h2 className="text-4xl font-black text-white font-mono tracking-tighter leading-none">{selectedStock.symbol}</h2>
                                        <p className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">{selectedStock.name}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: null }))}
                                    className="p-3 rounded-2xl bg-zinc-900/50 text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all border border-white/5"
                                >
                                    <X className="h-5 w-5" />
                                </button>
                            </div>

                            <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-10">
                                {/* Key Stats Grid */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-6 bg-zinc-900/40 rounded-3xl border border-white/5 space-y-1 relative overflow-hidden group">
                                        <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                            <TrendingUp className="w-8 h-8 text-blue-500" />
                                        </div>
                                        <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Price</div>
                                        <div className="text-3xl font-mono font-black text-white">{formatNum(selectedStock.last_close)}</div>
                                        <div className={`text-[11px] font-black ${selectedStock.change_p >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                            {selectedStock.change_p >= 0 ? "+" : ""}{selectedStock.change_p.toFixed(2)}% Today
                                        </div>
                                    </div>
                                    <div className="p-6 bg-zinc-900/40 rounded-3xl border border-white/5 space-y-1 relative overflow-hidden group">
                                        <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                            <Percent className="w-8 h-8 text-purple-500" />
                                        </div>
                                        <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500">RSI (14)</div>
                                        <div className={`text-3xl font-mono font-black ${selectedStock.rsi < 35 ? "text-emerald-400" : selectedStock.rsi > 65 ? "text-red-400" : "text-zinc-100"}`}>
                                            {selectedStock.rsi.toFixed(1)}
                                        </div>
                                        <div className="text-[10px] font-bold uppercase tracking-tighter text-zinc-600">
                                            {selectedStock.rsi < 35 ? "Oversold" : selectedStock.rsi > 65 ? "Overbought" : "Neutral Range"}
                                        </div>
                                    </div>
                                </div>

                                {/* Fundamentals Group */}
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2 text-[11px] font-black text-white uppercase tracking-[0.3em] border-b border-white/5 pb-4">
                                        <Landmark className="w-4 h-4 text-amber-500" />
                                        Company Profile
                                    </div>
                                    <div className="grid grid-cols-1 gap-2">
                                        {[
                                            { label: "Market Cap", val: formatCompact(selectedStock.market_cap), icon: Database },
                                            { label: "Sector", val: selectedStock.sector || "-", icon: LayoutTemplate },
                                            { label: "Industry", val: selectedStock.industry || "-", icon: PieChart },
                                            { label: "P/E Ratio", val: formatNum(selectedStock.pe_ratio, 1), icon: Scale },
                                            { label: "Dividend Yield", val: selectedStock.dividend_yield ? `${(selectedStock.dividend_yield * 100).toFixed(2)}%` : "N/A", icon: Coins },
                                            { label: "EPS (TTM)", val: formatNum(selectedStock.eps, 2), icon: Coins },
                                            { label: "Beta", val: formatNum(selectedStock.beta, 2), icon: TrendingUp },
                                        ].map((m) => (
                                            <div key={m.label} className="flex justify-between items-center p-4 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.05] transition-all">
                                                <div className="flex items-center gap-3">
                                                    <m.icon className="w-3.5 h-3.5 text-zinc-500" />
                                                    <span className="text-[11px] font-bold uppercase tracking-widest text-zinc-500">{m.label}</span>
                                                </div>
                                                <span className="font-mono font-black text-sm text-zinc-200">{m.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Technical Analysis Group */}
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2 text-[11px] font-black text-white uppercase tracking-[0.3em] border-b border-white/5 pb-4">
                                        <BarChart3 className="w-4 h-4 text-blue-500" />
                                        Technical Analysis
                                    </div>
                                    <div className="grid grid-cols-1 gap-2">
                                        {[
                                            { label: "EMA 50", val: formatNum(selectedStock.ema50) },
                                            { label: "EMA 200", val: formatNum(selectedStock.ema200) },
                                            { label: "Momentum", val: `${(selectedStock.momentum * 100).toFixed(2)}%`, color: selectedStock.momentum >= 0 ? "text-emerald-400" : "text-red-400" },
                                            { label: "ADX (Trend)", val: formatNum(selectedStock.adx14, 1) },
                                            { label: "ROC (Rate of Chg)", val: `${formatNum(selectedStock.roc12, 1)}%` },
                                        ].map((m) => (
                                            <div key={m.label} className="flex justify-between items-center p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                                                <span className="text-[11px] font-bold uppercase tracking-widest text-zinc-500">{m.label}</span>
                                                <span className={`font-mono font-black text-sm ${m.color || "text-zinc-200"}`}>{m.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <div className="p-8 border-t border-white/5 bg-white/[0.02] flex gap-4">
                                <button
                                    onClick={() => addSymbolToCompare(selectedStock.symbol)}
                                    className="flex-1 h-14 flex items-center justify-center gap-3 rounded-2xl bg-blue-600 hover:bg-blue-500 text-white font-black text-[11px] uppercase tracking-[0.2em] shadow-2xl shadow-blue-500/20 transition-all group"
                                >
                                    <ArrowLeftRight className="h-4 w-4 group-hover:rotate-12 transition-transform" />
                                    Compare
                                </button>
                                <button
                                    onClick={() => {
                                        if (isSaved(selectedStock.symbol)) removeSymbolBySymbol(selectedStock.symbol);
                                        else saveSymbol({ symbol: selectedStock.symbol, name: selectedStock.name, source: "tech_scanner", metadata: {} });
                                    }}
                                    className="w-14 h-14 flex items-center justify-center rounded-2xl border border-white/10 bg-zinc-900 group"
                                >
                                    {isSaved(selectedStock.symbol) ? (
                                        <BookmarkCheck className="h-5 w-5 text-blue-400" />
                                    ) : (
                                        <Bookmark className="h-5 w-5 text-zinc-500 group-hover:text-white" />
                                    )}
                                </button>
                            </div>
                        </div>
                    </>
                )}
            </div>

            <CountrySelectDialog
                open={showCountryDialog}
                onClose={() => setShowCountryDialog(false)}
                countries={countries}
                selectedCountry={country}
                onSelect={(c) => setTechScanner(prev => ({ ...prev, country: c }))}
            />
        </div>
    );
}
