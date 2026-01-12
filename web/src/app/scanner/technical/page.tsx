"use client";

import { useState, useEffect, useMemo } from "react";
import { Sliders, Search, Loader2, Globe, Database, TrendingUp, X, Filter, Bookmark, BookmarkCheck, ArrowLeftRight, ChevronLeft, ChevronRight } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import type { TechResult } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";

export default function TechnicalScannerPage() {
    const { t } = useLanguage();
    const { saveSymbol, isSaved } = useWatchlist();
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

    async function runScan() {
        await runTechScan();
    }

    return (
        <div className="flex flex-col gap-6 relative min-h-[calc(100vh-100px)]">
            <header className="flex flex-col gap-2">
                <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-3 uppercase">
                    <div className="p-2 rounded-xl bg-indigo-600 shadow-lg shadow-indigo-600/20">
                        <TrendingUp className="h-6 w-6 text-white" />
                    </div>
                    {t("tech.title")}
                </h1>
                <p className="text-sm text-zinc-500 font-medium">
                    {t("tech.subtitle")}
                </p>
            </header>

            <div className="flex flex-col lg:flex-row gap-6">

                {/* --- Filters Panel --- */}
                <div className="w-full lg:w-72 shrink-0 space-y-4">
                    <div className="rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl p-5 space-y-6 shadow-xl">
                        <div className="flex items-center gap-2 text-zinc-100 font-bold uppercase tracking-widest text-[11px] pb-4 border-b border-white/5">
                            <Filter className="h-3.5 w-3.5 text-indigo-400" />
                            {t("tech.config")}
                        </div>

                        {/* Country */}
                        <div className="space-y-2">
                            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] ml-1">{t("tech.market")}</label>
                            <button
                                onClick={() => setShowCountryDialog(true)}
                                className="w-full h-11 flex items-center justify-between rounded-xl border border-white/5 bg-zinc-900/50 px-4 text-sm text-zinc-200 hover:bg-zinc-800 transition-all group"
                            >
                                <div className="flex items-center gap-3">
                                    <Globe className="h-4 w-4 text-blue-500 opacity-70" />
                                    <span className="font-semibold">{country}</span>
                                </div>
                                <span className="text-[10px] text-zinc-600 group-hover:text-zinc-400 transition-colors">▼</span>
                            </button>
                            <CountrySelectDialog
                                open={showCountryDialog}
                                onClose={() => setShowCountryDialog(false)}
                                countries={countries}
                                selectedCountry={country}
                                onSelect={(c) => setTechScanner(prev => ({ ...prev, country: c }))}
                            />
                        </div>

                        {/* Indicators Grid */}
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] ml-1">{t("tech.rsi")}</label>
                                <div className="grid grid-cols-2 gap-2">
                                    <input
                                        type="number" placeholder="Min"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={rsiMin} onChange={e => setTechScanner(prev => ({ ...prev, rsiMin: e.target.value }))}
                                    />
                                    <input
                                        type="number" placeholder="Max"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={rsiMax} onChange={e => setTechScanner(prev => ({ ...prev, rsiMax: e.target.value }))}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] ml-1">{t("tech.adx")}</label>
                                <div className="grid grid-cols-2 gap-2">
                                    <input
                                        type="number" placeholder="Min"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={adxMin} onChange={e => setTechScanner(prev => ({ ...prev, adxMin: e.target.value }))}
                                    />
                                    <input
                                        type="number" placeholder="Max"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={adxMax} onChange={e => setTechScanner(prev => ({ ...prev, adxMax: e.target.value }))}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] ml-1">{t("tech.atr")}</label>
                                <div className="grid grid-cols-2 gap-2">
                                    <input
                                        type="number" placeholder="Min"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={atrMin} onChange={e => setTechScanner(prev => ({ ...prev, atrMin: e.target.value }))}
                                    />
                                    <input
                                        type="number" placeholder="Max"
                                        className="w-full h-9 rounded-lg bg-zinc-900 border border-white/5 px-3 text-sm text-zinc-100 placeholder:text-zinc-700 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                                        value={atrMax} onChange={e => setTechScanner(prev => ({ ...prev, atrMax: e.target.value }))}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Checkboxes Group */}
                        <div className="space-y-2.5 pt-2">
                            {[
                                { id: "aboveEma50", label: t("tech.price_above_ema50"), checked: aboveEma50, color: "text-indigo-500" },
                                { id: "aboveEma200", label: t("tech.price_above_ema200"), checked: aboveEma200, color: "text-indigo-500" },
                                { id: "goldenCross", label: t("tech.golden_cross"), checked: goldenCross, color: "text-amber-500" },
                                { id: "aboveVwap20", label: t("tech.price_above_vwap20"), checked: aboveVwap20, color: "text-blue-500" },
                                { id: "volumeAboveSma20", label: t("tech.volume_spike"), checked: volumeAboveSma20, color: "text-purple-500" },
                            ].map((cb) => (
                                <label key={cb.id} className="flex items-center gap-3 text-xs font-bold uppercase tracking-wider text-zinc-400 cursor-pointer group hover:text-white transition-all">
                                    <div className={`w-4 h-4 rounded border border-white/10 flex items-center justify-center transition-all ${cb.checked ? "bg-indigo-600 border-indigo-600" : "bg-zinc-800"}`}>
                                        {cb.checked && <X className="w-2.5 h-2.5 text-white stroke-[4px]" />}
                                    </div>
                                    <input
                                        type="checkbox"
                                        className="hidden"
                                        checked={cb.checked}
                                        onChange={e => setTechScanner(prev => ({ ...prev, [cb.id]: e.target.checked }))}
                                    />
                                    <span className={cb.checked ? "text-zinc-100" : ""}>{cb.label}</span>
                                </label>
                            ))}
                        </div>

                        {/* Action Buttons */}
                        <div className="pt-4 border-t border-white/5">
                            {loading ? (
                                <button
                                    onClick={stopTechScan}
                                    className="w-full h-12 flex items-center justify-center gap-2 rounded-xl bg-red-600/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-all font-black text-[11px] uppercase tracking-widest"
                                >
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    {t("tech.stop_scan")}
                                </button>
                            ) : (
                                <button
                                    onClick={runScan}
                                    className="w-full h-12 flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-xl shadow-indigo-600/20 transition-all font-black text-[11px] uppercase tracking-widest group"
                                >
                                    <Search className="h-4 w-4 group-hover:scale-110 transition-transform" />
                                    {t("tech.start_scan")}
                                </button>
                            )}
                        </div>
                    </div>
                </div>


                {/* --- Results Table --- */}
                <div className="flex-1 min-w-0 flex flex-col gap-4">

                    {/* Toolbar */}
                    <div className="flex flex-wrap items-center justify-between gap-4 py-1">
                        <div className="relative">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-600" />
                            <input
                                type="text" placeholder={t("tech.quick_search")}
                                value={searchTerm} onChange={e => setTechScanner(prev => ({ ...prev, searchTerm: e.target.value }))}
                                className="h-11 w-full md:w-80 rounded-xl bg-zinc-950/40 border border-white/5 pl-11 pr-4 text-sm text-zinc-200 focus:ring-1 focus:ring-indigo-500 focus:outline-none transition-all"
                            />
                        </div>
                        <div className="flex items-center gap-4">
                            {hasScanned && (
                                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500">
                                    {t("tech.found_matches").replace("{count}", filteredResults.length.toString())}
                                </div>
                            )}
                            {results.length > 0 && (
                                <button
                                    onClick={() => clearTechScannerView()}
                                    disabled={loading}
                                    className="h-11 px-6 rounded-xl border border-white/5 bg-zinc-950/20 text-[10px] font-black uppercase tracking-widest text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-50"
                                >
                                    {t("tech.clear_results")}
                                </button>
                            )}
                        </div>
                    </div>

                    {error && (
                        <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400 flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
                            <X className="h-4 w-4" /> {error}
                        </div>
                    )}

                    <div className="rounded-2xl border border-white/5 bg-zinc-950/40 backdrop-blur-xl overflow-hidden shadow-2xl min-h-[500px] flex flex-col">
                        <div className="flex-1 overflow-x-auto custom-scrollbar">
                            {loading && results.length === 0 ? (
                                <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-4 text-zinc-500">
                                    <div className="relative">
                                        <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                        <div className="absolute inset-0 blur-xl bg-indigo-500/20 animate-pulse" />
                                    </div>
                                    <p className="text-xs font-bold uppercase tracking-widest animate-pulse">Analysing market data...</p>
                                </div>
                            ) : !hasScanned ? (
                                <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-4 text-zinc-700">
                                    <Database className="h-12 w-12 opacity-10" />
                                    <p className="text-sm font-medium">{t("tech.ready")}</p>
                                </div>
                            ) : filteredResults.length === 0 ? (
                                <div className="flex h-full min-h-[400px] flex-col items-center justify-center py-20 text-zinc-600 font-medium">
                                    {t("tech.no_matches")}
                                </div>
                            ) : (
                                <table className="w-full text-left text-sm whitespace-nowrap">
                                    <thead className="bg-zinc-950/80 text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500 border-b border-white/5">
                                        <tr>
                                            <th className="px-6 py-4 font-black">{t("tech.table.symbol")}</th>
                                            <th className="px-5 py-4 text-right">{t("tech.table.price")}</th>
                                            <th className="px-5 py-4 text-center">RSI</th>
                                            <th className="px-5 py-4 text-right">EMA 50</th>
                                            <th className="px-5 py-4 text-right">EMA 200</th>
                                            <th className="px-5 py-4 text-right">{t("tech.table.momentum")}</th>
                                            <th className="px-5 py-4 text-right">ADX</th>
                                            <th className="px-5 py-4 text-right">ATR</th>
                                            <th className="px-5 py-4 text-right">Stoch</th>
                                            <th className="px-5 py-4 text-right">ROC</th>
                                            <th className="px-6 py-4 text-right">{t("tech.table.save")}</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-white/5">
                                        {pagedResults.map((r) => (
                                            <tr
                                                key={r.symbol}
                                                onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: r }))}
                                                className={`
                                                    cursor-pointer transition-colors group
                                                    ${selectedStock?.symbol === r.symbol ? "bg-indigo-500/10" : "hover:bg-white/[0.02]"}
                                                `}
                                            >
                                                <td className="px-6 py-4">
                                                    <div className="flex flex-col">
                                                        <span className="font-mono font-bold text-indigo-400 group-hover:text-indigo-300 transition-colors">{r.symbol}</span>
                                                        <span className="text-[10px] text-zinc-500 truncate max-w-[140px] font-medium uppercase tracking-tighter">{r.name}</span>
                                                    </div>
                                                </td>
                                                <td className="px-5 py-4 text-right">
                                                    <div className="font-mono text-zinc-100 font-bold">{r.last_close.toFixed(2)}</div>
                                                </td>
                                                <td className="px-5 py-4 text-center">
                                                    <div className={`
                                                        inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-black
                                                        ${r.rsi < 35 ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : r.rsi > 65 ? "bg-red-500/10 text-red-400 border border-red-500/20" : "bg-zinc-900 text-zinc-500"}
                                                    `}>
                                                        {r.rsi.toFixed(0)}
                                                    </div>
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-xs">
                                                    {r.ema50.toFixed(2)}
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-xs">
                                                    {r.ema200.toFixed(2)}
                                                </td>
                                                <td className="px-5 py-4 text-right font-mono text-xs">
                                                    <span className={`font-bold ${r.momentum >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                                        {r.momentum > 0 ? "+" : ""}{((r.momentum) * 100).toFixed(2)}%
                                                    </span>
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-xs">
                                                    {typeof r.adx14 === 'number' && Number.isFinite(r.adx14) ? r.adx14.toFixed(1) : "-"}
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-xs">
                                                    {typeof r.atr14 === 'number' && Number.isFinite(r.atr14) ? r.atr14.toFixed(2) : "-"}
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-xs">
                                                    {typeof r.stoch_k === 'number' && Number.isFinite(r.stoch_k) ? r.stoch_k.toFixed(0) : "-"}
                                                </td>
                                                <td className="px-5 py-4 text-right text-zinc-500 font-mono text-[11px]">
                                                    {typeof r.roc12 === 'number' && Number.isFinite(r.roc12) ? `${r.roc12.toFixed(1)}%` : "-"}
                                                </td>
                                                <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                                    <button
                                                        onClick={() => !isSaved(r.symbol) && saveSymbol({
                                                            symbol: r.symbol,
                                                            name: r.name,
                                                            source: "tech_scanner",
                                                            metadata: { rsi: r.rsi, momentum: r.momentum, ema50: r.ema50, price: r.last_close }
                                                        })}
                                                        className={`p-2 rounded-xl transition-all ${isSaved(r.symbol)
                                                            ? "text-indigo-400 bg-indigo-500/10 cursor-default"
                                                            : "text-zinc-600 hover:text-white hover:bg-zinc-800"
                                                            }`}
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

                        {/* Pagination Footer */}
                        {totalPages > 1 && (
                            <div className="bg-zinc-950/80 px-6 py-4 border-t border-white/5 flex items-center justify-between">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
                                    {t("pagination.page")} {currentPage} / {totalPages}
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        disabled={currentPage === 1}
                                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                        className="h-10 px-4 rounded-xl border border-white/5 flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
                                    >
                                        <ChevronLeft className="w-4 h-4" />
                                        {t("pagination.prev")}
                                    </button>
                                    <button
                                        disabled={currentPage === totalPages}
                                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                        className="h-10 px-4 rounded-xl border border-white/5 flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all disabled:opacity-20"
                                    >
                                        {t("pagination.next")}
                                        <ChevronRight className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* --- Detail Side Panel (Slide Over) --- */}
                {selectedStock && (
                    <>
                        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]" onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: null }))} />
                        <div className="fixed inset-y-0 right-0 w-80 bg-zinc-950 border-l border-white/10 shadow-2xl p-6 overflow-y-auto z-[101] animate-in slide-in-from-right duration-500">
                            <button
                                onClick={() => setTechScanner(prev => ({ ...prev, selectedStock: null }))}
                                className="absolute top-4 right-4 p-2 rounded-xl text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all"
                            >
                                <X className="h-5 w-5" />
                            </button>

                            <div className="mt-4 space-y-8">
                                <div className="space-y-1">
                                    <h2 className="text-3xl font-black text-white font-mono tracking-tighter">{selectedStock.symbol}</h2>
                                    <p className="text-xs font-bold uppercase tracking-widest text-zinc-500">{selectedStock.name}</p>
                                </div>

                                <div className="grid grid-cols-2 gap-3">
                                    <div className="p-4 bg-zinc-900/50 rounded-2xl border border-white/5">
                                        <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500 mb-1">Price</div>
                                        <div className="text-xl font-mono font-black text-white tracking-tight">{selectedStock.last_close.toFixed(2)}</div>
                                    </div>
                                    <div className="p-4 bg-zinc-900/50 rounded-2xl border border-white/5">
                                        <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500 mb-1">RSI</div>
                                        <div className={`text-xl font-mono font-black tracking-tight ${selectedStock.rsi < 35 ? "text-emerald-400" : selectedStock.rsi > 65 ? "text-red-400" : "text-zinc-300"}`}>
                                            {selectedStock.rsi.toFixed(1)}
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <h3 className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] border-b border-white/5 pb-3">Technical Metrics</h3>

                                    {[
                                        { label: "EMA 50", val: selectedStock.ema50.toFixed(2) },
                                        { label: "EMA 200", val: selectedStock.ema200.toFixed(2) },
                                        { label: "Momentum", val: `${(selectedStock.momentum * 100).toFixed(2)}%`, color: selectedStock.momentum >= 0 ? "text-emerald-400" : "text-red-400" },
                                        { label: "ADX (14)", val: Number.isFinite(selectedStock.adx14) ? selectedStock.adx14!.toFixed(1) : "-" },
                                        { label: "ATR (14)", val: Number.isFinite(selectedStock.atr14) ? selectedStock.atr14!.toFixed(2) : "-" },
                                        { label: "Stoch %K/%D", val: `${Number.isFinite(selectedStock.stoch_k) ? selectedStock.stoch_k!.toFixed(0) : "-"} / ${Number.isFinite(selectedStock.stoch_d) ? selectedStock.stoch_d!.toFixed(0) : "-"}` },
                                        { label: "ROC (12)", val: Number.isFinite(selectedStock.roc12) ? `${selectedStock.roc12!.toFixed(1)}%` : "-", color: (selectedStock.roc12 || 0) >= 0 ? "text-emerald-400" : "text-red-400" },
                                    ].map((m) => (
                                        <div key={m.label} className="flex justify-between items-center group">
                                            <span className="text-[11px] font-bold uppercase tracking-widest text-zinc-500 group-hover:text-zinc-400 transition-colors">{m.label}</span>
                                            <span className={`font-mono font-bold text-xs ${m.color || "text-zinc-300"}`}>{m.val}</span>
                                        </div>
                                    ))}
                                </div>

                                <div className="p-5 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 text-center space-y-2">
                                    <div className="text-[9px] text-indigo-400 uppercase tracking-widest font-black">AI Market Signal</div>
                                    <div className="font-black text-sm text-white tracking-widest uppercase">
                                        {selectedStock.rsi < 35 && selectedStock.momentum > 0 ? "STRONG ACCUMULATE" :
                                            selectedStock.rsi > 65 && selectedStock.momentum < 0 ? "POSSIBLE DISTRIBUTION" : "MARKET NEUTRAL"}
                                    </div>
                                </div>

                                <button
                                    onClick={() => {
                                        void addSymbolToCompare(selectedStock.symbol);
                                    }}
                                    className="w-full h-12 flex items-center justify-center gap-3 rounded-2xl bg-blue-600 hover:bg-blue-500 text-white font-black text-[11px] uppercase tracking-[0.2em] shadow-xl shadow-blue-500/20 transition-all group"
                                >
                                    <ArrowLeftRight className="h-4 w-4 group-hover:rotate-12 transition-transform" />
                                    Add to Compare
                                </button>
                            </div>
                        </div>
                    </>
                )}

            </div>
        </div>
    );
}
