"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { X, Search, Globe, CheckSquare, Square, Loader2 } from "lucide-react";
import { searchSymbols, type SymbolResult } from "@/lib/api";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";

interface CountrySymbolDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSelect: (symbols: string[]) => void;
    multiSelect?: boolean; // Allow multiple selection (default: true)
}

export default function CountrySymbolDialog({
    isOpen,
    onClose,
    onSelect,
    multiSelect = true,
}: CountrySymbolDialogProps) {
    // Force multiSelect to true if passed as undefined, though default handles it.
    // User requested "Add Button" flow always.
    const effectiveMultiSelect = multiSelect ?? true;
    const { t } = useLanguage();
    const {
        countries,
        countriesLoading,
        refreshCountries,
        syncedSymbols,
        syncedSymbolsLoading,
        refreshSyncedSymbols,
        isCountryActive,
        isSymbolActive,
        isAdmin,
        inventory
    } = useAppState();
    const [selectedCountry, setSelectedCountry] = useState<string>("Egypt");
    const [countrySearch, setCountrySearch] = useState("");
    const [isCountryOpen, setIsCountryOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [symbols, setSymbols] = useState<SymbolResult[]>([]);
    const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(false);
    const [displayLimit, setDisplayLimit] = useState(50);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const doSearch = useCallback(() => {
        // No-op: handled by local filtering of syncedSymbols
        setLoading(false);
    }, []);

    const handleLoadMore = () => {
        setDisplayLimit(prev => prev + 100);
    };

    // Initial load and reset
    useEffect(() => {
        if (isOpen) {
            // Only force refresh if empty
            if (countries.length === 0) {
                void refreshCountries({ force: true });
            }
            if (selectedCountry) {
                void refreshSyncedSymbols(selectedCountry);
            }
            setSearchQuery("");
            setSelectedSymbols(new Set());
            setDisplayLimit(100);
        }
    }, [isOpen, countries.length, refreshCountries, selectedCountry, refreshSyncedSymbols]);

    // Local filtering
    useEffect(() => {
        if (!syncedSymbols) {
            setSymbols([]);
            return;
        }
        const q = searchQuery.toLowerCase().trim();
        const filtered = syncedSymbols.filter(s => {
            const matchesQuery = !q || s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q);
            const isFund = s.symbol.toLowerCase().startsWith("fund_");

            if (isAdmin) return matchesQuery;

            return matchesQuery && !isFund;
        });
        setSymbols(filtered.slice(0, displayLimit));
    }, [syncedSymbols, searchQuery, displayLimit, isAdmin]);

    const getFullSymbol = (s: SymbolResult) => {
        if (s.symbol.includes(".")) return s.symbol;
        if (!s.exchange) return s.symbol;

        // Map exchange names to standard suffixes if needed
        const mapping: Record<string, string> = {
            "EGX": "EGX",
            "NYSE": "US",
            "NASDAQ": "US",
            "LSE": "LSE",
            "XETRA": "XETRA",
            "PA": "PA"
        };
        const suffix = mapping[s.exchange.toUpperCase()] || s.exchange;
        return `${s.symbol}.${suffix}`;
    };

    const handleSelect = (s: SymbolResult) => {
        const full = getFullSymbol(s);

        // Single selection mode: immediately select
        if (!effectiveMultiSelect) {
            setSelectedSymbols(new Set([full]));
            return;
        }

        // Multiple selection mode: toggle selection
        const next = new Set(selectedSymbols);
        if (next.has(full)) {
            next.delete(full);
        } else {
            next.add(full);
        }
        setSelectedSymbols(next);
    };

    const handleSelectAll = () => {
        if (selectedSymbols.size === symbols.length && symbols.length > 0) {
            setSelectedSymbols(new Set());
        } else {
            setSelectedSymbols(new Set(symbols.map(s => getFullSymbol(s))));
        }
    };

    const handleConfirm = () => {
        onSelect(Array.from(selectedSymbols));
        onClose();
    };

    if (!isOpen) return null;

    const filteredCountries = countries.filter(c =>
        (isAdmin || isCountryActive(c)) && c.toLowerCase().includes(countrySearch.toLowerCase())
    );

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

            <div className="relative w-full max-w-2xl bg-zinc-950 border border-zinc-800 rounded-2xl shadow-2xl flex flex-col max-h-[85vh] overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50">
                    <div>
                        <h3 className="text-lg font-bold text-white uppercase tracking-tight">Browse Symbols</h3>
                        <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5">Select symbols to analyze</p>
                    </div>
                    <button onClick={onClose} className="p-2 rounded-xl text-zinc-500 hover:text-white hover:bg-white/5 transition-all">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Search & Filters */}
                <div className="p-4 bg-zinc-900/30 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Custom Country Dropdown */}
                        <div className="relative">
                            <label className="block text-[10px] uppercase tracking-widest font-bold text-zinc-500 mb-1.5 ml-1">Market / Country</label>
                            <button
                                onClick={() => setIsCountryOpen(!isCountryOpen)}
                                className="w-full h-11 px-4 flex items-center justify-between rounded-xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-all group"
                            >
                                <div className="flex items-center gap-3">
                                    <Globe className="w-4 h-4 text-blue-500" />
                                    <span className="text-sm font-medium text-zinc-200">{selectedCountry || "Select Country"}</span>
                                    {selectedCountry && (
                                        <div className="flex items-center gap-2.5">
                                            <span className="text-zinc-700 font-light">|</span>
                                            {(() => {
                                                const stats = inventory.filter(i => i.country === selectedCountry);
                                                const totalLoaded = stats.reduce((acc, s) => acc + s.priceCount, 0);
                                                const totalExpected = stats.reduce((acc, s) => acc + s.expectedCount, 0);
                                                if (totalExpected === 0) return null;
                                                const isFullySynced = totalLoaded >= totalExpected && totalExpected > 0;
                                                return (
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-[11px] font-black font-mono tracking-tighter ${isFullySynced ? 'text-emerald-400' : totalLoaded > 0 ? 'text-zinc-300' : 'text-zinc-600'}`}>
                                                            {totalLoaded} <span className="text-[10px] text-zinc-700">/</span> {totalExpected}
                                                        </span>
                                                        <div className="flex items-center gap-1 opacity-60">
                                                            <div className="w-1.5 h-1.5 rounded-full bg-zinc-800 overflow-hidden relative">
                                                                <div
                                                                    className={`absolute inset-0 transition-all duration-1000 ${isFullySynced ? 'bg-emerald-500' : 'bg-blue-500'}`}
                                                                    style={{ width: `${(totalLoaded / totalExpected) * 100}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })()}
                                        </div>
                                    )}
                                </div>
                                <Loader2 className={`w-3 h-3 animate-spin text-zinc-600 ${countriesLoading ? "block" : "hidden"}`} />
                            </button>

                            {isCountryOpen && (
                                <div className="absolute top-full left-0 right-0 mt-2 z-50 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl overflow-hidden p-2 animate-in fade-in slide-in-from-top-2 duration-200">
                                    <input
                                        type="text"
                                        placeholder="Search countries..."
                                        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500 mb-2"
                                        value={countrySearch}
                                        onChange={(e) => setCountrySearch(e.target.value)}
                                        onClick={(e) => e.stopPropagation()}
                                    />
                                    <div className="max-h-48 overflow-y-auto custom-scrollbar">
                                        {filteredCountries.map(c => {
                                            const stats = inventory.filter(i => i.country === c);
                                            const totalLoaded = stats.reduce((acc, s) => acc + s.priceCount, 0);
                                            const totalExpected = stats.reduce((acc, s) => acc + s.expectedCount, 0);
                                            const isFullySynced = totalLoaded >= totalExpected && totalExpected > 0;
                                            const percent = totalExpected > 0 ? (totalLoaded / totalExpected) * 100 : 0;

                                            return (
                                                <button
                                                    key={c}
                                                    onClick={() => {
                                                        setSelectedCountry(c);
                                                        setIsCountryOpen(false);
                                                    }}
                                                    className={`w-full group/item flex flex-col gap-1.5 px-3 py-2.5 rounded-lg transition-all ${selectedCountry === c ? "bg-blue-600" : "hover:bg-zinc-800"}`}
                                                >
                                                    <div className="w-full flex items-center justify-between">
                                                        <span className={`text-xs font-bold ${selectedCountry === c ? "text-white" : "text-zinc-400 group-hover/item:text-zinc-200"}`}>{c}</span>
                                                        {totalExpected > 0 && (
                                                            <div className="flex items-center gap-2">
                                                                <span className={`text-[10px] font-black font-mono tracking-tight ${selectedCountry === c ? "text-blue-100" : isFullySynced ? "text-emerald-500/80" : "text-zinc-600 group-hover/item:text-zinc-500"}`}>
                                                                    {totalLoaded} <span className="opacity-40">/</span> {totalExpected}
                                                                </span>
                                                            </div>
                                                        )}
                                                    </div>
                                                    {totalExpected > 0 && (
                                                        <div className={`h-1 w-full rounded-full overflow-hidden ${selectedCountry === c ? "bg-blue-700/50" : "bg-zinc-900"}`}>
                                                            <div
                                                                className={`h-full transition-all duration-700 ${selectedCountry === c ? 'bg-white/40' : isFullySynced ? 'bg-emerald-500/40' : 'bg-zinc-800'}`}
                                                                style={{ width: `${percent}%` }}
                                                            />
                                                        </div>
                                                    )}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Search Input */}
                        <div className="relative">
                            <label className="block text-[10px] uppercase tracking-widest font-bold text-zinc-500 mb-1.5 ml-1">Search Symbol / Name</label>
                            <div className="flex gap-2">
                                <div className="relative flex-1">
                                    <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600" />
                                    <input
                                        type="text"
                                        placeholder="e.g. AAPL, COMI..."
                                        className="w-full h-11 pl-10 pr-4 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500 transition-all"
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        onKeyDown={(e) => e.key === "Enter" && void doSearch()}
                                    />
                                </div>
                                <button
                                    onClick={() => void doSearch()}
                                    disabled={loading}
                                    className="h-11 px-6 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-white font-bold text-[10px] uppercase tracking-widest transition-all disabled:opacity-50"
                                >
                                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Symbols List */}
                <div className="flex-1 overflow-y-auto px-4 py-2 custom-scrollbar max-h-[400px]">
                    {syncedSymbolsLoading && symbols.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center gap-3 text-zinc-600 py-20">
                            <Loader2 className="w-8 h-8 animate-spin" />
                            <p className="text-xs uppercase tracking-[0.2em] font-bold">Scanning Market Data...</p>
                        </div>
                    ) : symbols.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center gap-3 text-zinc-700 py-20 italic">
                            <p className="text-sm">No symbols found for this market.</p>
                            <p className="text-[10px] uppercase tracking-widest not-italic">Try a different name or search term</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-1 pb-4">
                            {/* Select All Button - Only show in multi-select mode */}
                            {effectiveMultiSelect && (
                                <button
                                    onClick={handleSelectAll}
                                    className="flex items-center gap-3 px-4 py-3 rounded-xl border border-dashed border-zinc-800 hover:bg-zinc-900/50 hover:border-zinc-700 transition-all mb-2 text-zinc-500 hover:text-blue-400 group"
                                >
                                    {selectedSymbols.size === symbols.length && symbols.length > 0 ? (
                                        <CheckSquare className="w-5 h-5 text-blue-500" />
                                    ) : (
                                        <Square className="w-5 h-5" />
                                    )}
                                    <span className="text-xs font-bold uppercase tracking-widest">
                                        {selectedSymbols.size === symbols.length ? "Deselect All" : `Select All ${symbols.length} Symbols`}
                                    </span>
                                </button>
                            )}

                            {symbols.map(s => {
                                const full = getFullSymbol(s);
                                const isSelected = selectedSymbols.has(full);
                                return (
                                    <button
                                        key={s.symbol}
                                        onClick={() => handleSelect(s)}
                                        className={`group flex items-center gap-4 px-4 py-3 rounded-xl transition-all border ${isSelected
                                            ? "bg-blue-600/10 border-blue-500/30"
                                            : "border-transparent hover:bg-white/5"
                                            }`}
                                    >
                                        <div className={`relative w-5 h-5 rounded border transition-all flex items-center justify-center ${isSelected
                                            ? "bg-blue-500 border-blue-500"
                                            : "bg-transparent border-zinc-700 group-hover:border-zinc-500"
                                            }`}>
                                            {isSelected && <X className="w-3.5 h-3.5 text-white stroke-[3px]" />}
                                        </div>

                                        <div className="flex-1 min-w-0 text-left">
                                            <div className="flex items-center gap-2">
                                                <span className={`font-mono font-bold transition-colors ${isSelected ? "text-blue-400" : "text-white"}`}>{s.symbol}</span>
                                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-medium uppercase tracking-tighter">{s.exchange}</span>
                                            </div>
                                            <div className="text-xs text-zinc-500 truncate mt-0.5 group-hover:text-zinc-400 transition-colors">
                                                {s.name}
                                            </div>
                                        </div>

                                        <div className="flex-shrink-0 flex items-center gap-4">
                                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-widest ${s.hasLocal ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20" : "text-zinc-600"}`}>
                                                {s.hasLocal ? "Cached" : ""}
                                            </span>
                                        </div>
                                    </button>
                                );
                            })}

                            {/* Load More Button */}
                            {symbols.length >= displayLimit && (
                                <button
                                    onClick={handleLoadMore}
                                    disabled={loading}
                                    className="w-full py-4 mt-2 rounded-xl border border-dashed border-zinc-800 hover:bg-zinc-900/50 hover:border-zinc-700 transition-all text-zinc-500 hover:text-white flex items-center justify-center gap-2 group disabled:opacity-50"
                                >
                                    {loading ? (
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : (
                                        <>
                                            <span className="text-[10px] font-black uppercase tracking-[0.2em]">Load More Symbols</span>
                                        </>
                                    )}
                                </button>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer / Confirm Actions */}
                <div className="border-t border-zinc-800 p-4 bg-zinc-900 flex items-center justify-between gap-4 shadow-[0_-10px_20px_-10px_rgba(0,0,0,0.5)]">
                    <div className="flex flex-col">
                        <p className="text-[9px] text-zinc-500 font-medium uppercase tracking-[0.2em] mt-0.5 sm:block hidden">
                            {loading ? "Refreshing..." : `${symbols.length} listings found`}
                        </p>
                    </div>
                    <div className="flex items-center gap-3 w-full sm:w-auto">
                        <button
                            onClick={onClose}
                            className="flex-1 sm:flex-none h-11 px-6 rounded-xl text-xs font-black uppercase tracking-widest text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-all"
                        >
                            Cancel
                        </button>
                        <button
                            disabled={selectedSymbols.size === 0}
                            onClick={handleConfirm}
                            className="flex-1 sm:flex-none h-11 px-10 rounded-xl bg-blue-600 text-xs font-black uppercase tracking-widest text-white hover:bg-blue-500 disabled:opacity-30 disabled:grayscale transition-all shadow-xl shadow-blue-500/20 relative group overflow-hidden"
                        >
                            <span className="relative z-10">Add Symbol</span>
                            <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
