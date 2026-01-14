"use client";

import { X, Search, Globe, ChevronRight } from "lucide-react";
import { useState, useMemo, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAppState } from "@/contexts/AppStateContext";

interface CountrySelectDialogProps {
    open: boolean;
    onClose: () => void;
    onSelect: (country: string) => void;
    countries: string[];
    selectedCountry: string;
    forcedAdmin?: boolean;
}

export default function CountrySelectDialog({
    open,
    onClose,
    onSelect,
    countries,
    selectedCountry,
    forcedAdmin = false,
}: CountrySelectDialogProps) {
    const { t } = useLanguage();
    const { isCountryActive, isAdmin: globalAdmin } = useAppState();
    const isAdmin = forcedAdmin || globalAdmin;
    const [search, setSearch] = useState("");

    const filteredCountries = useMemo(() => {
        let list = countries;
        if (!isAdmin) {
            list = list.filter(c => isCountryActive(c));
        }

        if (!search) return list;
        return list.filter((c) =>
            c.toLowerCase().includes(search.toLowerCase())
        );
    }, [countries, search, isCountryActive, isAdmin]);

    // Prevent body scroll when open
    useEffect(() => {
        if (open) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "unset";
        }
        return () => {
            document.body.style.overflow = "unset";
        };
    }, [open]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-md animate-in fade-in duration-300"
                onClick={onClose}
            />

            {/* Dialog Container */}
            <div className="relative w-full max-w-2xl overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-950/90 shadow-[0_0_50px_-12px_rgba(79,70,229,0.3)] animate-in fade-in zoom-in-95 duration-300 flex flex-col max-h-[85vh]">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-zinc-900/50 px-6 py-5 bg-gradient-to-r from-zinc-950 to-zinc-900">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                            <Globe className="h-5 w-5" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-zinc-100 tracking-tight">Select Market</h2>
                            <p className="text-xs text-zinc-500 font-medium">Choose a country to manage symbols</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-full p-2.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-100 transition-all active:scale-95"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                {/* Search Bar */}
                <div className="p-6 bg-zinc-950/50">
                    <div className="relative group">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-600 group-focus-within:text-indigo-400 transition-colors" />
                        <input
                            type="text"
                            placeholder="Search countries..."
                            className="h-14 w-full rounded-2xl border border-zinc-800 bg-zinc-900/50 pl-12 pr-4 text-base text-zinc-100 placeholder:text-zinc-600 focus:border-indigo-500/50 focus:ring-4 focus:ring-indigo-500/10 focus:outline-none transition-all"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            autoFocus
                        />
                        {search && (
                            <button
                                onClick={() => setSearch("")}
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-xs font-bold text-zinc-500 hover:text-zinc-300"
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </div>

                {/* Country Grid */}
                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                    {filteredCountries.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div className="h-16 w-16 rounded-full bg-zinc-900 flex items-center justify-center mb-4 border border-zinc-800">
                                <Search className="h-6 w-6 text-zinc-700" />
                            </div>
                            <h3 className="text-zinc-400 font-semibold text-lg">No results found</h3>
                            <p className="text-zinc-600 text-sm max-w-[200px]">Try searching with a different country name.</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-2 pb-6">
                            {filteredCountries.map((country) => {
                                const isSelected = country === selectedCountry;
                                return (
                                    <button
                                        key={country}
                                        onClick={() => {
                                            onSelect(country);
                                            onClose();
                                        }}
                                        className={`group relative flex items-center justify-between rounded-2xl px-5 py-4 text-sm font-semibold transition-all border ${isSelected
                                            ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400"
                                            : "bg-zinc-900/30 border-transparent text-zinc-400 hover:bg-zinc-800/80 hover:border-zinc-700 hover:text-zinc-100"
                                            }`}
                                    >
                                        <div className="flex items-center gap-4">
                                            <div className={`p-2 rounded-lg transition-colors ${isSelected ? "bg-indigo-500/20" : "bg-zinc-900 group-hover:bg-zinc-700"
                                                }`}>
                                                <Globe className={`h-4 w-4 ${isSelected ? "text-indigo-400" : "text-zinc-600 group-hover:text-zinc-400"}`} />
                                            </div>
                                            <span className="tracking-tight">{country}</span>
                                        </div>

                                        {isSelected ? (
                                            <div className="flex items-center gap-2">
                                                <span className="text-[10px] font-black uppercase text-indigo-500/60 tracking-widest">Active</span>
                                                <div className="h-2 w-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
                                            </div>
                                        ) : isAdmin && !isCountryActive(country) ? (
                                            <div className="flex items-center gap-2">
                                                <span className="text-[10px] font-black uppercase text-zinc-600 tracking-widest whitespace-nowrap">Empty / New</span>
                                                <div className="h-2 w-2 rounded-full bg-zinc-700" />
                                            </div>
                                        ) : (
                                            <ChevronRight className="h-4 w-4 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-zinc-600" />
                                        )}
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
