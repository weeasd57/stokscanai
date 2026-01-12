"use client";

import { X, Search, Globe } from "lucide-react";
import { useState, useMemo } from "react";
import { useLanguage } from "@/contexts/LanguageContext";

interface CountrySelectDialogProps {
    open: boolean;
    onClose: () => void;
    onSelect: (country: string) => void;
    countries: string[];
    selectedCountry: string;
}

export default function CountrySelectDialog({
    open,
    onClose,
    onSelect,
    countries,
    selectedCountry,
}: CountrySelectDialogProps) {
    const { t } = useLanguage();
    const [search, setSearch] = useState("");

    const filteredCountries = useMemo(() => {
        if (!search) return countries;
        return countries.filter((c) =>
            c.toLowerCase().includes(search.toLowerCase())
        );
    }, [countries, search]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-md overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
                <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
                    <h2 className="text-lg font-semibold text-zinc-100">Select Country</h2>
                    <button
                        onClick={onClose}
                        className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <div className="p-4 border-b border-zinc-800">
                    <div className="relative">
                        <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
                        <input
                            type="text"
                            placeholder="Search..."
                            className="h-10 w-full rounded-lg border border-zinc-800 bg-zinc-900 pl-9 pr-4 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-indigo-500 focus:outline-none"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            autoFocus
                        />
                    </div>
                </div>

                <div className="max-h-[60vh] overflow-y-auto p-2">
                    {filteredCountries.length === 0 ? (
                        <div className="py-8 text-center text-sm text-zinc-500">
                            No countries found.
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-1">
                            {filteredCountries.map((country) => (
                                <button
                                    key={country}
                                    onClick={() => {
                                        onSelect(country);
                                        onClose();
                                    }}
                                    className={`flex items-center justify-between rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${country === selectedCountry
                                            ? "bg-indigo-500/10 text-indigo-400"
                                            : "text-zinc-300 hover:bg-zinc-900 hover:text-zinc-100"
                                        }`}
                                >
                                    <div className="flex items-center gap-3">
                                        <Globe className="h-4 w-4 opacity-50" />
                                        {country}
                                    </div>
                                    {country === selectedCountry && (
                                        <div className="h-2 w-2 rounded-full bg-indigo-500" />
                                    )}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
