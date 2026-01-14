"use client";

import { useState } from "react";
import { Plus, Search, ArrowLeftRight, Activity, TrendingUp, LogIn } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { useAppState } from "@/contexts/AppStateContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import CompareTableView from "@/components/CompareTableView";
import CountrySymbolDialog from "@/components/CountrySymbolDialog";
import IndicatorStatistics from "@/components/IndicatorStatistics";
import ChartDialog from "@/components/ChartDialog";

export default function ComparisonScannerPage() {
    const { t } = useLanguage();
    const {
        state,
        addSymbolsToCompare,
        removeSymbolFromCompare,
        clearComparison,
        setComparisonScanner
    } = useAppState();
    const { user } = useAuth();
    const router = useRouter();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();

    const { results, loadingSymbols, errors } = state.comparisonScanner;

    const [searchTerm, setSearchTerm] = useState("");
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

    const handleAddManual = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchTerm.trim()) {
            void addSymbolsToCompare([searchTerm.trim()]);
            setSearchTerm("");
        }
    };

    const handleSave = (symbol: string) => {
        if (!user) {
            router.push("/login?redirect=/scanner/comparison");
            return;
        }

        if (isSaved(symbol)) {
            removeSymbolBySymbol(symbol);
            return;
        }

        // Get metadata from comparison results if available
        const data = results[symbol];
        const fundamentals = data?.fundamentals as any;
        const metadata = data ? {
            price: fundamentals?.last_close ?? fundamentals?.close,
            name: fundamentals?.name,
            signal: data.signal,
            precision: data.precision
        } : {};

        saveSymbol({
            symbol: symbol.toUpperCase(),
            name: fundamentals?.name || symbol,
            source: "tech_scanner" as const,
            metadata,
            entryPrice: fundamentals?.last_close ?? fundamentals?.close ?? null,
        });
    };

    return (
        <div className="flex flex-col gap-10 max-w-[1600px] mx-auto pb-20 p-4 md:p-8">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div className="space-y-2">
                    <div className="flex items-center gap-3">
                        <div className="p-3 rounded-2xl bg-blue-600 shadow-xl shadow-blue-600/20">
                            <ArrowLeftRight className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-black tracking-tight text-white uppercase italic">
                            Comparison <span className="text-blue-500">Scanner</span>
                        </h1>
                    </div>
                    <p className="text-sm text-zinc-500 font-medium max-w-lg">
                        Analyze and compare technical indicator performance and AI win rates across multiple assets simultaneously.
                    </p>
                </div>
            </div>


            {/* Indicator Statistics Section */}
            {Object.keys(results).length > 0 && (
                <section className="animate-in fade-in slide-in-from-top-4 duration-700">
                    <IndicatorStatistics results={results} />
                </section>
            )}

            {/* Controls Section */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center bg-zinc-950/20 p-2 rounded-[2.5rem] border border-white/5 backdrop-blur-sm">
                <form onSubmit={handleAddManual} className="lg:col-span-10 flex gap-2">
                    <div className="relative flex-1">
                        <Search className="absolute left-5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-600" />
                        <input
                            type="text"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            placeholder={t("ticker.placeholder") || "Add symbol..."}
                            className="w-full h-14 pl-14 pr-6 rounded-[2rem] border border-white/5 bg-zinc-900/50 text-sm text-zinc-200 outline-none focus:ring-1 focus:ring-blue-500/30 transition-all font-mono placeholder:text-zinc-700"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault();
                                    if (searchTerm.trim()) {
                                        void addSymbolsToCompare([searchTerm.trim()]);
                                        setSearchTerm("");
                                    }
                                }
                            }}
                        />
                    </div>
                </form>

                <div className="lg:col-span-2">
                    <button
                        type="button"
                        onClick={() => setIsDialogOpen(true)}
                        className="w-full h-14 px-6 rounded-[2rem] border border-white/5 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 font-black text-[11px] uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 active:scale-95"
                    >
                        {t("btn.browse")}
                    </button>
                </div>
            </div>

            {/* Results Table */}
            <section className="animate-in fade-in slide-in-from-bottom-8 duration-1000">
                <div className="flex justify-end mb-4">
                    {Object.keys(results).length > 0 && (
                        <button
                            onClick={clearComparison}
                            className="text-xs text-red-400 hover:text-red-300 transition-colors uppercase tracking-wider font-bold"
                        >
                            CLEAR ALL
                        </button>
                    )}
                </div>

                <CompareTableView
                    results={results}
                    loadingSymbols={loadingSymbols}
                    errors={errors}
                    onRemove={removeSymbolFromCompare}
                    onSave={handleSave}
                    onChart={setSelectedSymbol}
                    isSaved={isSaved}
                />
            </section>

            <CountrySymbolDialog
                isOpen={isDialogOpen}
                onClose={() => setIsDialogOpen(false)}
                onSelect={(syms) => {
                    void addSymbolsToCompare(syms);
                    setIsDialogOpen(false);
                }}
                multiSelect={true}
            />

            <ChartDialog
                symbol={selectedSymbol}
                onClose={() => setSelectedSymbol(null)}
            />

        </div>
    );
}
