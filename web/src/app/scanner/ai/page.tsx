"use client";

import { useMemo, useState } from "react";
import { Brain, AlertTriangle, Loader2, Globe, Bookmark, BookmarkCheck, Info, X, BarChart2, LineChart, Sliders, Activity } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";
import { predictStock, type ScanResult } from "@/lib/api";
import CountrySelectDialog from "@/components/CountrySelectDialog";
import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import StockLogo from "@/components/StockLogo";

export default function AIScannerPage() {
    const { t } = useLanguage();
    const { saveSymbol, removeSymbolBySymbol, isSaved } = useWatchlist();
    const { state, countries, setAiScanner, runAiScan, stopAiScan, aiScanLoading: loading, aiScanError: error, clearAiScannerView, restoreLastAiScan } = useAppState();

    const { country, scanAll, results, progress, hasScanned, showPrecisionInfo, selected, detailData, rfPreset, rfParamsJson, chartType, showEma50, showEma200, showBB, showRsi, showVolume, scanHistory } = state.aiScanner;
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [showAdvancedRf, setShowAdvancedRf] = useState(false);
    const [showCountryDialog, setShowCountryDialog] = useState(false);

    function mergeRfJson(nextPartial: Record<string, unknown>) {
        setAiScanner((prev) => {
            let base: Record<string, unknown> = {};
            try {
                const parsed = JSON.parse((prev as any).rfParamsJson || "{}");
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                    base = parsed;
                }
            } catch {
            }
            return { ...prev, rfParamsJson: JSON.stringify({ ...base, ...nextPartial }) };
        });
    }

    const quickRf = useMemo(() => {
        const defaults = {
            n_estimators: 80,
            max_depth: 10,
            min_samples_leaf: 3,
            min_samples_split: 10,
            max_features: "sqrt" as "sqrt" | "log2" | "auto",
        };

        try {
            const v = JSON.parse(rfParamsJson || "{}");
            if (!v || typeof v !== "object" || Array.isArray(v)) return defaults;
            return {
                n_estimators: Number.isFinite(Number((v as any).n_estimators)) ? Number((v as any).n_estimators) : defaults.n_estimators,
                max_depth: (v as any).max_depth === null || (v as any).max_depth === "" ? null : (Number.isFinite(Number((v as any).max_depth)) ? Number((v as any).max_depth) : defaults.max_depth),
                min_samples_leaf: Number.isFinite(Number((v as any).min_samples_leaf)) ? Number((v as any).min_samples_leaf) : defaults.min_samples_leaf,
                min_samples_split: Number.isFinite(Number((v as any).min_samples_split)) ? Number((v as any).min_samples_split) : defaults.min_samples_split,
                max_features: (typeof (v as any).max_features === "string" ? (v as any).max_features : defaults.max_features) as any,
            };
        } catch {
            return defaults;
        }
    }, [rfParamsJson]);

    const parsedRfParams = useMemo(() => {
        try {
            const v = JSON.parse(rfParamsJson || "{}");
            if (!v || typeof v !== "object" || Array.isArray(v)) return null;
            return v as Record<string, unknown>;
        } catch {
            return null;
        }
    }, [rfParamsJson]);

    const effectiveRfParams = useMemo(() => {
        if (showAdvancedRf) return parsedRfParams;
        return {
            n_estimators: Math.max(1, Math.min(2000, Math.round(quickRf.n_estimators))),
            max_depth: quickRf.max_depth === null ? null : Math.max(1, Math.min(256, Math.round(quickRf.max_depth))),
            min_samples_leaf: Math.max(1, Math.min(10000, Math.round(quickRf.min_samples_leaf))),
            min_samples_split: Math.max(2, Math.min(10000, Math.round(quickRf.min_samples_split))),
            max_features: quickRf.max_features === "auto" ? "sqrt" : quickRf.max_features,
            n_jobs: -1,
        } as Record<string, unknown>;
    }, [parsedRfParams, quickRf, showAdvancedRf]);

    async function openDetails(row: ScanResult) {
        setAiScanner((prev) => ({ ...prev, selected: row, detailData: null }));
        setDetailError(null);
        setDetailLoading(true);
        try {
            const res = await predictStock({
                ticker: row.symbol,
                exchange: row.exchange ?? undefined,
                includeFundamentals: false,
                rfPreset: rfPreset,
                rfParams: effectiveRfParams ?? undefined,
            });
            setAiScanner((prev) => ({ ...prev, detailData: res }));
        } catch (err) {
            setDetailError(err instanceof Error ? err.message : "Failed to load chart");
        } finally {
            setDetailLoading(false);
        }
    }

    async function runScan() {
        if (showAdvancedRf && rfParamsJson && parsedRfParams === null) {
            return;
        }
        await runAiScan({ rfParams: effectiveRfParams ?? null, minPrecision: 0.6 });
    }

    return (
        <div className="flex flex-col gap-8 pb-20 max-w-[1600px] mx-auto">
            <header className="flex flex-col gap-2">
                <h1 className="text-3xl font-black tracking-tighter text-white flex items-center gap-4 uppercase italic">
                    <div className="p-3 rounded-2xl bg-indigo-600 shadow-xl shadow-indigo-600/20">
                        <Brain className="h-6 w-6 text-white" />
                    </div>
                    {t("ai.title")}
                </h1>
                <p className="text-sm text-zinc-500 font-medium max-w-2xl">
                    {t("ai.subtitle")}
                </p>
            </header>


            <div className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl p-8 shadow-2xl overflow-hidden relative">
                {/* Background Decor */}
                <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-600/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2" />

                <div className="flex flex-col md:flex-row md:items-center justify-between mb-10 gap-6 relative z-10">
                    <div className="flex flex-col gap-3">
                        <h2 className="text-xs font-black text-zinc-400 uppercase tracking-[0.3em]">Market Parameters</h2>
                        <div className="flex flex-wrap items-center gap-6">
                            <button
                                onClick={() => setShowCountryDialog(true)}
                                disabled={loading}
                                className="h-12 flex items-center gap-3 rounded-2xl border border-white/5 bg-zinc-900/50 px-5 text-sm font-bold text-zinc-200 hover:bg-zinc-800 transition-all group"
                            >
                                <Globe className="h-4 w-4 text-blue-500 opacity-60 group-hover:rotate-12 transition-transform" />
                                <span className="uppercase tracking-widest">{country}</span>
                            </button>

                            <label className="flex items-center gap-3 text-xs font-black uppercase tracking-widest text-zinc-400 cursor-pointer group hover:text-white transition-all">
                                <div className={`w-5 h-5 rounded-lg border border-white/10 flex items-center justify-center transition-all ${scanAll ? "bg-indigo-600 border-indigo-600 shadow-lg shadow-indigo-600/30" : "bg-zinc-800"}`}>
                                    {scanAll && <X className="w-3 h-3 text-white stroke-[3px]" />}
                                </div>
                                <input
                                    type="checkbox"
                                    className="hidden"
                                    checked={scanAll}
                                    onChange={e => setAiScanner(prev => ({ ...prev, scanAll: e.target.checked }))}
                                    disabled={loading}
                                />
                                <span>{t("ai.scan_all")}</span>
                            </label>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        {results.length > 0 && (
                            <button
                                type="button"
                                onClick={() => clearAiScannerView()}
                                disabled={loading}
                                className="h-12 px-6 rounded-2xl border border-white/5 bg-zinc-900/50 text-[10px] font-black uppercase tracking-widest text-zinc-400 hover:text-white transition-all"
                            >
                                {t("tech.clear_results")}
                            </button>
                        )}

                        {results.length === 0 && scanHistory.length > 0 && (
                            <button
                                type="button"
                                onClick={() => void restoreLastAiScan()}
                                disabled={loading}
                                className="h-12 px-6 rounded-2xl border border-white/5 bg-zinc-900/50 text-[10px] font-black uppercase tracking-widest text-zinc-400 hover:text-white transition-all"
                            >
                                {t("tech.restore_last")}
                            </button>
                        )}

                        {loading ? (
                            <button
                                onClick={stopAiScan}
                                className="h-12 flex items-center gap-3 rounded-2xl bg-red-600/10 border border-red-500/20 px-8 text-[11px] font-black uppercase tracking-widest text-red-500 hover:bg-red-500/20 transition-all"
                            >
                                <span className="h-2 w-2 rounded-full bg-red-500 animate-ping" />
                                {t("ai.stop_scan")}
                            </button>
                        ) : (
                            <button
                                onClick={runScan}
                                className="h-12 flex items-center gap-3 rounded-2xl bg-indigo-600 px-10 text-[11px] font-black uppercase tracking-[0.2em] text-white shadow-2xl shadow-indigo-600/30 hover:bg-indigo-500 transition-all group active:scale-95"
                            >
                                <Brain className="h-4 w-4 group-hover:scale-110 transition-transform" />
                                {t("ai.start_scan")}
                            </button>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 relative z-10">
                    <div className="rounded-2xl border border-white/5 bg-zinc-900/30 p-5 space-y-3">
                        <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{t("ai.model_preset")}</div>
                        <div className="relative">
                            <select
                                value={rfPreset}
                                onChange={(e) => setAiScanner(prev => ({ ...prev, rfPreset: e.target.value as any }))}
                                disabled={loading}
                                className="w-full h-11 rounded-xl bg-zinc-950 border border-white/5 px-4 text-xs font-bold text-zinc-200 uppercase tracking-widest outline-none appearance-none hover:bg-zinc-900 transition-all"
                            >
                                <option value="fast">LIGHT (FAST)</option>
                                <option value="default">BALANCED</option>
                                <option value="accurate">DEEP (ACCURATE)</option>
                            </select>
                            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-[8px] text-zinc-600">▼</div>
                        </div>
                    </div>

                    <div className="md:col-span-2 rounded-2xl border border-white/5 bg-zinc-900/30 p-5">
                        <div className="flex items-center justify-between gap-3 mb-4">
                            <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{t("ai.model_options")}</div>
                            <button
                                type="button"
                                onClick={() => setShowAdvancedRf((v) => !v)}
                                disabled={loading}
                                className={`text-[9px] font-black uppercase tracking-widest px-3 py-1.5 rounded-lg border transition-all ${showAdvancedRf ? "bg-white text-black border-white" : "text-zinc-400 border-white/5 hover:border-zinc-700 hover:text-zinc-200"}`}
                            >
                                {showAdvancedRf ? "Developer Mode" : "Simple Mode"}
                            </button>
                        </div>

                        {!showAdvancedRf ? (
                            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                                {[
                                    { label: "Trees", val: quickRf.n_estimators, key: "n_estimators" },
                                    { label: "Max Depth", val: quickRf.max_depth === null ? "Auto" : quickRf.max_depth, key: "max_depth" },
                                    { label: "Min Leaf", val: quickRf.min_samples_leaf, key: "min_samples_leaf" },
                                    { label: "Min Split", val: quickRf.min_samples_split, key: "min_samples_split" },
                                ].map((item) => (
                                    <div key={item.label} className="space-y-1.5">
                                        <div className="text-[9px] font-black text-zinc-600 uppercase tracking-tighter">{item.label}</div>
                                        <input
                                            type="text"
                                            value={item.val}
                                            onChange={(e) => {
                                                const v = e.target.value === "" || e.target.value === "Auto" ? null : Number(e.target.value);
                                                mergeRfJson({ [item.key]: v });
                                            }}
                                            disabled={loading}
                                            className="w-full h-9 rounded-lg bg-zinc-950 border border-white/5 px-3 text-[11px] font-mono font-bold text-zinc-300 focus:border-indigo-500/50 outline-none transition-all"
                                        />
                                    </div>
                                ))}
                                <div className="space-y-1.5">
                                    <div className="text-[9px] font-black text-zinc-600 uppercase tracking-tighter">Features</div>
                                    <select
                                        value={quickRf.max_features}
                                        onChange={(e) => mergeRfJson({ max_features: e.target.value })}
                                        disabled={loading}
                                        className="w-full h-9 rounded-lg bg-zinc-950 border border-white/5 px-2 text-[10px] font-black uppercase tracking-widest text-zinc-300 outline-none"
                                    >
                                        <option value="sqrt">SQRT</option>
                                        <option value="log2">LOG2</option>
                                        <option value="auto">AUTO</option>
                                    </select>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <div className="text-[9px] font-black text-zinc-600 uppercase tracking-widest">RF_PARAMS_JSON</div>
                                    <button
                                        type="button"
                                        onClick={() => setAiScanner(prev => ({ ...prev, rfParamsJson: "{}" }))}
                                        disabled={loading}
                                        className="text-[9px] font-black uppercase tracking-widest text-indigo-400 hover:text-indigo-300 transition-colors"
                                    >
                                        Reset to defaults
                                    </button>
                                </div>
                                <textarea
                                    value={rfParamsJson}
                                    onChange={(e) => setAiScanner(prev => ({ ...prev, rfParamsJson: e.target.value }))}
                                    disabled={loading}
                                    rows={3}
                                    className={`w-full resize-none rounded-xl border p-4 text-[10px] font-mono tracking-tight leading-relaxed transition-all outline-none ${parsedRfParams === null ? "border-red-500/30 bg-red-900/5 text-red-300" : "border-white/5 bg-zinc-950 text-emerald-400/80"}`}
                                />
                            </div>
                        )}
                    </div>
                </div>

                {loading && (
                    <div className="flex flex-col items-center justify-center py-6 gap-6 relative z-10 animate-in fade-in zoom-in-95 duration-500">
                        <div className="w-full max-w-xl bg-zinc-900/50 rounded-full h-1.5 overflow-hidden border border-white/5">
                            <div
                                className="bg-indigo-500 h-full transition-all duration-700 ease-out shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                                style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
                            />
                        </div>
                        <div className="flex items-center gap-4 bg-zinc-900/50 px-6 py-3 rounded-2xl border border-white/5 backdrop-blur-md">
                            <div className="relative">
                                <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
                                <div className="absolute inset-0 blur-lg bg-indigo-500/50 animate-pulse" />
                            </div>
                            <p className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-300">
                                Analyzing {country}: <span className="text-white font-mono">{progress.current} / {progress.total}</span>
                            </p>
                        </div>
                    </div>
                )}
            </div>

            <CountrySelectDialog
                open={showCountryDialog}
                onClose={() => setShowCountryDialog(false)}
                countries={countries}
                selectedCountry={country}
                onSelect={(c) => setAiScanner(prev => ({ ...prev, country: c }))}
            />

            {(results.length > 0 || loading) && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-6 duration-1000">

                    {/* Results Table */}
                    <div className={`overflow-hidden rounded-[2rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl shadow-2xl flex flex-col ${selected ? "lg:col-span-8" : "lg:col-span-12"}`}>
                        <div className="bg-zinc-950/80 px-8 py-5 border-b border-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Activity className="w-4 h-4 text-emerald-500" />
                                <h3 className="text-sm font-black text-white uppercase tracking-widest">{t("ai.matches")}</h3>
                            </div>
                            <button
                                type="button"
                                onClick={() => setAiScanner(prev => ({ ...prev, showPrecisionInfo: !prev.showPrecisionInfo }))}
                                className="inline-flex items-center gap-2 rounded-xl border border-white/5 bg-zinc-900/50 px-4 py-2 text-[10px] font-black uppercase tracking-widest text-zinc-400 hover:text-white transition-all"
                            >
                                <Info className="h-3.5 w-3.5 text-indigo-500" />
                                Stats Key
                            </button>
                        </div>

                        {showPrecisionInfo && (
                            <div className="px-8 py-4 text-[11px] font-medium text-zinc-500 bg-indigo-500/5 border-b border-indigo-500/10 leading-relaxed italic">
                                {t("ai.precision_info")}
                            </div>
                        )}

                        <div className="overflow-x-auto custom-scrollbar">
                            <table className="w-full text-left text-sm whitespace-nowrap">
                                <thead className="bg-zinc-950/80 text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500 border-b border-white/5">
                                    <tr>
                                        <th className="px-8 py-5">{t("ai.table.symbol")}</th>
                                        <th className="px-6 py-5">{t("ai.table.name")}</th>
                                        <th className="px-6 py-5 text-right">{t("ai.table.price")}</th>
                                        <th className="px-6 py-5 text-center">{t("ai.table.precision")}</th>
                                        <th className="px-8 py-5 text-right">{t("ai.table.save")}</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    {results.map((r) => (
                                        <tr
                                            key={r.symbol}
                                            onClick={() => openDetails(r)}
                                            className={`cursor-pointer group transition-all ${selected?.symbol === r.symbol ? "bg-indigo-600/10" : "hover:bg-white/[0.02]"}`}
                                        >
                                            <td className="px-8 py-5">
                                                <div className="flex items-center gap-4">
                                                    <StockLogo symbol={r.symbol} logoUrl={r.logo_url} size="md" />
                                                    <span className="font-mono font-black text-indigo-400 group-hover:text-indigo-300 transition-colors">{r.symbol}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-5">
                                                <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-tighter truncate max-w-[200px] block">{r.name}</span>
                                            </td>
                                            <td className="px-6 py-5 text-right">
                                                <span className="font-mono font-black text-zinc-100">{r.last_close.toFixed(2)}</span>
                                            </td>
                                            <td className="px-6 py-5 text-center">
                                                <div className={`inline-flex items-center h-7 px-3 rounded-lg text-[10px] font-black border tracking-widest ${r.precision > 0.7
                                                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                                    : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                                                    }`}>
                                                    {(r.precision * 100).toFixed(1)}%
                                                </div>
                                            </td>
                                            <td className="px-8 py-5 text-right" onClick={(e) => e.stopPropagation()}>
                                                <button
                                                    onClick={() => {
                                                        if (isSaved(r.symbol)) {
                                                            removeSymbolBySymbol(r.symbol);
                                                        } else {
                                                            saveSymbol({
                                                                symbol: r.symbol,
                                                                name: r.name,
                                                                source: "ai_scanner",
                                                                metadata: {
                                                                    precision: r.precision,
                                                                    last_close: r.last_close,
                                                                    logo_url: r.logo_url
                                                                }
                                                            });
                                                        }
                                                    }}
                                                    className={`p-2.5 rounded-xl transition-all ${isSaved(r.symbol)
                                                        ? "text-indigo-400 bg-indigo-500/10"
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
                        </div>

                        <div className="bg-zinc-950/80 px-8 py-3 text-[9px] font-black uppercase tracking-[0.2em] text-zinc-600 border-t border-white/5 flex justify-between">
                            <span>{t("dash.scanned").replace("{count}", progress.current.toString())}</span>
                            <span>{results.length} OPPORTUNITIES FILTERED</span>
                        </div>
                    </div>

                    {/* Details Panel */}
                    {selected && (
                        <div className="lg:col-span-4 rounded-[2rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl shadow-2xl overflow-hidden h-fit lg:sticky lg:top-8 animate-in slide-in-from-right-8 duration-700">
                            <div className="px-6 py-5 border-b border-white/5 flex items-center justify-between bg-zinc-950/60">
                                <div className="space-y-0.5 min-w-0">
                                    <div className="text-xl font-black text-white font-mono tracking-tighter truncate">{selected.symbol}</div>
                                    <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest truncate">{selected.name}</div>
                                </div>
                                <button
                                    onClick={() => setAiScanner(prev => ({ ...prev, selected: null, detailData: null }))}
                                    className="p-2.5 rounded-xl text-zinc-500 hover:text-white hover:bg-zinc-800 transition-all"
                                >
                                    <X className="h-4.5 w-4.5" />
                                </button>
                            </div>

                            <div className="p-6 space-y-8">
                                {/* Chart Controls */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest italic">{t("ai.chart_ctrl")}</span>
                                        <div className="flex gap-1.5 p-1 rounded-xl bg-zinc-900/50 border border-white/5">
                                            <button
                                                onClick={() => setAiScanner(prev => ({ ...prev, chartType: "candle" }))}
                                                className={`p-2 rounded-lg transition-all ${chartType === "candle" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-zinc-600 hover:text-zinc-400"}`}
                                            >
                                                <BarChart2 className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => setAiScanner(prev => ({ ...prev, chartType: "area" }))}
                                                className={`p-2 rounded-lg transition-all ${chartType === "area" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-zinc-600 hover:text-zinc-400"}`}
                                            >
                                                <LineChart className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        {[
                                            { id: "showEma50", label: "EMA 50", color: "bg-orange-500" },
                                            { id: "showEma200", label: "EMA 200", color: "bg-cyan-500" },
                                            { id: "showBB", label: "BB", color: "bg-purple-500" },
                                            { id: "showRsi", label: "RSI", color: "bg-pink-500" },
                                        ].map((ind) => (
                                            <label key={ind.id} className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-zinc-900/30 border border-white/5 cursor-pointer hover:bg-zinc-800 transition-all group">
                                                <div className={`w-3 h-3 rounded-full border border-white/10 ${state.aiScanner[ind.id as keyof typeof state.aiScanner] ? ind.color : "bg-transparent"}`} />
                                                <input
                                                    type="checkbox"
                                                    className="hidden"
                                                    checked={!!state.aiScanner[ind.id as keyof typeof state.aiScanner]}
                                                    onChange={(e) => setAiScanner(prev => ({ ...prev, [ind.id]: e.target.checked }))}
                                                />
                                                <span className="text-[9px] font-black uppercase tracking-widest text-zinc-500 group-hover:text-zinc-300">{ind.label}</span>
                                            </label>
                                        ))}
                                    </div>
                                </div>

                                {detailError && (
                                    <div className="p-4 rounded-2xl bg-red-500/5 border border-red-500/20 flex flex-col items-center gap-2 text-center">
                                        <AlertTriangle className="h-5 w-5 text-red-500" />
                                        <p className="text-[11px] font-bold text-red-400/80">{detailError}</p>
                                    </div>
                                )}

                                {detailLoading && (
                                    <div className="flex flex-col items-center justify-center py-20 gap-4">
                                        <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                                        <p className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Building visual report...</p>
                                    </div>
                                )}

                                {detailData && (
                                    <article className="space-y-6 animate-in fade-in duration-500">
                                        <div className="rounded-2xl border border-white/5 bg-zinc-900/20 overflow-hidden">
                                            <CandleChart
                                                rows={detailData.testPredictions}
                                                showEma50={showEma50}
                                                showEma200={showEma200}
                                                showBB={showBB}
                                                showRsi={showRsi}
                                                showVolume={showVolume}
                                                chartType={chartType}
                                            />
                                        </div>
                                        <div className="p-4 rounded-2xl bg-indigo-600/5 border border-indigo-500/10">
                                            <div className="text-[9px] font-black text-indigo-400 uppercase tracking-widest mb-3">Historical Accuracy Bench</div>
                                            <TableView rows={detailData.testPredictions} ticker={selected.symbol} />
                                        </div>
                                    </article>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
