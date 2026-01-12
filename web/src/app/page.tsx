"use client";

import { useMemo, useRef, useState, useEffect, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { TrendingDown, TrendingUp, BarChart2, LineChart, Globe, Brain, AlertTriangle, CheckCircle2, Bookmark, BookmarkCheck, LayoutDashboard, Activity, Search } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWatchlist } from "@/contexts/WatchlistContext";
import { useAppState } from "@/contexts/AppStateContext";

import CandleChart from "@/components/CandleChart";
import TableView from "@/components/TableView";
import CountrySymbolDialog from "@/components/CountrySymbolDialog";

const DEFAULT_TICKER = "AAPL";

function formatPct(v: number) {
  return `${(v * 100).toFixed(2)}%`;
}

function formatNumber(v: number | null | undefined) {
  if (v === null || v === undefined) return "-";
  return new Intl.NumberFormat("en-US").format(v);
}

export default function HomePage() {
  const { t } = useLanguage();
  const { saveSymbol, removeSymbol, isSaved, watchlist } = useWatchlist();
  const { state, setHome, runHomePredict, clearHomeView, restoreLastHomePredict } = useAppState();
  const { ticker, data, chartType, showEma50, showEma200, showBB, showRsi, showVolume, predictHistory } = state.home;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSymbolDialog, setShowSymbolDialog] = useState(false);

  const searchParams = useSearchParams();
  const lastQueryTickerRef = useRef<string | null>(null);
  const queryTicker = searchParams.get("ticker");

  useEffect(() => {
    if (!queryTicker) return;
    if (lastQueryTickerRef.current === queryTicker) return;
    lastQueryTickerRef.current = queryTicker;
    void runPrediction([queryTicker]);
  }, [queryTicker]);

  const buyCount = useMemo(() => {
    if (!data) return 0;
    return data.testPredictions.reduce((acc, r) => acc + (r.pred === 1 ? 1 : 0), 0);
  }, [data]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const normalized = ticker.trim().toUpperCase();
    if (!/^[A-Z0-9.\-]{1,24}$/.test(normalized)) {
      setError("Invalid ticker format");
      return;
    }
    setLoading(true);
    try {
      await runHomePredict(normalized);
    } catch (err) {
      clearHomeView();
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function runPrediction(symbols: string[]) {
    if (symbols.length === 0) return;
    const first = symbols[0];
    setHome({ ticker: first });
    setError(null);
    setLoading(true);
    try {
      await runHomePredict(first);
    } catch (err) {
      clearHomeView();
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-10 pb-20 max-w-[1600px] mx-auto">
      {/* Hero Section */}
      <header className="flex flex-col gap-3">
        <h1 className="text-4xl font-black tracking-tighter text-white uppercase italic flex items-center gap-4">
          <div className="p-3 rounded-2xl bg-indigo-600 shadow-xl shadow-indigo-600/20">
            <Activity className="h-6 w-6 text-white" />
          </div>
          {t("app.title")}
        </h1>
        <p className="text-sm text-zinc-500 font-medium max-w-lg">{t("app.subtitle")}</p>
      </header>

      {/* Main Search Controls */}
      <section className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 p-8 shadow-2xl backdrop-blur-xl relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-600/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 group-hover:bg-indigo-600/10 transition-colors" />

        <form onSubmit={onSubmit} className="flex flex-col md:flex-row items-stretch gap-6 relative z-10">
          <div className="flex-1 space-y-3">
            <label className="text-[10px] font-black text-zinc-600 uppercase tracking-[0.3em] ml-1">{t("ticker.label")}</label>
            <div className="relative">
              <input
                value={ticker}
                onChange={(e) => setHome({ ticker: e.target.value })}
                className="h-16 w-full rounded-2xl border border-white/5 bg-zinc-900/50 px-6 text-xl font-black text-white outline-none focus:ring-1 focus:ring-indigo-500/30 transition-all placeholder:text-zinc-800 font-mono"
                placeholder={t("ticker.placeholder")}
                inputMode="text"
                autoCapitalize="characters"
                spellCheck={false}
              />
              <Search className="absolute right-6 top-1/2 -translate-y-1/2 h-6 w-6 text-zinc-800" />
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-end gap-3 min-w-fit">
            <button
              type="button"
              onClick={() => setShowSymbolDialog(true)}
              className="h-16 px-8 rounded-2xl border border-white/5 bg-zinc-900/50 text-xs font-black uppercase tracking-[0.2em] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all flex items-center gap-3 active:scale-95"
            >
              <Globe className="h-4 w-4 text-blue-500" />
              {t("btn.browse")}
            </button>

            <button
              type="submit"
              disabled={loading}
              className={`h-16 px-12 rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 shadow-2xl active:scale-95 ${loading ? "bg-zinc-800 text-zinc-500" : "bg-indigo-600 text-white shadow-indigo-600/30 hover:bg-indigo-500"}`}
            >
              {loading && <div className="w-4 h-4 rounded-full border-2 border-white/20 border-t-white animate-spin" />}
              {loading ? t("btn.running") : t("btn.run")}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-6 p-4 rounded-xl bg-red-500/5 border border-red-500/10 flex items-center gap-3 text-red-400 text-xs font-bold animate-in slide-in-from-top-2 duration-300">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}
      </section>

      {data && (
        <section className="space-y-10 animate-in fade-in slide-in-from-bottom-6 duration-1000">
          {/* Top Stats Banner */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { label: t("result.precision"), val: formatPct(data.precision), sub: `${t("result.window")}: ${data.testPredictions.length} days`, color: "text-white" },
              { label: t("result.last_close"), val: data.lastClose.toFixed(2), sub: `${t("result.date")}: ${data.lastDate}`, color: "text-zinc-100" },
              {
                label: t("result.signal"),
                val: data.tomorrowPrediction === 1 ? t("signal.up") : t("signal.down"),
                sub: `Buy signals in test: ${buyCount}`,
                isSignal: true,
                color: data.tomorrowPrediction === 1 ? "text-emerald-400" : "text-red-400"
              },
            ].map((stat, idx) => (
              <div key={idx} className="relative group rounded-[2.5rem] border border-white/5 bg-zinc-950/40 p-8 pt-10 shadow-2xl backdrop-blur-xl overflow-hidden">
                <span className="absolute top-8 left-8 text-[10px] font-black text-zinc-600 uppercase tracking-[0.3em]">{stat.label}</span>
                <div className={`text-4xl font-black font-mono tracking-tighter mt-4 flex items-center gap-4 ${stat.color}`}>
                  {stat.isSignal && (data.tomorrowPrediction === 1 ? <TrendingUp className="h-8 w-8" /> : <TrendingDown className="h-8 w-8" />)}
                  {stat.val}
                </div>
                <div className="mt-3 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{stat.sub}</div>

                {stat.isSignal && (
                  <button
                    onClick={() => {
                      const savedItem = watchlist.find((item) => String(item.symbol).toUpperCase() === String(ticker).toUpperCase());
                      if (savedItem) {
                        removeSymbol(savedItem.id);
                      } else {
                        saveSymbol({
                          symbol: ticker,
                          name: (data.fundamentals as any)?.name || ticker,
                          source: "home",
                          metadata: { prediction: data.tomorrowPrediction, precision: data.precision, price: data.lastClose }
                        });
                      }
                    }}
                    className={`absolute top-8 right-8 p-3 rounded-2xl transition-all ${isSaved(ticker)
                      ? "text-indigo-400 bg-indigo-500/10 hover:bg-indigo-500/20 shadow-lg shadow-indigo-600/10"
                      : "text-zinc-700 hover:text-white hover:bg-zinc-800"
                      }`}
                    title={isSaved(ticker) ? "Remove from Watchlist" : "Add to Watchlist"}
                  >
                    {isSaved(ticker) ? <BookmarkCheck className="h-5 w-5" /> : <Bookmark className="h-5 w-5" />}
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* AI Insights Layer */}
          {data.fundamentals && (
            <div className="rounded-[3rem] border border-white/10 bg-gradient-to-br from-indigo-950/40 via-zinc-950/40 to-zinc-950 p-10 shadow-2xl relative overflow-hidden">
              <div className="absolute -top-24 -left-24 w-96 h-96 bg-indigo-600/10 blur-[120px] rounded-full" />
              <div className="absolute bottom-0 right-0 w-full h-px bg-gradient-to-r from-transparent via-indigo-500/20 to-transparent" />

              <div className="flex items-center gap-6 mb-10 relative">
                <div className="p-4 rounded-[1.5rem] bg-indigo-600/20 border border-indigo-500/30 shadow-2xl shadow-indigo-600/20">
                  <Brain className="h-8 w-8 text-indigo-400" />
                </div>
                <div className="space-y-1">
                  <h2 className="text-2xl font-black text-white uppercase tracking-tighter italic">{t("home.insights.title")}</h2>
                  <p className="text-[10px] font-black text-indigo-400/60 uppercase tracking-[0.3em]">{t("home.insights.subtitle")}</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
                <div className="rounded-3xl border border-white/5 bg-zinc-900/30 p-6 space-y-4">
                  <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{t("home.insights.valuation")}</div>
                  <div className="text-3xl font-black text-white font-mono">{formatNumber(data.fundamentals.peRatio ?? null)} <span className="text-xs text-zinc-600 font-sans ml-1">P/E</span></div>
                  <div className="text-[11px] font-bold text-zinc-400 leading-relaxed italic">
                    {data.fundamentals.peRatio ? (
                      data.fundamentals.peRatio < 15 ? t("home.insights.undervalued") :
                        data.fundamentals.peRatio < 30 ? t("home.insights.fair") :
                          data.fundamentals.peRatio < 100 ? t("home.insights.expensive") : t("home.insights.bubble")
                    ) : "Data Unavailable"}
                  </div>
                </div>

                <div className="rounded-3xl border border-white/5 bg-zinc-900/30 p-6 space-y-4">
                  <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{t("home.insights.volatility")}</div>
                  <div className="text-3xl font-black text-white font-mono">{formatNumber(data.fundamentals.beta ?? null)} <span className="text-xs text-zinc-600 font-sans ml-1">Beta</span></div>
                  <div className="text-[11px] font-bold text-zinc-400 leading-relaxed italic">
                    {data.fundamentals.beta ? (
                      data.fundamentals.beta < 0.8 ? t("home.insights.low_vol") :
                        data.fundamentals.beta < 1.2 ? t("home.insights.avg_vol") : t("home.insights.high_vol")
                    ) : "Data Unavailable"}
                  </div>
                </div>

                <div className="rounded-3xl border border-white/5 bg-zinc-900/30 p-6 space-y-4">
                  <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{t("home.insights.mcap")}</div>
                  <div className="text-3xl font-black text-white font-mono">
                    {data.fundamentals.marketCap ? "$" + (data.fundamentals.marketCap / 1e9).toFixed(1) + "B" : "N/A"}
                  </div>
                  <div className="text-[11px] font-bold text-zinc-400 leading-relaxed italic">
                    {data.fundamentals.marketCap ? (
                      (data.fundamentals.marketCap / 1e9) > 10 ? t("home.insights.large_cap") :
                        (data.fundamentals.marketCap / 1e9) > 2 ? t("home.insights.mid_cap") : t("home.insights.small_cap")
                    ) : "Data Unavailable"}
                  </div>
                </div>
              </div>

              {/* Conclusion Bar */}
              {data.tomorrowPrediction === 1 && typeof data.fundamentals.peRatio === 'number' && (
                <div className="mt-10 p-5 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 flex items-center gap-4 animate-pulse">
                  <CheckCircle2 className="h-6 w-6 text-indigo-400" />
                  <span className="text-xs font-black uppercase tracking-widest text-indigo-100 italic">
                    {data.fundamentals.peRatio < 30 ? t("home.insights.strong_buy") : t("home.insights.cautious_buy")}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Powerful Visualization Layer */}
          <div className="rounded-[3rem] border border-white/5 bg-zinc-950/40 p-10 shadow-2xl backdrop-blur-xl space-y-10 overflow-hidden relative">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
              <div className="space-y-1">
                <h3 className="text-xl font-black text-white uppercase tracking-tight italic">{t("home.chart.title")}</h3>
                <p className="text-[10px] font-black text-zinc-600 uppercase tracking-widest leading-relaxed">{t("home.chart.subtitle")}</p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="flex p-1.5 rounded-2xl bg-zinc-900/50 border border-white/5">
                  <button
                    onClick={() => setHome({ chartType: "candle" })}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${chartType === "candle" ? "bg-white text-zinc-950 shadow-xl" : "text-zinc-500 hover:text-white"}`}
                  >
                    <BarChart2 className="w-3.5 h-3.5" />
                    Candle
                  </button>
                  <button
                    onClick={() => setHome({ chartType: "area" })}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${chartType === "area" ? "bg-white text-zinc-950 shadow-xl" : "text-zinc-500 hover:text-white"}`}
                  >
                    <LineChart className="w-3.5 h-3.5" />
                    Area
                  </button>
                </div>

                <div className="h-6 w-px bg-white/5 mx-2" />

                <div className="flex flex-wrap items-center gap-4 bg-zinc-900/30 px-6 py-2.5 rounded-[1.5rem] border border-white/5">
                  {[
                    { id: "showEma50", label: "EMA 50", color: "bg-orange-500" },
                    { id: "showEma200", label: "EMA 200", color: "bg-cyan-500" },
                    { id: "showBB", label: "BB", color: "bg-purple-500" },
                    { id: "showRsi", label: "RSI", color: "bg-pink-500" },
                    { id: "showVolume", label: "Vol", color: "bg-blue-500" },
                  ].map((ind) => (
                    <label key={ind.label} className="flex items-center gap-2.5 cursor-pointer group">
                      <div className={`w-3 h-3 rounded-full border border-white/10 transition-all ${state.home[ind.id as keyof typeof state.home] ? ind.color + " shadow-[0_0_10px_rgba(255,255,255,0.2)]" : "bg-zinc-800"}`} />
                      <input
                        type="checkbox"
                        className="hidden"
                        checked={!!state.home[ind.id as keyof typeof state.home]}
                        onChange={(e) => setHome({ [ind.id]: e.target.checked })}
                      />
                      <span className="text-[9px] font-black text-zinc-600 uppercase tracking-widest group-hover:text-zinc-400">{ind.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-zinc-950/50 rounded-[2.5rem] border border-white/5 p-4 overflow-hidden box-content relative">
              <CandleChart
                rows={data.testPredictions}
                showEma50={showEma50}
                showEma200={showEma200}
                showBB={showBB}
                showRsi={showRsi}
                showVolume={showVolume}
                chartType={chartType}
              />
            </div>

            <TableView rows={data.testPredictions} ticker={ticker} />
          </div>

          {/* Fundamental Table Layer */}
          {!(data as any).fundamentalsError && (
            <div className="rounded-[3rem] border border-white/5 bg-zinc-950/40 p-10 shadow-2xl backdrop-blur-xl space-y-8">
              <div className="flex items-center justify-between gap-6">
                <div className="space-y-1">
                  <h3 className="text-xl font-black text-white uppercase tracking-tight italic">{t("home.fundamentals.title")}</h3>
                  <p className="text-[10px] font-black text-indigo-400/60 uppercase tracking-[0.3em]">{(data.fundamentals as any).name || ticker}</p>
                </div>
                <div className="p-3 rounded-xl bg-violet-600/10 border border-violet-500/20">
                  <LayoutDashboard className="w-5 h-5 text-violet-400" />
                </div>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                {[
                  { label: "Market Cap", val: formatNumber(data.fundamentals.marketCap) },
                  { label: "P/E Ratio", val: formatNumber(data.fundamentals.peRatio ?? null), mono: true },
                  { label: "EPS", val: formatNumber(data.fundamentals.eps ?? null), mono: true },
                  { label: "Beta", val: formatNumber(data.fundamentals.beta ?? null), mono: true },
                  { label: "Div Yield", val: (data.fundamentals as any).dividendYield ? formatPct((data.fundamentals as any).dividendYield) : "-" },
                  { label: "52W Range", val: `${formatNumber((data.fundamentals as any).low52)} - ${formatNumber((data.fundamentals as any).high52)}`, small: true },
                  { label: "Sector", val: data.fundamentals.sector ?? "-", span: 2 },
                ].map((item, i) => (
                  <div key={i} className={`rounded-[1.5rem] border border-white/5 bg-zinc-900/40 p-5 space-y-1.5 ${item.span ? "md:col-span-2" : ""}`}>
                    <div className="text-[9px] font-black text-zinc-600 uppercase tracking-widest">{item.label}</div>
                    <div className={`font-black text-zinc-100 truncate ${item.small ? "text-[11px]" : "text-sm"} ${item.mono ? "font-mono text-indigo-400/80" : ""}`}>{item.val}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      <footer className="mt-auto border-t border-white/5 pt-8 text-[10px] font-black uppercase tracking-[0.3em] text-zinc-700 text-center">
        {t("home.footer.disclaimer")}
      </footer>

      <CountrySymbolDialog
        isOpen={showSymbolDialog}
        onClose={() => setShowSymbolDialog(false)}
        onSelect={runPrediction}
        multiSelect={false}
      />
    </div>
  );
}
