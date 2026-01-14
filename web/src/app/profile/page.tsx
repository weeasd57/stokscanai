"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";
import { useAuth } from "@/contexts/AuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { Trash2, LineChart, Cpu, X, Loader2, Save, Target, ShieldAlert, Award, Clock } from "lucide-react";
import { predictStock } from "@/lib/api";
import CandleChart from "@/components/CandleChart";
import ConfirmDialog from "@/components/ConfirmDialog";
import type { PredictResponse } from "@/lib/types";

type ProfileRow = {
  default_target_pct: number;
  default_stop_pct: number;
  username: string | null;
  display_name: string | null;
};

type PositionRow = {
  id: string;
  symbol: string;
  name: string | null;
  source: "home" | "ai_scanner" | "tech_scanner";
  added_at: string;
  metadata: any;
  entry_price: number | null;
  target_pct: number;
  stop_pct: number;
  target_price: number | null;
  stop_price: number | null;
  status: "open" | "hit_target" | "hit_stop" | "closed_manual";
  status_at: string | null;
  status_price: number | null;
};

type StatsRow = {
  open_count: number;
  win_count: number;
  loss_count: number;
  total_count: number;
  win_rate: number;
} | null;

export default function ProfilePage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const { t } = useLanguage();
  const supabase = useMemo(() => createSupabaseBrowserClient(), []);

  const [profile, setProfile] = useState<ProfileRow | null>(null);
  const [positions, setPositions] = useState<PositionRow[]>([]);
  const [stats, setStats] = useState<StatsRow>(null);

  const [defaultsTarget, setDefaultsTarget] = useState("5");
  const [defaultsStop, setDefaultsStop] = useState("2");
  const [savingDefaults, setSavingDefaults] = useState(false);

  const [openRouterKey, setOpenRouterKey] = useState("");
  const [customRules, setCustomRules] = useState("");
  const [savingAi, setSavingAi] = useState(false);

  const [evaluating, setEvaluating] = useState(false);
  const [evalLog, setEvalLog] = useState<string[]>([]);

  const [chartPosition, setChartPosition] = useState<PositionRow | null>(null);
  const [chartData, setChartData] = useState<PredictResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [positionToDelete, setPositionToDelete] = useState<string | null>(null);
  const [removing, setRemoving] = useState(false);
  const [evalResults, setEvalResults] = useState<any[] | null>(null);
  const [showEvalDialog, setShowEvalDialog] = useState(false);

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, router, user]);

  const reloadAll = useCallback(async () => {
    if (!user) return;

    const { data: profileRow } = await supabase
      .from("profiles")
      .select("default_target_pct, default_stop_pct, username, display_name")
      .eq("id", user.id)
      .maybeSingle();

    if (profileRow) {
      setProfile(profileRow as ProfileRow);
      setDefaultsTarget(String((profileRow as any).default_target_pct ?? 5));
      setDefaultsStop(String((profileRow as any).default_stop_pct ?? 2));
    }

    const { data: statsRow } = await supabase.from("my_position_stats").select("*").maybeSingle();
    setStats((statsRow ?? null) as any);

    const { data: posRows } = await supabase
      .from("positions")
      .select(
        "id, symbol, name, source, added_at, metadata, entry_price, target_pct, stop_pct, target_price, stop_price, status, status_at, status_price"
      )
      .order("added_at", { ascending: false });
    setPositions((posRows ?? []) as any);
  }, [supabase, user]);

  useEffect(() => {
    if (!user) return;
    void reloadAll();
  }, [reloadAll, user]);

  async function saveDefaults() {
    if (!user) return;
    const target = Number(defaultsTarget);
    const stop = Number(defaultsStop);
    if (!Number.isFinite(target) || target <= 0 || target > 100) return;
    if (!Number.isFinite(stop) || stop <= 0 || stop > 100) return;

    setSavingDefaults(true);
    try {
      await supabase
        .from("profiles")
        .update({ default_target_pct: target, default_stop_pct: stop })
        .eq("id", user.id);
      await reloadAll();
    } finally {
      setSavingDefaults(false);
    }
  }


  async function removePosition(id: string) {
    setPositionToDelete(id);
    setConfirmOpen(true);
  }

  async function handleConfirmDelete() {
    if (!positionToDelete) return;
    setRemoving(true);
    try {
      await supabase.from("positions").delete().eq("id", positionToDelete);
      await reloadAll();
    } finally {
      setRemoving(false);
      setConfirmOpen(false);
      setPositionToDelete(null);
    }
  }

  async function openChart(pos: PositionRow) {
    setChartPosition(pos);
    setChartLoading(true);
    setChartError(null);
    setChartData(null);
    try {
      const res = await predictStock({ ticker: pos.symbol, includeFundamentals: false });
      setChartData(res);
    } catch (err: any) {
      setChartError(err.message || "Failed to load chart");
    } finally {
      setChartLoading(false);
    }
  }

  async function evaluateOpenPositions() {
    if (!user) return;
    setEvaluating(true);
    setEvalLog([]);
    setEvalResults(null);
    try {
      const open = positions.filter((p) => p.status === "open");
      if (open.length === 0) {
        setEvalLog(["No open positions."]);
        return;
      }
      setEvalLog([`Evaluating ${open.length} positions...`]);
      const res = await fetch("/api/positions/evaluate_open_history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({
          positions: open.map((p) => ({
            id: p.id,
            symbol: p.symbol,
            entry_price: p.entry_price,
            entry_at: (p as any).entry_at ?? null,
            added_at: p.added_at,
            target_price: p.target_price,
            stop_price: p.stop_price,
          })),
        }),
      });
      if (!res.ok) {
        setEvalLog((prev) => [...prev, `Request failed: ${res.status}`]);
        return;
      }
      const decisions = await res.json();
      setEvalResults(decisions);
      setShowEvalDialog(true);

      setEvalLog((prev) => [...prev, "Saving results to database..."]);
      for (const d of decisions) {
        // We always call RPC now to save the latest price even for 'open' ones
        if (d.price) {
          await supabase.rpc("evaluate_position", {
            p_position_id: d.id,
            p_current_price: d.price,
            p_as_of: d.as_of
          });
        }
      }
      await reloadAll();
      setEvalLog((prev) => [...prev, "Done."]);
    } catch (err: any) {
      setEvalLog((prev) => [...prev, `Error: ${err.message}`]);
    } finally {
      setEvaluating(false);
    }
  }

  if (loading) return <div className="flex h-96 items-center justify-center"><Loader2 className="h-8 w-8 animate-spin text-indigo-500" /></div>;
  if (!user) return null;

  const winRate = stats?.win_rate ?? 0;

  return (
    <div className="flex flex-col gap-10 pb-20 max-w-[1600px] mx-auto">
      <header className="flex flex-col gap-3">
        <h1 className="text-4xl font-black tracking-tighter text-white uppercase italic">{t("nav.profile")}</h1>
        <p className="text-sm text-zinc-500 font-medium max-w-lg">{t("profile.track")}</p>
      </header>

      {/* Stats Overview */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {[
          { label: t("profile.stats.open"), val: stats?.open_count ?? 0, color: "text-blue-400", icon: <Clock className="w-4 h-4" /> },
          { label: t("profile.stats.wins"), val: stats?.win_count ?? 0, color: "text-emerald-400", icon: <Award className="w-4 h-4" /> },
          { label: t("profile.stats.losses"), val: stats?.loss_count ?? 0, color: "text-red-400", icon: <ShieldAlert className="w-4 h-4" /> },
          { label: t("profile.stats.winrate"), val: `${(winRate * 100).toFixed(1)}%`, color: "text-white", icon: <Target className="w-4 h-4" /> },
        ].map((s) => (
          <div key={s.label} className="relative overflow-hidden group rounded-[2rem] border border-white/5 bg-zinc-950/40 p-6 backdrop-blur-xl shadow-2xl transition-all hover:border-white/10">
            <div className="flex items-center gap-3 mb-2">
              <div className={`p-2 rounded-xl bg-white/5 ${s.color}`}>{s.icon}</div>
              <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{s.label}</span>
            </div>
            <div className={`text-3xl font-black tracking-tighter ${s.color}`}>{s.val}</div>
            <div className="absolute -bottom-4 -right-4 w-20 h-20 bg-white/5 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-700" />
          </div>
        ))}
      </section>

      <div className="grid grid-cols-1 gap-8">
        {/* Trading Defaults */}
        <section className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 p-8 shadow-2xl backdrop-blur-xl space-y-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-600/5 blur-[100px] rounded-full -translate-y-1/2 translate-x-1/2" />

          <div className="relative">
            <h2 className="text-xl font-black text-white uppercase tracking-tight mb-2">{t("profile.defaults.title")}</h2>
            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest leading-relaxed">{t("profile.defaults.subtitle")}</p>
          </div>

          <div className="grid grid-cols-2 gap-6 relative">
            <div className="space-y-3">
              <label className="text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em] ml-1">{t("profile.defaults.target")}</label>
              <div className="relative group">
                <input
                  type="number"
                  value={defaultsTarget}
                  onChange={(e) => setDefaultsTarget(e.target.value)}
                  className="h-14 w-full rounded-2xl border border-white/5 bg-zinc-900/50 px-5 text-lg font-black text-indigo-400 outline-none focus:ring-1 focus:ring-indigo-500/30 transition-all font-mono"
                />
                <span className="absolute right-5 top-1/2 -translate-y-1/2 font-black text-zinc-700">%</span>
              </div>
            </div>
            <div className="space-y-3">
              <label className="text-[10px] font-black text-zinc-600 uppercase tracking-[0.2em] ml-1">{t("profile.defaults.stop")}</label>
              <div className="relative group">
                <input
                  type="number"
                  value={defaultsStop}
                  onChange={(e) => setDefaultsStop(e.target.value)}
                  className="h-14 w-full rounded-2xl border border-white/5 bg-zinc-900/50 px-5 text-lg font-black text-red-400 outline-none focus:ring-1 focus:ring-red-500/30 transition-all font-mono"
                />
                <span className="absolute right-5 top-1/2 -translate-y-1/2 font-black text-zinc-700">%</span>
              </div>
            </div>
          </div>

          <button
            onClick={saveDefaults}
            disabled={savingDefaults}
            className="h-14 w-full rounded-2xl bg-indigo-600 text-[11px] font-black uppercase tracking-[0.2em] text-white shadow-xl shadow-indigo-600/20 hover:bg-indigo-500 transition-all active:scale-[0.98] flex items-center justify-center gap-3 relative overflow-hidden group"
          >
            {savingDefaults ? <Loader2 className="h-5 w-5 animate-spin" /> : <Save className="h-5 w-5 group-hover:scale-110 transition-transform" />}
            {t("profile.defaults.title")}
          </button>
        </section>

      </div>

      {/* Positions Section */}
      <section className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 p-8 shadow-2xl backdrop-blur-xl space-y-8 min-h-[500px]">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="space-y-2">
            <h2 className="text-2xl font-black text-white uppercase tracking-tight">{t("profile.positions.title")}</h2>
            <p className="text-sm text-zinc-500 font-medium max-w-xl">{t("profile.positions.subtitle")}</p>
          </div>
          <button
            onClick={evaluateOpenPositions}
            disabled={evaluating}
            className="h-12 px-8 rounded-2xl bg-white text-zinc-950 text-[11px] font-black uppercase tracking-widest hover:bg-zinc-200 transition-all shadow-xl shadow-white/5 disabled:opacity-50"
          >
            {evaluating ? "Evaluating..." : t("profile.positions.evaluate")}
          </button>
        </div>

        {evalLog.length > 0 && (
          <div className="p-5 rounded-[1.5rem] bg-black/40 border border-white/5 animate-in slide-in-from-top-4 duration-500">
            <div className="text-[9px] font-black text-zinc-600 uppercase tracking-widest mb-3 flex items-center gap-2">
              <div className="w-1 h-1 rounded-full bg-indigo-500 animate-pulse" />
              Runtime Evaluation Console
            </div>
            <pre className="max-h-32 overflow-auto text-[10px] font-mono text-indigo-400 custom-scrollbar opacity-80 leading-loose">{evalLog.join("\n")}</pre>
          </div>
        )}

        <div className="overflow-hidden rounded-[2rem] border border-white/5 bg-zinc-950/80">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-zinc-950/80 text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500 border-b border-white/5">
                <tr>
                  <th className="px-8 py-5">{t("profile.table.symbol")}</th>
                  <th className="px-6 py-5 text-center">{t("profile.table.status")}</th>
                  <th className="px-6 py-5 text-right font-mono">LAST DATE</th>
                  <th className="px-6 py-5 text-right font-mono">LAST PRICE</th>
                  <th className="px-6 py-5 text-right font-mono">% CHG</th>
                  <th className="px-6 py-5 text-right">{t("profile.table.entry")}</th>
                  <th className="px-6 py-5 text-right">{t("profile.table.target")}</th>
                  <th className="px-6 py-5 text-right">{t("profile.table.stop")}</th>
                  <th className="px-8 py-5 text-right">{t("profile.table.actions")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {positions.map((p) => (
                  <tr key={p.id} onClick={() => openChart(p)} className="group hover:bg-white/[0.05] transition-colors cursor-pointer border-b border-white/5">
                    <td className="px-8 py-6">
                      <div className="flex flex-col gap-1">
                        <span className="font-mono font-black text-indigo-400 group-hover:text-indigo-300 transition-colors">{p.symbol}</span>
                        <div className="flex items-center gap-2 text-[10px] font-black text-zinc-600 uppercase tracking-widest">
                          <Clock className="w-2.5 h-2.5" />
                          {new Date(p.added_at).toLocaleDateString()}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-6 text-center">
                      <span className={`
                          inline-flex items-center h-7 px-4 rounded-full text-[10px] font-black uppercase tracking-widest border
                          ${p.status === 'hit_target' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                          p.status === 'hit_stop' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                            p.status === 'open' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 'bg-zinc-800/10 text-zinc-500 border-white/5'}
                        `}>
                        {p.status.replace("_", " ")}
                      </span>
                    </td>
                    <td className="px-6 py-6 text-right font-mono font-bold text-zinc-400 text-xs">
                      {p.status_at ? new Date(p.status_at).toLocaleDateString() : "--"}
                    </td>
                    <td className="px-6 py-6 text-right font-mono font-black text-white">
                      {p.status_price?.toFixed(2) ?? "--"}
                    </td>
                    <td className="px-6 py-6 text-right font-mono font-black">
                      {(() => {
                        if (!p.entry_price || !p.status_price) return <span className="text-zinc-600">--</span>;
                        const pct = ((p.status_price - p.entry_price) / p.entry_price) * 100;
                        const isProf = pct >= 0;
                        return (
                          <span className={isProf ? "text-emerald-400" : "text-red-400"}>
                            {isProf ? "+" : ""}{pct.toFixed(2)}%
                          </span>
                        );
                      })()}
                    </td>
                    <td className="px-6 py-6 text-right font-mono font-black text-zinc-100">{p.entry_price ?? "--"}</td>
                    <td className="px-6 py-6 text-right font-mono font-black text-emerald-400/80">{p.target_price?.toFixed(2) ?? "--"}</td>
                    <td className="px-6 py-6 text-right font-mono font-black text-red-400/80">{p.stop_price?.toFixed(2) ?? "--"}</td>
                    <td className="px-8 py-6 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => openChart(p)}
                          className="p-2.5 rounded-xl border border-white/5 text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 transition-all"
                        >
                          <LineChart className="h-4.5 w-4.5" />
                        </button>
                        <button
                          onClick={() => removePosition(p.id)}
                          className="p-2.5 rounded-xl border border-white/5 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                        >
                          <Trash2 className="h-4.5 w-4.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {positions.length === 0 && (
                  <tr>
                    <td className="px-8 py-20 text-center" colSpan={10}>
                      <div className="flex flex-col items-center gap-4 text-zinc-700">
                        <Target className="h-10 w-10 opacity-10" />
                        <span className="text-xs font-black uppercase tracking-[0.3em]">No Active Positions</span>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Chart Modal */}
      {chartPosition && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/90 backdrop-blur-xl animate-in fade-in duration-500">
          <div className="w-full max-w-5xl bg-zinc-950/80 border border-white/10 rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col max-h-[90vh] relative">

            {/* Modal Header */}
            <div className="flex items-center justify-between px-10 py-8 border-b border-white/5">
              <div className="space-y-1">
                <h3 className="text-3xl font-black text-white font-mono tracking-tighter flex items-center gap-4">
                  {chartPosition.symbol}
                  <span className="text-[10px] font-black text-zinc-500 uppercase tracking-[0.3em] font-sans px-4 py-1.5 rounded-xl border border-white/5">
                    {new Date(chartPosition.added_at).toLocaleDateString()}
                  </span>
                </h3>
              </div>
              <button
                onClick={() => setChartPosition(null)}
                className="p-3 rounded-2xl bg-zinc-900/50 text-zinc-500 hover:text-white transition-all active:scale-95"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-10 custom-scrollbar bg-black/40">
              {chartLoading ? (
                <div className="h-[400px] flex flex-col items-center justify-center text-zinc-600 gap-4">
                  <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                  <span className="text-[10px] font-black uppercase tracking-widest italic">Synchronizing Data...</span>
                </div>
              ) : chartError ? (
                <div className="h-[400px] flex flex-col items-center justify-center text-red-500 gap-4">
                  <ShieldAlert className="h-10 w-10 opacity-20" />
                  <span className="text-xs font-bold">{chartError}</span>
                </div>
              ) : chartData ? (
                <div className="animate-in fade-in zoom-in-95 duration-700 bg-zinc-950/50 rounded-[2rem] border border-white/5 overflow-hidden p-6 box-content">
                  <CandleChart
                    rows={chartData.testPredictions}
                    targetPrice={chartPosition.target_price ?? undefined}
                    stopPrice={chartPosition.stop_price ?? undefined}
                    savedDate={chartPosition.added_at.split('T')[0]}
                    showVolume={true}
                    showEma50={true}
                  />
                </div>
              ) : null}
            </div>

            {/* Modal Footer */}
            <div className="px-10 py-8 border-t border-white/5 bg-zinc-950/80 flex items-center justify-between">
              <div className="flex flex-col">
                <span className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em] mb-1">Entry Value</span>
                <span className="text-xl font-mono font-black text-blue-400">{chartPosition.entry_price ?? "N/A"}</span>
              </div>
              <div className="flex gap-12">
                <div className="flex flex-col items-end">
                  <span className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em] mb-1">Take Profit</span>
                  <span className="text-xl font-mono font-black text-emerald-400">{chartPosition.target_price?.toFixed(2) ?? "N/A"}</span>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em] mb-1">Stop Loss</span>
                  <span className="text-xl font-mono font-black text-red-400">{chartPosition.stop_price?.toFixed(2) ?? "N/A"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmOpen}
        title="Remove Position"
        message="Are you sure you want to remove this position? This action cannot be undone."
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleConfirmDelete}
        isLoading={removing}
        confirmLabel="Remove"
        variant="danger"
      />

      {/* Evaluation Results Dialog */}
      {showEvalDialog && evalResults && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="w-full max-w-2xl bg-zinc-950 border border-white/10 rounded-[2rem] shadow-2xl overflow-hidden flex flex-col max-h-[80vh]">
            <div className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-zinc-900/50">
              <div className="flex items-center gap-3">
                <Cpu className="w-5 h-5 text-indigo-400" />
                <h3 className="text-xl font-bold text-white uppercase tracking-tight">Evaluation Results</h3>
              </div>
              <button
                onClick={() => setShowEvalDialog(false)}
                className="p-2 rounded-xl bg-zinc-800 text-zinc-400 hover:text-white transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-8 space-y-4 custom-scrollbar">
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead className="text-[10px] font-black uppercase tracking-widest text-zinc-500 border-b border-white/5">
                  <tr>
                    <th className="pb-4">Symbol</th>
                    <th className="pb-4 text-center">Status</th>
                    <th className="pb-4 text-right">Last Date</th>
                    <th className="pb-4 text-right">Last Price</th>
                    <th className="pb-4 text-right">% CHG</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {evalResults.map((r) => (
                    <tr key={r.id} className="text-xs">
                      <td className="py-4 font-mono font-bold text-indigo-400">{r.symbol}</td>
                      <td className="py-4 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border ${r.status === 'hit_target' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                            r.status === 'hit_stop' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                              'bg-blue-500/10 text-blue-400 border-blue-500/20'
                          }`}>
                          {r.status.replace("_", " ")}
                        </span>
                      </td>
                      <td className="py-4 text-right font-mono text-zinc-400">{r.as_of ? new Date(r.as_of).toLocaleDateString() : "--"}</td>
                      <td className="py-4 text-right font-mono font-bold text-white">{r.price?.toFixed(2) ?? "--"}</td>
                      <td className="py-4 text-right font-mono font-bold">
                        {r.change_pct !== null ? (
                          <span className={r.change_pct >= 0 ? "text-emerald-400" : "text-red-400"}>
                            {r.change_pct >= 0 ? "+" : ""}{r.change_pct.toFixed(2)}%
                          </span>
                        ) : "--"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="px-8 py-6 border-t border-white/5 bg-zinc-900/50 flex justify-end">
              <button
                onClick={() => setShowEvalDialog(false)}
                className="px-6 py-2.5 rounded-xl bg-white text-zinc-950 text-xs font-black uppercase tracking-widest hover:bg-zinc-200 transition-all"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
