"use client";

import React, { useState, useEffect } from "react";
import { Activity, Calendar, Play, TrendingUp, Target, AlertTriangle, CheckCircle2, FileText, Globe, Trash2, Eye, EyeOff, History as HistoryIcon, ChevronDown, LineChart, Database, Users, Cpu, ShieldCheck, Zap, Info } from "lucide-react";
import { getLocalModels, type LocalModelMeta, getBacktests, getBacktestTrades, deleteBacktest, updateBacktestVisibility } from "@/lib/api";
import { useAppState } from "@/contexts/AppStateContext";
import { toast } from "sonner";
import ConfirmDialog from "@/components/ConfirmDialog";
import { TradeTimeline } from "./TradeTimeline";
import Egx30MonthlyChart from "./Egx30MonthlyChart";

interface BacktestResult {
  totalTrades: number;
  winRate: number;
  avgReturnPerTrade: number;
  netProfit: number;
  trades: BacktestTrade[];
}

interface BacktestTrade {
  date: string;
  symbol: string;
  entry: number;
  exit: number;
  result: string;
  pnl_pct: number;
  features?: {
    profit_cash?: number;
    cumulative_profit?: number;
    backtest_status?: string;
    trade_date?: string;
    votes?: any;
    [key: string]: any;
  };
}



const CouncilAuditPanel = ({ bt }: { bt: any }) => {
  const rejectedProfitable = bt.rejected_profitable_trades || 0;

  return (
    <div className="mt-6 p-6 rounded-2xl bg-red-500/5 border border-red-500/20 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-red-500" />
          <h4 className="text-sm font-black text-white uppercase tracking-wider">Council Audit: Member Efficiency</h4>
        </div>
        <div className="px-3 py-1 rounded-full bg-red-500/20 text-red-400 text-[10px] font-black uppercase tracking-tighter">
          {rejectedProfitable} Profitable Trades Killed
        </div>
      </div>

      <p className="text-xs text-zinc-400 leading-relaxed max-w-2xl">
        The council filtered out <span className="text-red-400 font-bold">{rejectedProfitable}</span> opportunities that would have resulted in a profit.
        Review the voting logs below to identify which member is being too conservative.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
        <div className="p-4 rounded-xl bg-zinc-900/40 border border-white/5 space-y-2">
          <span className="text-[10px] font-bold text-zinc-500 uppercase">Suspicious Member</span>
          <div className="text-sm font-black text-white">RSI / Fundamental?</div>
          <div className="text-[9px] text-zinc-500">Check "NO" votes on filtered wins.</div>
        </div>
        <div className="p-4 rounded-xl bg-zinc-900/40 border border-white/5 space-y-2">
          <span className="text-[10px] font-bold text-zinc-500 uppercase">Council Impact</span>
          <div className="text-sm font-black text-emerald-400">+{((bt.post_council_win_rate - bt.pre_council_win_rate) || 0).toFixed(1)}% Win Rate</div>
          <div className="text-[9px] text-zinc-500">Net win rate improvement.</div>
        </div>
        <div className="p-4 rounded-xl bg-zinc-900/40 border border-white/5 space-y-2">
          <span className="text-[10px] font-bold text-zinc-500 uppercase">Action Required</span>
          <div className="text-sm font-black text-indigo-400">Refine Weights</div>
          <div className="text-[9px] text-zinc-500">Consider lowering the "King" weight.</div>
        </div>
      </div>
    </div>
  );
};

const BacktestAnalysisModal = ({ isOpen, onClose, bt, trades, loading }: { isOpen: boolean, onClose: () => void, bt: any, trades: any[], loading: boolean }) => {
  const [actualRange, setActualRange] = React.useState<{ start: string; end: string; days: number } | null>(null);
  const [activeTab, setActiveTab] = React.useState<'summary' | 'trades' | 'chart'>('summary');
  const [showFilteredOnly, setShowFilteredOnly] = React.useState<boolean>(true);

  const buildDailyStats = React.useCallback(() => {
    const daily: Record<string, { count: number; notional: number }> = {};
    let maxCount = 0;
    let maxNotional = 0;

    const cappedTrades = trades || [];
    for (const t of cappedTrades) {
      const entryRaw = t.features?.entry_date || t.features?.trade_date || t.created_at;
      const exitRaw = t.features?.exit_date || t.features?.trade_date || t.created_at;
      const entry = new Date(entryRaw);
      const exit = new Date(exitRaw);
      if (!Number.isFinite(entry.getTime()) || !Number.isFinite(exit.getTime())) continue;

      const start = entry < exit ? entry : exit; // handle reversed inputs
      const end = exit > entry ? exit : entry;
      // Avoid runaway loops: cap at 400 days
      const maxDays = 400;
      let steps = 0;
      const cursor = new Date(start);
      const size = Math.max(0, Number((t.features as any)?.position_size ?? (t.features as any)?.qty ?? 1) || 0);
      const entryPrice = Math.max(0, Number(t.entry_price) || 0);
      const notionalPerDay = entryPrice * (size || 1);

      while (cursor <= end && steps < maxDays) {
        const key = cursor.toISOString().slice(0, 10);
        if (!daily[key]) daily[key] = { count: 0, notional: 0 };
        daily[key].count += 1;
        daily[key].notional += notionalPerDay;
        maxCount = Math.max(maxCount, daily[key].count);
        maxNotional = Math.max(maxNotional, daily[key].notional);
        cursor.setDate(cursor.getDate() + 1);
        steps += 1;
      }
    }

    const dailyArray = Object.entries(daily)
      .map(([date, v]) => ({ date, ...v }))
      .sort((a, b) => (a.date < b.date ? -1 : 1));

    return { dailyArray, maxCount, maxNotional };
  }, [trades]);

  React.useEffect(() => {
    if (!trades || trades.length === 0) {
      setActualRange(null);
      return;
    }
    const dates = trades
      .map((t: any) => t.features?.entry_date || t.features?.trade_date || t.created_at)
      .filter(Boolean)
      .map((d: any) => new Date(d))
      .filter((d: Date) => Number.isFinite(d.getTime()));
    if (dates.length === 0) {
      setActualRange(null);
      return;
    }
    dates.sort((a, b) => a.getTime() - b.getTime());
    const start = dates[0];
    const end = dates[dates.length - 1];
    const days = Math.max(0, Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)));
    setActualRange({
      start: start.toISOString().slice(0, 10),
      end: end.toISOString().slice(0, 10),
      days,
    });
  }, [trades]);
  const filteredTrades = React.useMemo(() => {
    if (!showFilteredOnly) return trades;
    return (trades || []).filter(t => {
      const status = t.features?.backtest_status || t.Status || t.status || 'Accepted';
      return status === 'Accepted';
    });
  }, [trades, showFilteredOnly]);

  if (!isOpen || !bt) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="bg-zinc-950 border border-white/10 rounded-3xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl animate-in fade-in zoom-in-95 duration-300">
        {/* Header */}
        <div className="p-4 border-b border-white/5 flex items-center justify-between bg-zinc-900/40 backdrop-blur-md">
          <div className="flex items-center gap-4 flex-1 min-w-0">
            <div className="p-2.5 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 shadow-lg shadow-indigo-500/5 flex-shrink-0">
              <LineChart className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="text-base font-black text-white tracking-tight truncate leading-none">
                  {bt.model_name?.replace(".pkl", "")}
                </h3>
                <span className="text-zinc-500 text-[9px] font-black px-1.5 py-0.5 rounded bg-white/5 border border-white/5 flex-shrink-0 uppercase tracking-wider">{bt.exchange}</span>
              </div>
              <p className="text-[9px] text-zinc-500 mt-1 uppercase tracking-[0.2em] font-black truncate opacity-60">
                Interactive Analysis Layer
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 flex-shrink-0">
            {/* Intelligent Toggle Group */}
            <div className="flex items-center bg-zinc-950/60 p-1 rounded-xl border border-white/5 shadow-inner">
              <button
                onClick={() => setShowFilteredOnly(true)}
                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all flex items-center gap-2 ${showFilteredOnly ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 shadow-lg shadow-emerald-500/10' : 'text-zinc-600 hover:text-zinc-400'}`}
              >
                <Eye className="h-3 w-3" />
                Filtered
              </button>
              <button
                onClick={() => setShowFilteredOnly(false)}
                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all flex items-center gap-2 ${!showFilteredOnly ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 shadow-lg shadow-indigo-500/10' : 'text-zinc-600 hover:text-zinc-400'}`}
              >
                <EyeOff className="h-3 w-3" />
                Raw Data
              </button>
            </div>

            <div className="h-6 w-px bg-white/5" />

            {/* Navigation Tabs */}
            <div className="flex bg-zinc-950/60 p-1 rounded-xl border border-white/5 shadow-inner">
              <button
                onClick={() => setActiveTab('summary')}
                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${activeTab === 'summary' ? 'bg-white/10 text-white' : 'text-zinc-600 hover:text-zinc-400'}`}
              >
                Summary
              </button>
              <button
                onClick={() => setActiveTab('trades')}
                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${activeTab === 'trades' ? 'bg-white/10 text-white' : 'text-zinc-600 hover:text-zinc-400'}`}
              >
                Trades
              </button>
              <button
                onClick={() => setActiveTab('chart')}
                className={`px-3 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${activeTab === 'chart' ? 'bg-white/10 text-white' : 'text-zinc-600 hover:text-zinc-400'}`}
              >
                Chart
              </button>
            </div>

            <button onClick={onClose} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-zinc-500 hover:text-white transition-all border border-white/5">
              <ChevronDown className="h-4 w-4 rotate-180" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6 custom-scrollbar">
          {activeTab === 'summary' ? (
            <div className="space-y-8 animate-in fade-in zoom-in-95 duration-300">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Strategy Only Analysis */}
                <div className="rounded-2xl border border-white/5 bg-zinc-900/40 p-6 space-y-4 shadow-inner">
                  <div className="flex items-center justify-between border-b border-white/5 pb-2">
                    <h4 className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Strategy Only</h4>
                    <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500 text-[8px] font-bold uppercase">Pre-Council</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-1">
                      <span className="text-[9px] text-zinc-500 font-bold uppercase">Trades</span>
                      <div className="text-lg font-mono font-black text-white">{bt.pre_council_trades || bt.total_trades}</div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[9px] text-zinc-500 font-bold uppercase">Win Rate</span>
                      <div className="text-lg font-mono font-black text-white">{bt.pre_council_win_rate ? `${bt.pre_council_win_rate.toFixed(1)}%` : (bt.win_rate ? `${Number(bt.win_rate).toFixed(1)}%` : '—')}</div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[9px] text-zinc-500 font-bold uppercase">Profit</span>
                      <div className={`text-lg font-mono font-black ${(Number(bt.pre_council_profit_pct) || Number(bt.profit_pct) || 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {bt.pre_council_profit_pct ? `${Number(bt.pre_council_profit_pct).toFixed(1)}%` : (bt.profit_pct ? `${Number(bt.profit_pct).toFixed(1)}%` : '—')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* With Filter Analysis */}
                <div className="rounded-2xl border border-indigo-500/20 bg-indigo-500/5 p-6 space-y-4 shadow-inner backdrop-blur-sm">
                  <div className="flex items-center justify-between border-b border-indigo-500/10 pb-2">
                    <h4 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em]">With {bt.council_model?.replace('.pkl', '') || 'Filter'}</h4>
                    <span className="px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 text-[8px] font-bold uppercase">Post-Council</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-1">
                      <span className="text-[9px] text-indigo-400/60 font-bold uppercase">Trades</span>
                      <div className="text-lg font-mono font-black text-white">{bt.post_council_trades || bt.total_trades}</div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[9px] text-indigo-400/60 font-bold uppercase">Win Rate</span>
                      <div className="text-lg font-mono font-black text-emerald-400">{bt.post_council_win_rate ? `${bt.post_council_win_rate.toFixed(1)}%` : (bt.win_rate ? `${Number(bt.win_rate).toFixed(1)}%` : '—')}</div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[9px] text-indigo-400/60 font-bold uppercase">Profit</span>
                      <div className={`text-lg font-mono font-black ${(Number(bt.post_council_profit_pct) || Number(bt.profit_pct) || 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {bt.post_council_profit_pct ? `${Number(bt.post_council_profit_pct).toFixed(1)}%` : (bt.profit_pct ? `${Number(bt.profit_pct).toFixed(1)}%` : '—')}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <CouncilAuditPanel bt={bt} />

              <div className="flex items-center justify-center gap-16 p-8 rounded-2xl bg-white/[0.02] border border-white/5">
                <div className="flex flex-col items-center">
                  <span className="text-[9px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-2">Trade Reduction</span>
                  <div className="text-3xl font-black text-white">
                    {bt.pre_council_trades && bt.post_council_trades ?
                      `-${Math.round(((bt.pre_council_trades - bt.post_council_trades) / bt.pre_council_trades) * 100)}%` :
                      '—'}
                  </div>
                </div>
                <div className="w-px h-12 bg-white/5" />
                <div className="flex flex-col items-center">
                  <span className="text-[9px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-2">Win Rate Boost</span>
                  <div className={`text-3xl font-black ${Number(bt.post_council_win_rate) - Number(bt.pre_council_win_rate) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                    {bt.pre_council_win_rate && bt.post_council_win_rate ?
                      `+${(bt.post_council_win_rate - bt.pre_council_win_rate).toFixed(1)}pp` :
                      '—'}
                  </div>
                </div>
                <div className="w-px h-12 bg-white/5" />
                <div className="flex flex-col items-center">
                  <span className="text-[9px] font-black text-zinc-500 uppercase tracking-[0.2em] mb-2">Actual Range</span>
                  <div className="text-sm font-black text-white">
                    {actualRange ? `${actualRange.start} → ${actualRange.end}` : "—"}
                  </div>
                  <div className="text-[10px] font-bold text-zinc-500">
                    {actualRange ? `${actualRange.days} days` : ""}
                  </div>
                </div>
              </div>
            </div>
          ) : activeTab === 'trades' ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-300">
              {loading ? (
                <div className="flex flex-col items-center justify-center h-64 gap-4">
                  <Activity className="h-10 w-10 text-indigo-500 animate-spin" />
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Parsing voting records...</span>
                </div>
              ) : (
                <div className="rounded-2xl border border-white/5 overflow-hidden">
                  <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-zinc-900/90 backdrop-blur z-10">
                      <tr className="border-b border-white/10">
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-left">Asset</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-left">Entry Date</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-left">Exit Date</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-center">Timing</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-left">Pricing</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-center">Radar</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-center">Fund</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-right whitespace-nowrap">P/L %</th>
                        <th className="px-6 py-4 text-[10px] font-black text-zinc-500 uppercase tracking-widest text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {filteredTrades.map((t: any, i: number) => {
                        const status = t.features?.backtest_status || 'Accepted';
                        const isWin = t.profit_loss_pct > 0;
                        const isRejected = status === 'Rejected';

                        return (
                          <tr key={i} className={`group hover:bg-white/[0.02] transition-colors ${isRejected ? 'opacity-40 grayscale-[0.8]' : ''}`}>
                            <td className="px-6 py-4 text-left">
                              <span className="text-sm font-black text-white">{t.symbol}</span>
                            </td>
                            <td className="px-6 py-4 text-left">
                              <span className="text-xs font-mono text-zinc-400">{t.features?.entry_date || t.features?.trade_date || t.created_at?.slice(0, 10) || 'N/A'}</span>
                            </td>
                            <td className="px-6 py-4 text-left">
                              <span className="text-xs font-mono text-zinc-400">{t.features?.exit_date || t.features?.trade_date || t.created_at?.slice(0, 10) || 'N/A'}</span>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <div className="flex flex-col items-center">
                                <span className="text-xs font-mono text-zinc-300">
                                  {(() => {
                                    const entryDate = t.features?.entry_date || t.features?.trade_date;
                                    const exitDate = t.features?.exit_date || t.features?.trade_date;
                                    if (!entryDate || !exitDate) return 'N/A';
                                    try {
                                      const entry = new Date(entryDate).getTime();
                                      const exit = new Date(exitDate).getTime();
                                      if (!Number.isFinite(entry) || !Number.isFinite(exit)) return 'N/A';
                                      const days = Math.ceil((exit - entry) / (1000 * 60 * 60 * 24));
                                      return days >= 0 ? `${days}d` : 'N/A';
                                    } catch { return 'N/A'; }
                                  })()}
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-left">
                              <div className="flex flex-col font-mono text-[11px]">
                                <span className="text-zinc-500">In: {(t.entry_price || 0) < 0.1 ? (t.entry_price || 0).toFixed(6) : (t.entry_price || 0).toFixed(2)}</span>
                                <span className="text-zinc-300 font-bold">Out: {(t.exit_price || 0) < 0.1 ? (t.exit_price || 0).toFixed(6) : (t.exit_price || 0).toFixed(2)}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <span className="text-xs font-mono font-bold text-zinc-200">
                                {(() => {
                                  let radarScore = (t.features as any)?.radar_score ?? (t.features as any)?.ai_score ?? (t.features as any)?.score ?? (t as any)?.score ?? (t as any)?.Score;
                                  if (radarScore === null || radarScore === undefined || Number.isNaN(Number(radarScore))) return '—';
                                  const n = Number(radarScore);
                                  return n <= 1 ? `${(n * 100).toFixed(1)}%` : `${n.toFixed(1)}%`;
                                })()}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <span className="text-xs font-mono font-bold text-zinc-200">
                                {(() => {
                                  const fundScore = (t.features as any)?.fund_score ?? (t.features as any)?.fundamental_score ?? (t.features as any)?.validator_score ?? (t as any)?.fund_score ?? (t as any)?.Validator_Score;
                                  if (fundScore === null || fundScore === undefined || Number.isNaN(Number(fundScore))) return '—';
                                  const n = Number(fundScore);
                                  return n <= 1 ? `${(n * 100).toFixed(1)}%` : `${n.toFixed(1)}%`;
                                })()}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right">
                              <span className={`text-sm font-mono font-black ${isWin ? 'text-emerald-400' : (t.profit_loss_pct === 0 ? 'text-zinc-500' : 'text-red-400')}`}>
                                {t.profit_loss_pct > 0 ? '+' : ''}{t.profit_loss_pct.toFixed(1)}%
                              </span>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-black uppercase ${isRejected ? 'bg-zinc-800 text-zinc-500' : (isWin ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400')}`}>
                                {isRejected ? 'Filter' : (t.status || 'Closed')}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            // Chart tab
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-300">
              <TradeTimeline trades={filteredTrades} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


export default function BacktestTab() {
  const { inventory } = useAppState();

  const normalizeThreshold01 = (value: number): number => {
    if (!Number.isFinite(value)) return 0;
    const raw = Number(value);
    const v = raw > 1 ? raw / 100 : raw;
    return Math.min(1, Math.max(0, v));
  };

  // State
  const [models, setModels] = useState<(string | LocalModelMeta)[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedExchange, setSelectedExchange] = useState("EGX");
  const [selectedCouncilModel, setSelectedCouncilModel] = useState<string | null>(null);
  const [startDate, setStartDate] = useState(new Date().toISOString().slice(0, 10));
  const [endDate, setEndDate] = useState(new Date().toISOString().slice(0, 10));
  const [metaThreshold, setMetaThreshold] = useState<number>(0.6);
  const [councilThreshold, setCouncilThreshold] = useState<number>(0.1);

  const [running, setRunning] = useState(false);
  const [currentBacktestId, setCurrentBacktestId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [indexFallbackById, setIndexFallbackById] = useState<Record<string, number | null>>({});

  // Trades dialog state
  const [tradesOpen, setTradesOpen] = useState(false);

  const [tradesLoading, setTradesLoading] = useState(false);
  const [tradesRows, setTradesRows] = useState<any[]>([]);
  const [tradesTitle, setTradesTitle] = useState<string>("");
  const [viewingBacktest, setViewingBacktest] = useState<any>(null);

  // Multi-selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [confirmConfig, setConfirmConfig] = useState<{
    title: string;
    message: string;
    onConfirm: () => void;
    isLoading: boolean;
  }>({
    title: "",
    message: "",
    onConfirm: () => { },
    isLoading: false,
  });

  // Load models on mount
  useEffect(() => {
    async function loadModels() {
      setModelsLoading(true);
      try {
        const data = await getLocalModels();
        setModels(data);
        if (data.length > 0) {
          const names = data.map((m) => (typeof m === "string" ? m : m.name));

          const defaultAi =
            names.find((n) => n === "KING 👑.pkl") ??
            names.find((n) => n.includes("KING")) ??
            names[0];
          setSelectedModel(defaultAi ?? null);

          const defaultCouncil =
            names.find((n) => n === "The_Council_Validator.pkl") ??
            names.find((n) => n.toLowerCase().includes("council")) ??
            names.find((n) => n.includes("KING")) ??
            (names.length > 1 ? names[1] : null);
          setSelectedCouncilModel(defaultCouncil);
        }
      } catch (err) {
        console.error("Failed to load models:", err);
      } finally {
        setModelsLoading(false);
      }
    }
    loadModels();
    loadHistory();
  }, []);

  // Polling for status updates
  useEffect(() => {
    let interval: any;
    if (running && currentBacktestId) {
      interval = setInterval(async () => {
        try {
          const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
          const res = await fetch(`${baseUrl}/backtests`); // Refresh full history
          if (!res.ok) return;
          const data = await res.json();
          setHistory(data);

          // Find current one
          const current = data.find((b: any) => b.id === currentBacktestId);
          if (current) {
            setStatusMsg(current.status_msg);
            if (current.status === "completed") {
              setRunning(false);
              setCurrentBacktestId(null);
              setStatusMsg(null);
              toast.success("Simulation Completed", {
                description: `Successfully processed ${current.total_trades} trades.`
              });
              clearInterval(interval);
            } else if (current.status === "failed") {
              setRunning(false);
              setCurrentBacktestId(null);
              setStatusMsg(null);
              setError(current.status_msg || "Simulation failed");
              toast.error("Simulation Failed");
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [running, currentBacktestId]);

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      const data = await getBacktests();
      setHistory(data);
    } catch (err) {
      console.error("Failed to load backtest history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    if (!history || history.length === 0) return;
    const missing = history.filter((bt) => {
      const hasValue =
        (bt as any).benchmark_return_pct !== null &&
        (bt as any).benchmark_return_pct !== undefined &&
        !Number.isNaN(Number((bt as any).benchmark_return_pct));
      const hasWinRate =
        (bt as any).benchmark_win_rate !== null &&
        (bt as any).benchmark_win_rate !== undefined &&
        !Number.isNaN(Number((bt as any).benchmark_win_rate));
      return !hasValue && !hasWinRate && (bt as any).exchange === "EGX" && bt.start_date && bt.end_date;
    });
    if (missing.length === 0) return;

    let cancelled = false;
    (async () => {
      const updates: Record<string, number | null> = {};
      await Promise.all(
        missing.map(async (bt: any) => {
          try {
            const res = await fetch(`/api/egx30/range?start=${bt.start_date}&end=${bt.end_date}`);
            const json = await res.json();
            const val = res.ok ? Number(json?.return_pct) : null;
            updates[bt.id] = Number.isFinite(val) ? val : null;
          } catch {
            updates[bt.id] = null;
          }
        })
      );
      if (cancelled) return;
      setIndexFallbackById((prev) => ({ ...prev, ...updates }));
    })();

    return () => {
      cancelled = true;
    };
  }, [history]);

  // Get unique exchanges from inventory
  const exchanges = Array.from(new Set(inventory.map((i) => i.exchange).filter(Boolean))) as string[];

  const handleRun = async () => {
    if (!selectedModel) return;

    setRunning(true);
    setError(null); // Clear previous errors
    setResult(null);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      const payload = {
        exchange: selectedExchange,
        model: selectedModel,
        start_date: startDate,
        end_date: endDate,
        council_model: selectedCouncilModel,
        council_threshold: normalizeThreshold01(councilThreshold),
        meta_threshold: normalizeThreshold01(metaThreshold),
      };
      // Intentionally silent (avoid noisy debug logs in the browser console)

      const res = await fetch(`${baseUrl}/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Backtest failed (${res.status})`);
      }

      if (data.status === "queued") {
        setCurrentBacktestId(data.id);
        setStatusMsg("Queued...");
        toast.info("Simulation Started", {
          description: data.message,
        });
      } else if (data.totalTrades !== undefined) {
        setResult(data);
        loadHistory();
        setRunning(false);
      } else {
        toast.error("Simulation Failed", {
          description: data.detail || "Unknown error occurred",
        });
        setRunning(false);
      }
    } catch (err: any) {
      setError(err.message || "Backtest failed");
      toast.error("Backtest Failed", {
        description: err.message || "An unexpected error occurred.",
      });
      setRunning(false);
      loadHistory();
    }
  };

  const handleOpenTrades = async (bt: any) => {
    setTradesOpen(true);
    setTradesLoading(true);
    setTradesRows([]);
    const title = `${bt.model_name || "Model"} • ${bt.exchange || ""}`;
    setTradesTitle(title);
    setViewingBacktest(bt);
    try {
      const data = await getBacktestTrades(bt.id);
      setTradesRows(Array.isArray(data) ? data : []);
    } catch (err: any) {
      toast.error("Failed to load trades", { description: err.message || "Unknown error" });
    } finally {
      setTradesLoading(false);
    }
  };

  const handleToggleVisibility = async (id: string, current: boolean) => {
    try {
      await updateBacktestVisibility(id, !current);
      toast.success("Visibility Updated", {
        description: `Backtest is now ${!current ? "Public" : "Hidden"}`
      });
      loadHistory();
    } catch (err: any) {
      toast.error("Failed to update visibility", {
        description: err.message
      });
    }
  };

  const handleDelete = async (id: string) => {
    setConfirmConfig({
      title: "Delete Backtest?",
      message: "Are you sure you want to delete this simulation? This action cannot be undone.",
      isLoading: false,
      onConfirm: async () => {
        setConfirmConfig(prev => ({ ...prev, isLoading: true }));
        try {
          await deleteBacktest(id);
          toast.success("Deleted Successfully");
          setSelectedIds(prev => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
          loadHistory();
          setIsConfirmOpen(false);
        } catch (err: any) {
          toast.error("Delete failed", { description: err.message });
        } finally {
          setConfirmConfig(prev => ({ ...prev, isLoading: false }));
        }
      }
    });
    setIsConfirmOpen(true);
  };

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return;

    setConfirmConfig({
      title: `Delete ${selectedIds.size} Items?`,
      message: `Are you sure you want to permanently delete these ${selectedIds.size} backtest records?`,
      isLoading: false,
      onConfirm: async () => {
        setConfirmConfig(prev => ({ ...prev, isLoading: true }));
        let count = 0;
        let failCount = 0;
        for (const id of Array.from(selectedIds)) {
          try {
            await deleteBacktest(id);
            count++;
          } catch {
            failCount++;
          }
        }
        toast.success(`Deleted ${count} items`, {
          description: failCount > 0 ? `Failed to delete ${failCount} items.` : undefined
        });
        setSelectedIds(new Set());
        loadHistory();
        setIsConfirmOpen(false);
        setConfirmConfig(prev => ({ ...prev, isLoading: false }));
      }
    });
    setIsConfirmOpen(true);
  };

  const handleBulkVisibility = async (visible: boolean) => {
    if (selectedIds.size === 0) return;

    setConfirmConfig({
      title: `${visible ? "Publish" : "Hide"} ${selectedIds.size} Items?`,
      message: `Change visibility for ${selectedIds.size} selected items to ${visible ? "Public" : "Hidden"}?`,
      isLoading: false,
      onConfirm: async () => {
        setConfirmConfig(prev => ({ ...prev, isLoading: true }));
        let count = 0;
        for (const id of Array.from(selectedIds)) {
          try {
            await updateBacktestVisibility(id, visible);
            count++;
          } catch { }
        }
        toast.success(`Updated ${count} items`);
        loadHistory();
        setIsConfirmOpen(false);
        setConfirmConfig(prev => ({ ...prev, isLoading: false }));
      }
    });
    setIsConfirmOpen(true);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === history.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(history.map(b => b.id)));
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };


  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 shadow-lg shadow-emerald-500/5">
          <Activity className="h-6 w-6 text-emerald-400" />
        </div>
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight uppercase">Simulation <span className="text-emerald-500">Radar</span></h2>
          <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]">Deep Learning historical engine</p>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="relative group overflow-hidden rounded-[2.5rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl p-1 shadow-2xl transition-all duration-500 hover:shadow-indigo-500/10">
        {/* Decorative background elements */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-600/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-emerald-600/5 blur-[120px] rounded-full translate-y-1/2 -translate-x-1/2" />

        <div className="relative bg-zinc-900/40 rounded-[2.4rem] p-8 space-y-10">
          {/* Section 1: Core Configuration */}
          <div className="space-y-6">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-4 w-1 bg-indigo-500 rounded-full" />
              <h3 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Core Network Settings</h3>
            </div>

            <div className="grid gap-8 md:grid-cols-3">
              {/* Model Selector */}
              <div className="space-y-3">
                <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                  <Database className="h-3 w-3" /> AI Model
                </label>
                <select
                  value={selectedModel || ""}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={modelsLoading}
                  className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all cursor-pointer hover:border-emerald-500/30"
                >
                  {modelsLoading ? (
                    <option className="bg-zinc-900 text-white">Loading intelligence units...</option>
                  ) : (
                    models.map((m) => {
                      const name = typeof m === "string" ? m : m.name;
                      return (
                        <option key={name} value={name} className="bg-zinc-900 text-white">
                          {name}
                        </option>
                      );
                    })
                  )}
                </select>
              </div>

              {/* Council Selector */}
              <div className="space-y-3">
                <label className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                  <Users className="h-3 w-3" /> Council Filter
                </label>
                <select
                  value={selectedCouncilModel || ""}
                  onChange={(e) => setSelectedCouncilModel(e.target.value)}
                  disabled={modelsLoading}
                  className="w-full h-14 px-5 rounded-2xl border border-indigo-500/10 bg-indigo-500/5 text-white font-mono text-sm focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all cursor-pointer hover:border-indigo-500/30"
                >
                  <option value="" className="bg-zinc-900 text-white">Direct Execution (No Council)</option>
                  {models.map((m) => {
                    const name = typeof m === "string" ? m : m.name;
                    return (
                      <option key={name} value={name} className="bg-zinc-900 text-white">
                        {name}
                      </option>
                    );
                  })}
                </select>
              </div>

              {/* Exchange Selector */}
              <div className="space-y-3">
                <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                  <Globe className="h-3 w-3" /> Market Exchange
                </label>
                <select
                  value={selectedExchange}
                  onChange={(e) => setSelectedExchange(e.target.value)}
                  className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all cursor-pointer hover:border-emerald-500/30"
                >
                  <option value="EGX" className="bg-zinc-900 text-white">EGX (Egypt)</option>
                  <option value="US" className="bg-zinc-900 text-white">US (United States)</option>
                  <option value="CRYPTO" className="bg-zinc-900 text-white">CRYPTO (Crypto)</option>
                  {exchanges
                    .filter((e) => e !== "EGX" && e !== "US" && e !== "CRYPTO")
                    .map((ex) => (
                      <option key={ex} value={ex} className="bg-zinc-900 text-white">
                        {ex}
                      </option>
                    ))}
                </select>
              </div>
            </div>
          </div>

          {/* Section 2: Intelligence Filters & Temporal Range */}
          <div className="pt-6 border-t border-white/5 grid gap-12 md:grid-cols-2">
            {/* Filters Group */}
            <div className="space-y-6">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-emerald-500 rounded-full" />
                <h3 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Intelligence Optimization</h3>
              </div>

              <div className="grid gap-6">
                {/* Meta Threshold */}
                <div className="space-y-3">
                  <label className="text-[9px] font-bold text-emerald-400 uppercase tracking-widest flex items-center justify-between">
                    <span className="flex items-center gap-2"><Cpu className="h-3 w-3" /> Meta Confidence</span>
                    <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">Scale: 0.0 — 1.0</span>
                  </label>
                  <div className="relative group/input">
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={metaThreshold}
                      onChange={(e) => setMetaThreshold(Number(e.target.value))}
                      className="w-full h-14 pl-5 pr-12 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-lg focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all"
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500 font-mono text-xs">0-1</div>
                  </div>
                </div>

                {/* Council Threshold - Smart Conditional Logic */}
                {selectedCouncilModel && (
                  <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-500">
                    <label className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest flex items-center justify-between">
                      <span className="flex items-center gap-2"><ShieldCheck className="h-3 w-3" /> Council consensus</span>
                      <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">Scale: 0.0 — 1.0</span>
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={councilThreshold}
                        onChange={(e) => setCouncilThreshold(Number(e.target.value))}
                        className="w-full h-14 pl-5 pr-12 rounded-2xl border border-indigo-500/20 bg-indigo-500/5 text-white font-mono text-lg focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                      />
                      <div className="absolute right-4 top-1/2 -translate-y-1/2 text-indigo-400/60 font-mono text-xs">0-1</div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Date Range Group */}
            <div className="space-y-6">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-zinc-500 rounded-full" />
                <h3 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Temporal Bounds</h3>
              </div>

              <div className="grid gap-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                      <Calendar className="h-3 w-3" /> Start Date
                    </label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                      <Calendar className="h-3 w-3" /> End Date
                    </label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all"
                    />
                  </div>
                </div>

                {/* EGX Component */}
                {selectedExchange === "EGX" && (
                  <Egx30MonthlyChart
                    onSelectMonth={({ startDate, endDate }) => {
                      setStartDate(startDate);
                      setEndDate(endDate);
                    }}
                  />
                )}
              </div>
            </div>
          </div>

          {/* Action Button and Status */}
          <div className="pt-8 border-t border-white/5 space-y-4">
            <div className="flex gap-4">
              <button
                onClick={handleRun}
                disabled={running || !selectedModel || modelsLoading}
                className={`flex-1 group/btn relative h-16 rounded-[1.25rem] overflow-hidden transition-all duration-500 ${running || !selectedModel || modelsLoading ? 'cursor-not-allowed opacity-50' : 'hover:scale-[1.01] active:scale-[0.99] shadow-xl hover:shadow-indigo-500/10'}`}
              >
                {/* Button Background Gradient */}
                <div className={`absolute inset-0 bg-gradient-to-r ${running ? 'from-zinc-800 to-zinc-900' : 'from-indigo-600 to-violet-600'} transition-all duration-500`} />

                {/* Hover shine effect */}
                <div className="absolute inset-0 opacity-0 group-hover/btn:opacity-100 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover/btn:animate-[shimmer_2s_infinite] transition-all" />

                <div className="relative flex items-center justify-center gap-3">
                  {modelsLoading ? (
                    <>
                      <Activity className="h-5 w-5 animate-spin text-white/50" />
                      <span className="text-sm font-black text-white uppercase tracking-[0.2em]">Synchronizing Intelligence...</span>
                    </>
                  ) : running ? (
                    <>
                      <div className="relative h-6 w-6">
                        <div className="absolute inset-0 rounded-full border-2 border-white/20" />
                        <div className="absolute inset-0 rounded-full border-2 border-t-white animate-spin" />
                      </div>
                      <span className="text-sm font-black text-white uppercase tracking-[0.2em]">Executing Simulation Layer...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="h-5 w-5 text-white fill-white animate-pulse" />
                      <span className="text-sm font-black text-white uppercase tracking-[0.2em]">{!selectedModel ? "Select Model to Initiate" : "Initiate Full Backtest"}</span>
                    </>
                  )}
                </div>
              </button>

              {result && (
                <button
                  onClick={() => setTradesOpen(true)}
                  className="h-16 px-8 rounded-[1.25rem] border border-white/10 bg-white/5 text-white hover:bg-white/10 transition-all flex items-center gap-3 font-black text-[10px] uppercase tracking-widest shadow-xl"
                >
                  <Activity className="h-4 w-4 text-emerald-400" />
                  View Analysis
                </button>
              )}
            </div>

            {statusMsg && (
              <div className="flex items-center gap-3 p-4 rounded-2xl bg-emerald-500/5 border border-emerald-500/10 text-emerald-400 text-[10px] font-mono animate-in fade-in slide-in-from-left-2 duration-500">
                <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span className="uppercase tracking-[0.1em] font-black opacity-60">Engine Status:</span>
                <span className="font-bold">{statusMsg}</span>
              </div>
            )}

            {error && (
              <div className="p-4 rounded-2xl bg-red-500/5 border border-red-500/10 flex items-center gap-3 text-red-400 text-xs font-bold animate-in shake duration-500">
                <AlertTriangle className="h-4 w-4" />
                {error}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="relative overflow-hidden rounded-3xl border border-emerald-500/20 bg-zinc-900/60 backdrop-blur-3xl p-8 shadow-2xl animate-in fade-in slide-in-from-bottom-8 duration-700">
          <div className="absolute -top-24 -left-24 h-48 w-48 bg-emerald-500/10 blur-[100px]" />

          <div className="relative flex items-center gap-4 mb-8">
            <div className="p-3 rounded-2xl bg-emerald-500/20 border border-emerald-500/30">
              <CheckCircle2 className="h-6 w-6 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-xl font-black text-white uppercase tracking-tight">Backtest <span className="text-emerald-500">Insights</span></h3>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Performance breakdown for {selectedModel}</p>
            </div>
          </div>

          <div className="relative grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="group rounded-2xl border border-white/5 bg-white/[0.02] p-6 text-center hover:bg-white/[0.04] transition-all">
              <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3 group-hover:text-zinc-400 transition-colors">Total Trades</div>
              <div className="text-4xl font-black text-white font-mono tracking-tighter">{result.totalTrades.toLocaleString()}</div>
            </div>
            <div className="group rounded-2xl border border-emerald-500/10 bg-emerald-500/[0.02] p-6 text-center hover:bg-emerald-500/[0.04] transition-all">
              <div className="text-[10px] font-black text-emerald-500/60 uppercase tracking-widest mb-3 group-hover:text-emerald-500 transition-colors">Win Rate</div>
              <div className="text-4xl font-black text-emerald-400 font-mono tracking-tighter">{result.winRate.toFixed(1)}%</div>
            </div>
            <div className="group rounded-2xl border border-white/5 bg-white/[0.02] p-6 text-center hover:bg-white/[0.04] transition-all">
              <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3 group-hover:text-zinc-400 transition-colors">Avg Return</div>
              <div className="text-4xl font-black text-white font-mono tracking-tighter">{result.avgReturnPerTrade.toFixed(2)}%</div>
            </div>
            <div className="group rounded-2xl border border-white/5 bg-white/[0.02] p-6 text-center hover:bg-white/[0.04] transition-all">
              <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-3 group-hover:text-zinc-400 transition-colors">Net Profit</div>
              <div className={`text-4xl font-black font-mono tracking-tighter ${result.netProfit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {(result.netProfit || 0).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* History Section */}
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 shadow-lg shadow-indigo-500/5">
              <HistoryIcon className="h-6 w-6 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-2xl font-black text-white tracking-tight uppercase">Historical <span className="text-indigo-500">Archive</span></h3>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]">Audit and manage previous runs</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {selectedIds.size > 0 && (
              <div className="flex items-center gap-2 p-1 bg-white/5 rounded-2xl animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="px-3 text-[10px] font-black text-indigo-400 uppercase tracking-widest border-r border-white/10 mr-1">
                  {selectedIds.size} SELECTED
                </div>
                <button
                  onClick={() => handleBulkVisibility(true)}
                  className="p-2.5 rounded-xl text-zinc-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-all"
                  title="Make Public"
                >
                  <Eye className="h-4 w-4" />
                </button>
                <button
                  onClick={() => handleBulkVisibility(false)}
                  className="p-2.5 rounded-xl text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-all"
                  title="Hide All"
                >
                  <EyeOff className="h-4 w-4" />
                </button>
                <button
                  onClick={handleBulkDelete}
                  className="p-2.5 rounded-xl text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
                  title="Delete Selected"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            )}
            <button
              onClick={loadHistory}
              disabled={historyLoading}
              className={`p-3.5 rounded-2xl bg-white/5 text-zinc-400 hover:text-white transition-all border border-white/10 ${historyLoading ? "animate-spin text-indigo-500" : ""}`}
            >
              <Activity className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-white/5 bg-zinc-950/40 overflow-hidden overflow-x-auto">
          <table className="w-full text-[10px] text-left">
            <thead className="bg-zinc-900/80 text-zinc-500 font-black uppercase tracking-widest border-b border-white/5">
              <tr>
                <th className="px-6 py-4 text-center">
                  <input
                    type="checkbox"
                    checked={history.length > 0 && selectedIds.size === history.length}
                    onChange={toggleSelectAll}
                    className="rounded border-zinc-800 bg-zinc-950 text-indigo-500 focus:ring-indigo-500/20"
                  />
                </th>

                <th className="px-6 py-4 text-left">Model & Exchange</th>
                <th className="px-6 py-4 text-center">Thresholds</th>
                <th className="px-6 py-4 text-center">Period</th>
                <th className="px-6 py-4 text-center">Trades</th>
                <th className="px-6 py-4 text-center">Win Rate</th>
                <th className="px-6 py-4 text-right">Profit %</th>
                <th className="px-6 py-4 text-right">Profit (Cash)</th>
                <th className="px-6 py-4 text-right">Index Win Rate</th>
                <th className="px-6 py-4 text-center">Public</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-400">
              {historyLoading ? (
                <tr>
                  <td colSpan={11} className="px-6 py-10 text-center animate-pulse">Retrieving archive data...</td>
                </tr>
              ) : history.length === 0 ? (
                <tr>
                  <td colSpan={11} className="px-6 py-10 text-center text-zinc-600">No historical records found</td>
                </tr>
              ) : (
                history.map((bt) => (
                  <React.Fragment key={bt.id}>
                    <tr className={`hover:bg-white/[0.02] transition-colors group ${selectedIds.has(bt.id) ? "bg-indigo-500/[0.03]" : ""}`}>
                      <td className="px-6 py-4 text-center">
                        <input
                          type="checkbox"
                          checked={selectedIds.has(bt.id)}
                          onChange={() => toggleSelect(bt.id)}
                          className="rounded border-zinc-800 bg-zinc-950 text-indigo-500 focus:ring-indigo-500/20"
                        />
                      </td>

                      <td className="px-6 py-4">
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-2">
                            <span className="text-white font-black">{bt.model_name?.replace(".pkl", "") || "N/A"}</span>
                            {bt.status === "processing" && <Activity className="h-3 w-3 text-emerald-400 animate-spin" />}
                          </div>
                          <span className="text-[8px] font-bold text-zinc-600 uppercase tracking-widest">{bt.exchange} • {new Date(bt.created_at).toLocaleDateString()}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="flex flex-col items-center gap-1">
                          {bt.council_model ? (
                            <span className="px-2 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-mono text-[9px] uppercase tracking-tighter">
                              {bt.council_model.replace(".pkl", "")}
                              <span className="ml-1 opacity-60">@{bt.council_threshold ?? 0.1}</span>
                            </span>
                          ) : (
                            <span className="text-zinc-600 uppercase text-[8px] font-black tracking-widest">NONE</span>
                          )}
                          <span className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest">
                            Meta: {bt.meta_threshold ?? 0.4}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="flex flex-col items-center">
                          <span className="font-mono text-[10px] text-zinc-400">{bt.start_date ? new Date(bt.start_date).toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }) : '—'}</span>
                          <div className="w-px h-2 bg-zinc-800 my-1" />
                          <span className="font-mono text-[10px] text-zinc-400">{bt.end_date ? new Date(bt.end_date).toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }) : '—'}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-center font-mono">
                        <button
                          onClick={() => handleOpenTrades(bt)}
                          className="flex flex-col items-center group/trades hover:bg-white/5 p-2 rounded-xl transition-all"
                          title="Click to view full analysis and trades"
                        >
                          <span className="text-zinc-500 text-[8px] font-bold uppercase mb-0.5 group-hover/trades:text-indigo-400 transition-colors">Pre/Post</span>
                          <span className="font-bold text-zinc-300">
                            {bt.pre_council_trades || bt.total_trades}
                            <span className="mx-1 opacity-20">/</span>
                            <span className="text-indigo-400 group-hover/trades:text-indigo-300 transition-colors">{bt.post_council_trades || bt.total_trades}</span>
                          </span>
                        </button>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="flex flex-col items-center gap-1">
                          <span className="font-mono text-[10px] text-zinc-500">{(bt.pre_council_win_rate || bt.win_rate || 0).toFixed(1)}%</span>
                          <div className="h-px w-4 bg-white/5" />
                          <span className="font-mono font-bold text-emerald-400">{(bt.post_council_win_rate || bt.win_rate || 0).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex flex-col items-end gap-1">
                          <span className={`font-mono text-[9px] ${(bt.pre_council_profit_pct || bt.profit_pct || 0) >= 0 ? "text-emerald-500/60" : "text-red-500/60"}`}>
                            {(bt.pre_council_profit_pct || bt.profit_pct || 0).toFixed(1)}%
                          </span>
                          <span className={`font-mono font-bold ${(bt.post_council_profit_pct || bt.profit_pct || 0) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                            {(bt.post_council_profit_pct || bt.profit_pct || 0).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right whitespace-nowrap">
                        {(() => {
                          const tradesLog = (bt as any).trades_log as any[] | undefined;
                          let totalProfitCash = 0;
                          let requiredBalance = 0;

                          if (Array.isArray(tradesLog) && tradesLog.length > 0) {
                            const daily: Record<string, number> = {};

                            for (const t of tradesLog) {
                              const entryRaw =
                                (t as any).Entry_Date ||
                                (t as any).entry_date ||
                                (t as any).features?.entry_date ||
                                (t as any).features?.trade_date;
                              const exitRaw =
                                (t as any).Exit_Date ||
                                (t as any).exit_date ||
                                (t as any).features?.exit_date ||
                                (t as any).features?.trade_date;

                              const entry = new Date(entryRaw);
                              const exit = new Date(exitRaw);
                              if (!Number.isFinite(entry.getTime()) || !Number.isFinite(exit.getTime())) continue;

                              // profit_cash
                              const pc = Number((t as any).Profit_Cash ?? (t as any).profit_cash ?? (t as any).features?.profit_cash ?? 0) || 0;
                              totalProfitCash += pc;

                              // position_cash = رأس المال المستخدم في الصفقة
                              const positionCash = Number(
                                (t as any).Position_Cash ??
                                (t as any).position_cash ??
                                (t as any).features?.position_cash ??
                                0
                              ) || 0;

                              if (positionCash <= 0) continue;

                              const start = entry < exit ? entry : exit;
                              const end = exit > entry ? exit : entry;

                              const cursor = new Date(start);
                              let steps = 0;
                              const maxDays = 400;

                              while (cursor <= end && steps < maxDays) {
                                const key = cursor.toISOString().slice(0, 10);
                                if (!daily[key]) daily[key] = 0;
                                daily[key] += positionCash;
                                if (daily[key] > requiredBalance) requiredBalance = daily[key];
                                cursor.setDate(cursor.getDate() + 1);
                                steps += 1;
                              }
                            }
                          }

                          const colorClass = (totalProfitCash || 0) >= 0 ? "text-emerald-400" : "text-red-400";

                          return (
                            <div className={`font-mono font-bold text-[10px] ${colorClass}`}>
                              <span className="text-[9px] mr-1">Bal:</span>
                              <span className="text-sm">
                                {requiredBalance > 0 ? Math.round(requiredBalance).toLocaleString() : "—"}
                              </span>
                              <span className="mx-1 opacity-40">/</span>
                              <span className="text-sm">
                                {totalProfitCash !== 0 ? Math.round(Math.abs(totalProfitCash)).toLocaleString() : "0"}
                              </span>
                              <span className="ml-1 text-[8px] opacity-60">EGP</span>
                            </div>
                          );
                        })()}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="font-mono font-bold text-zinc-300">
                          {(() => {
                            const val =
                              (bt as any).benchmark_return_pct ??
                              (bt as any).benchmark_win_rate ??
                              (indexFallbackById as any)?.[bt.id] ??
                              null;
                            if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
                            return `${Number(val).toFixed(2)}%`;
                          })()}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <button onClick={() => handleToggleVisibility(bt.id, bt.is_public)} className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${bt.is_public ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-zinc-900/60 border-white/5 text-zinc-600"}`}>
                          {bt.is_public ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                          <span className="text-[9px] font-black uppercase tracking-widest">{bt.is_public ? "Visible" : "Hidden"}</span>
                        </button>
                      </td>
                      <td className="px-6 py-4 text-right">

                        <button onClick={() => handleDelete(bt.id)} className="p-2 rounded-xl text-zinc-700 hover:text-red-500 hover:bg-red-500/10 transition-all opacity-0 group-hover:opacity-100">
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  </React.Fragment >
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <BacktestAnalysisModal
        isOpen={tradesOpen}
        onClose={() => {
          setTradesOpen(false);
          setViewingBacktest(null);
        }}
        bt={viewingBacktest}
        trades={tradesRows}
        loading={tradesLoading}
      />
      <ConfirmDialog
        isOpen={isConfirmOpen}
        title={confirmConfig.title}
        message={confirmConfig.message}
        isLoading={confirmConfig.isLoading}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={confirmConfig.onConfirm}
        variant="danger"
      />
    </div>
  );
}
