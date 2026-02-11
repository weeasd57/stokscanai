"use client";

import React, { useState, useEffect } from "react";
import { Activity, Calendar, Play, TrendingUp, Target, AlertTriangle, CheckCircle2, FileText, Globe, Trash2, Eye, Wallet, EyeOff, History as HistoryIcon, ChevronDown, LineChart, Database, Users, Cpu, ShieldCheck, Zap, Info } from "lucide-react";
import { getLocalModels, type LocalModelMeta, getBacktests, getBacktestTrades, deleteBacktest, updateBacktestVisibility } from "@/lib/api";
import { useAppState } from "@/contexts/AppStateContext";
import { toast } from "sonner";
import ConfirmDialog from "@/components/ConfirmDialog";
import { TradeTimeline } from "./TradeTimeline";
import Egx30MonthlyChart from "./Egx30MonthlyChart";
import {
  LineChart as RLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, Cell, ReferenceLine
} from 'recharts';

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
                              <span className="text-xs font-mono text-zinc-400">
                                {(() => {
                                  const rawDate = t.features?.entry_date || t.features?.trade_date || t.created_at;
                                  if (!rawDate) return 'N/A';
                                  try {
                                    const d = new Date(rawDate);
                                    if (!Number.isFinite(d.getTime())) return 'N/A';
                                    const yyyy = d.getFullYear();
                                    const mm = String(d.getMonth() + 1).padStart(2, '0');
                                    const dd = String(d.getDate()).padStart(2, '0');
                                    const hh = String(d.getHours()).padStart(2, '0');
                                    const min = String(d.getMinutes()).padStart(2, '0');
                                    return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
                                  } catch { return 'N/A'; }
                                })()}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-left">
                              <span className="text-xs font-mono text-zinc-400">
                                {(() => {
                                  const rawDate = t.features?.exit_date || t.features?.trade_date || t.created_at;
                                  if (!rawDate) return 'N/A';
                                  try {
                                    const d = new Date(rawDate);
                                    if (!Number.isFinite(d.getTime())) return 'N/A';
                                    const yyyy = d.getFullYear();
                                    const mm = String(d.getMonth() + 1).padStart(2, '0');
                                    const dd = String(d.getDate()).padStart(2, '0');
                                    const hh = String(d.getHours()).padStart(2, '0');
                                    const min = String(d.getMinutes()).padStart(2, '0');
                                    return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
                                  } catch { return 'N/A'; }
                                })()}
                              </span>
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
                                <span className="text-zinc-500">In: {(t.entry_price || 0) < 0.1 ? (t.entry_price || 0).toFixed(8) : (t.entry_price || 0).toFixed(2)}</span>
                                <span className="text-zinc-300 font-bold">Out: {(t.exit_price || 0) < 0.1 ? (t.exit_price || 0).toFixed(8) : (t.exit_price || 0).toFixed(2)}</span>
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
                                  const fundScore = (t.features as any)?.fund_score ?? (t.features as any)?.fundamental_score ?? (t as any)?.Fund_Score ?? (t as any)?.fund_score ?? (t as any)?.Validator_Score;
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

const OptimizationResultsDialog = ({ isOpen, onClose, bt, handleVerifyCandidate }: { isOpen: boolean, onClose: () => void, bt: any, handleVerifyCandidate: (t: any, p: any) => void }) => {
  if (!isOpen || !bt) return null;

  let rawTrials: any[] = [];
  try {
    rawTrials = typeof bt.trades_log === 'string' ? JSON.parse(bt.trades_log) : (bt.trades_log || []);
  } catch (e) { rawTrials = []; }

  // Defensive data mapping to handle varying key names from different backend versions
  const trials = rawTrials.map((t, i) => {
    const profit = Number(t.profit_percent ?? t.profit ?? t.profit_pct ?? 0);
    const winRate = Number(t.win_rate ?? t.winrate ?? 0);
    const target = Number(t.target_percent ?? t.target_pct ?? t.target ?? 0);
    const sl = Number(t.stop_loss_percent ?? t.stop_loss_pct ?? t.stop_loss ?? 0);
    const meta = Number(t.wave_confluence ?? t.meta_threshold ?? t.king_threshold ?? 0);
    const validator = Number(t.validator_threshold ?? t.council_threshold ?? 0);
    const tradesCount = Number(t.total_trades ?? t.trades ?? 0);

    return {
      ...t,
      display_profit: profit,
      display_win_rate: winRate,
      display_target: target,
      display_sl: sl,
      display_meta: meta,
      display_validator: validator,
      display_trades: tradesCount,
      index: i + 1
    };
  });

  // Prepare chart data
  const scatterData = trials.map((t) => ({
    id: t.index,
    x: t.display_win_rate,
    y: t.display_profit,
    z: t.display_trades,
    payload: t
  }));

  const top5 = [...trials]
    .sort((a, b) => b.display_profit - a.display_profit)
    .slice(0, 5)
    .map((t) => ({
      name: `Trial ${t.test_number || t.index}`,
      profit: t.display_profit,
      winRate: t.display_win_rate,
      trades: t.display_trades,
      payload: t
    }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const val = data.payload || data;
      return (
        <div className="bg-zinc-950/90 backdrop-blur-md border border-white/10 p-4 rounded-2xl shadow-2xl shrink-0 min-w-[180px] z-50">
          <p className="text-[10px] font-black text-indigo-400 uppercase mb-3 border-b border-white/5 pb-2 tracking-widest">Result Details</p>
          <div className="space-y-2 font-mono">
            <div className="flex justify-between items-center gap-4">
              <span className="text-[10px] text-zinc-500 uppercase font-bold">Meta/C:</span>
              <span className="text-white text-[11px] font-black">{val.display_meta} / {val.display_validator}</span>
            </div>
            <div className="flex justify-between items-center gap-4">
              <span className="text-[10px] text-zinc-500 uppercase font-bold">Target:</span>
              <span className="text-emerald-400 text-[11px] font-black">{val.display_target}%</span>
            </div>
            <div className="flex justify-between items-center gap-4">
              <span className="text-[10px] text-zinc-500 uppercase font-bold">SL:</span>
              <span className="text-red-400 text-[11px] font-black">{val.display_sl}%</span>
            </div>
            <div className="h-px bg-white/5 my-1" />
            <div className="flex justify-between items-center gap-4">
              <span className="text-[10px] text-zinc-500 uppercase font-bold">Win Rate:</span>
              <span className="text-indigo-400 text-[11px] font-black">{val.display_win_rate.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center gap-4">
              <span className="text-[10px] text-zinc-500 uppercase font-bold">Net Profit:</span>
              <span className={`text-[11px] font-black ${val.display_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {val.display_profit.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="absolute inset-0" onClick={onClose} />
      <div className="relative w-full max-w-6xl max-h-[90vh] overflow-hidden bg-zinc-950 border border-white/10 rounded-[2.5rem] shadow-2xl flex flex-col animate-in slide-in-from-bottom-4 zoom-in-95 duration-300">

        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between p-8 bg-zinc-950/90 backdrop-blur-xl border-b border-white/5">
          <div className="flex items-center gap-6">
            <div className="p-4 rounded-2xl bg-indigo-500/20 border border-indigo-500/30 shadow-lg shadow-indigo-500/10">
              <Target className="h-7 w-7 text-indigo-400" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-2xl font-black text-white uppercase tracking-tight">
                  Optimization Results
                </h3>
                <span className="px-3 py-1 rounded-xl bg-white/5 border border-white/5 text-[10px] text-zinc-500 font-black uppercase tracking-widest">
                  {trials.length} Configs Tested
                </span>
              </div>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.3em] mt-1.5 opacity-60">
                {bt?.model_name || "Unknown Optimization"}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-3 rounded-2xl bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-all border border-white/5 hover:scale-105 active:scale-95">
            <Trash2 className="h-6 w-6 rotate-45" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-8 space-y-10 custom-scrollbar">
          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Scatter Plot */}
            <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-8 relative overflow-hidden h-[350px] flex flex-col hover:border-white/10 transition-all">
              <div className="flex items-center justify-between mb-6 shrink-0">
                <div>
                  <h4 className="text-base font-black text-white uppercase tracking-tight">Performance Landscape</h4>
                  <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-1">Win Rate vs Net Profit %</p>
                </div>
                <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
                  <TrendingUp className="h-4 w-4" />
                </div>
              </div>
              <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="Win Rate"
                      unit="%"
                      stroke="#71717a"
                      fontSize={10}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(val) => val.toFixed(0)}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="Profit"
                      unit="%"
                      stroke="#71717a"
                      fontSize={10}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(val) => val.toFixed(0)}
                    />
                    <Tooltip cursor={{ strokeDasharray: '3 3', stroke: '#ffffff20' }} content={<CustomTooltip />} />
                    <Scatter name="Trials" data={scatterData}>
                      {scatterData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.y > 0 ? '#10b981' : '#ef4444'}
                          fillOpacity={0.4}
                          strokeWidth={2}
                          stroke={entry.y > 0 ? '#34d399' : '#f87171'}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Bar Chart */}
            <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-8 relative overflow-hidden h-[350px] flex flex-col hover:border-white/10 transition-all">
              <div className="flex items-center justify-between mb-6 shrink-0">
                <div>
                  <h4 className="text-base font-black text-white uppercase tracking-tight">Top Candidates</h4>
                  <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-1">Highest Profitability Configs</p>
                </div>
                <div className="p-2.5 rounded-xl bg-orange-500/10 text-orange-400 border border-orange-500/20">
                  <Target className="h-4 w-4" />
                </div>
              </div>
              <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={top5} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} vertical={false} stroke="#ffffff05" />
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" width={60} tick={{ fontSize: 9, fill: '#71717a', fontWeight: 700 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: '#ffffff03' }} />
                    <Bar dataKey="profit" radius={[0, 8, 8, 0]} barSize={24}>
                      {top5.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.profit >= 0 ? '#10b981' : '#ef4444'} fillOpacity={0.6} strokeWidth={1} stroke={entry.profit >= 0 ? '#34d399' : '#f87171'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-zinc-950/40 overflow-hidden shadow-inner">
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-[11px] text-left border-collapse">
                <thead className="bg-zinc-950 text-zinc-500 font-black uppercase tracking-widest border-b border-white/5">
                  <tr>
                    <th className="px-8 py-5">Config</th>
                    <th className="px-6 py-5 text-center">Settings (T/S)</th>
                    <th className="px-6 py-5 text-center">Trades (Pre/Post)</th>
                    <th className="px-6 py-5 text-center">Win Rate</th>
                    <th className="px-6 py-5 text-right">Net Profit</th>
                    <th className="px-8 py-5 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-zinc-400 font-mono">
                  {trials.map((t, i) => (
                    <tr key={i} className={`group hover:bg-white/[0.03] transition-all ${t.is_best ? "bg-indigo-500/[0.08]" : ""}`}>
                      <td className="px-8 py-5">
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-2">
                            <span className="text-white font-black text-sm">{t.display_meta}</span>
                            <span className="text-zinc-600">/</span>
                            <span className="text-indigo-400 font-black">{t.display_validator}</span>
                          </div>
                          <span className="text-[9px] text-zinc-500 uppercase font-black tracking-tighter opacity-60">
                            Meta Confidence / Council
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-5 text-center">
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-900/50 border border-white/5">
                          <div className="flex flex-col items-center">
                            <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Target</span>
                            <span className="text-emerald-400 font-black">{t.display_target}%</span>
                          </div>
                          <div className="w-px h-6 bg-white/10" />
                          <div className="flex flex-col items-center">
                            <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Stop</span>
                            <span className="text-red-400 font-black">{t.display_sl}%</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-5 text-center">
                        <div className="flex flex-col items-center">
                          <div className="flex items-center justify-center gap-1.5 mb-1">
                            <span className="text-zinc-500 font-bold">{t.pre_council_trades || "-"}</span>
                            <span className="text-zinc-700">➜</span>
                            <span className="text-white font-black text-sm">{t.display_trades}</span>
                          </div>
                          <div className="h-1 w-12 bg-zinc-900 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-500"
                              style={{ width: `${Math.min(100, ((t.display_trades / (t.pre_council_trades || t.display_trades || 1)) * 100))}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-5 text-center">
                        <div className="flex flex-col items-center">
                          <span className={`text-base font-black ${t.display_win_rate >= 50 ? 'text-indigo-400' : 'text-zinc-400'}`}>
                            {t.display_win_rate.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className={`px-8 py-5 text-right font-black text-base ${t.display_profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        <div className="flex flex-col items-end">
                          <span>{t.display_profit.toFixed(2)}%</span>
                          {t.profit_cash !== undefined && (
                            <span className="text-[10px] opacity-40 font-mono">
                              {Math.round(t.profit_cash).toLocaleString()} units
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-8 py-5 text-center">
                        <div className="flex flex-col items-center gap-3">
                          {t.is_best ? (
                            <div className="flex flex-col items-center">
                              <span className="px-3 py-1 rounded-full bg-indigo-500 text-white text-[9px] font-black uppercase tracking-widest shadow-lg shadow-indigo-500/20 mb-2">
                                Best Model
                              </span>
                            </div>
                          ) : (
                            <span className="text-[10px] text-zinc-600 font-black uppercase tracking-widest">Trial {t.index}</span>
                          )}
                          {handleVerifyCandidate && (
                            <button
                              onClick={() => handleVerifyCandidate(t, bt)}
                              className="px-4 py-2 rounded-xl bg-white/5 hover:bg-indigo-500/20 text-zinc-500 hover:text-indigo-400 text-[10px] font-black uppercase border border-white/10 hover:border-indigo-500/30 transition-all flex items-center gap-2 group/btn"
                            >
                              <Eye className="h-3.5 w-3.5 transition-transform group-hover/btn:scale-110" />
                              Load Param
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  )).reverse()}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};


const OptimizationResultsDialog_OLD = ({ isOpen, onClose, bt, handleVerifyCandidate }: { isOpen: boolean, onClose: () => void, bt: any, handleVerifyCandidate: (t: any, p: any) => void }) => {
  if (!isOpen || !bt) return null;

  let trials: any[] = [];
  try {
    trials = typeof bt.trades_log === 'string' ? JSON.parse(bt.trades_log) : (bt.trades_log || []);
  } catch (e) { trials = []; }

  // Prepare chart data
  const scatterData = trials.map((t, i) => ({
    id: i + 1,
    x: t.win_rate || 0,
    y: t.profit_percent || 0,
    z: t.total_trades || 10,
    payload: t
  }));

  const top5 = [...trials].sort((a, b) => (b.profit_percent || 0) - (a.profit_percent || 0)).slice(0, 5).map((t, i) => ({
    name: `Trial ${t.test_number || i + 1}`,
    profit: t.profit_percent || 0,
    winRate: t.win_rate || 0,
    trades: t.total_trades || 0
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const val = data.payload || data;
      return (
        <div className="bg-zinc-900 border border-white/10 p-3 rounded-lg shadow-xl shrink-0 min-w-[150px] z-50">
          <p className="text-[10px] font-black text-white uppercase mb-2 border-b border-white/5 pb-1">Result Details</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono">
            <span className="text-zinc-500">Wave:</span>
            <span className="text-white text-right">{val.wave_confluence}</span>
            <span className="text-zinc-500">Target:</span>
            <span className="text-white text-right">{val.target_percent}%</span>
            <span className="text-zinc-500">Sl:</span>
            <span className="text-white text-right">{val.stop_loss_percent}%</span>
            <span className="text-zinc-500">Win Rate:</span>
            <span className="text-emerald-400 text-right">{Number(val.win_rate).toFixed(1)}%</span>
            <span className="text-zinc-500">Profit:</span>
            <span className="text-emerald-400 text-right">{Number(val.profit_percent).toFixed(2)}%</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="absolute inset-0" onClick={onClose} />
      <div className="relative w-full max-w-5xl max-h-[85vh] overflow-y-auto bg-zinc-950 border border-white/10 rounded-3xl shadow-2xl flex flex-col animate-in slide-in-from-bottom-4 zoom-in-95 duration-300">

        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between p-6 bg-zinc-950/90 backdrop-blur-xl border-b border-white/5">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-indigo-500/20 border border-indigo-500/30 shadow-lg shadow-indigo-500/10">
              <Target className="h-6 w-6 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-xl font-black text-white uppercase tracking-tight flex items-center gap-2">
                Optimization Results
                <span className="px-2 py-0.5 rounded-lg bg-white/5 border border-white/5 text-[10px] text-zinc-500">
                  {trials.length} Trials
                </span>
              </h3>
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mt-0.5">
                {bt?.model_name || "Unknown Optimization"}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/10 text-zinc-400 hover:text-white transition-all border border-transparent hover:border-white/5">
            <Trash2 className="h-5 w-5 rotate-45" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="rounded-2xl border border-white/5 bg-zinc-900/20 overflow-hidden">
            <table className="w-full text-[10px] text-left">
              <thead className="bg-zinc-950 text-zinc-500 font-black uppercase tracking-widest border-b border-white/5">
                <tr>
                  <th className="px-6 py-4">KING / Council</th>
                  <th className="px-6 py-4 text-center">Trades</th>
                  <th className="px-6 py-4 text-center">Win Rate</th>
                  <th className="px-6 py-4 text-right">Net Profit</th>
                  <th className="px-6 py-4 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-zinc-400 font-mono">
                {trials.map((t: any, i: number) => (
                  <tr key={i} className={`hover:bg-white/[0.02] transition-colors ${t.is_best ? "bg-indigo-500/[0.05]" : ""}`}>
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <span className="text-white font-bold">{t.wave_confluence}</span>
                        <span className="text-[9px] text-zinc-600 uppercase font-black tracking-tighter">
                          VAL: {t.validator_threshold ?? "None"}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">{t.trades}</td>
                    <td className="px-6 py-4 text-center">{t.win_rate}%</td>
                    <td className={`px-6 py-4 text-right font-bold ${(t.profit || 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {(t.profit || 0).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="flex flex-col items-center gap-2">
                        {t.is_best ? (
                          <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 text-[8px] font-black uppercase tracking-tighter">Best Candidate</span>
                        ) : (
                          <span className="text-zinc-600">Trial {i + 1}</span>
                        )}
                        {handleVerifyCandidate && (
                          <button
                            onClick={() => handleVerifyCandidate(t, bt)}
                            className="px-2 py-1 rounded bg-white/5 hover:bg-indigo-500/20 text-zinc-400 hover:text-indigo-400 text-[9px] font-bold uppercase border border-white/5 hover:border-indigo-500/20 transition-all flex items-center gap-1.5"
                            title="Load parameters into Manual Backtest"
                          >
                            <span className="text-emerald-400 text-[9px] font-black uppercase tracking-widest flex items-center gap-1">
                              Verify <Eye className="h-3 w-3" />
                            </span>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )).reverse()}
              </tbody>
            </table>
          </div>
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
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>("1D");
  const [selectedCouncilModel, setSelectedCouncilModel] = useState<string | null>(null);
  const [startDate, setStartDate] = useState(new Date().toISOString().slice(0, 10));
  const [endDate, setEndDate] = useState(new Date().toISOString().slice(0, 10));
  const [metaThreshold, setMetaThreshold] = useState<number>(0.6);
  const [targetPct, setTargetPct] = useState<number>(15);
  const [stopLossPct, setStopLossPct] = useState<number>(5);
  const [councilThreshold, setCouncilThreshold] = useState<number>(0.1);
  const [startingCapital, setStartingCapital] = useState<number>(100000);

  // Automation State
  const [activeMainTab, setActiveMainTab] = useState<"manual" | "automation">("manual");
  const [automationStep, setAutomationStep] = useState<number>(0.05);
  const [optTargets, setOptTargets] = useState<string>("5, 10, 15");
  const [optStopLosses, setOptStopLosses] = useState<string>("3, 5, 7");
  const [optKingValues, setOptKingValues] = useState<string>("0.7, 0.8, 0.9");
  const [optWaveValues, setOptWaveValues] = useState<string>("0.0, 0.55, 0.7");
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [currentOptId, setCurrentOptId] = useState<string | null>(null);
  const [selectedHistoryOptId, setSelectedHistoryOptId] = useState<string | null>(null);

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

  // Auto-set timeframe based on exchange
  useEffect(() => {
    if (selectedExchange === "CRYPTO") {
      setSelectedTimeframe("1H");
    } else {
      setSelectedTimeframe("1D");
    }
  }, [selectedExchange]);
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
    if ((running && currentBacktestId) || (isOptimizing && currentOptId)) {
      const targetId = activeMainTab === "automation" ? currentOptId : currentBacktestId;
      interval = setInterval(async () => {
        try {
          const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
          const res = await fetch(`${baseUrl}/backtests`); // Refresh full history
          if (!res.ok) return;
          const data = await res.json();
          setHistory(data);

          // Find current one
          const current = data.find((b: any) => b.id === targetId);
          if (current) {
            setStatusMsg(current.status_msg);
            if (current.status === "completed") {
              if (activeMainTab === "automation") {
                setIsOptimizing(false);
                // setCurrentOptId(null);
                toast.success("Optimization Completed", {
                  description: `Found best threshold: ${current.meta_threshold}`
                });
              } else {
                setRunning(false);
                setCurrentBacktestId(null);
              }
              setStatusMsg(null);
              clearInterval(interval);
            } else if (current.status === "failed") {
              if (activeMainTab === "automation") {
                setIsOptimizing(false);
                setCurrentOptId(null);
              } else {
                setRunning(false);
                setCurrentBacktestId(null);
              }
              setStatusMsg(null);
              setError(current.status_msg || "Task failed");
              toast.error("Task Failed");
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [running, currentBacktestId, isOptimizing, currentOptId, activeMainTab]);

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


  const handleVerifyCandidate = (trial: any, parentOpt: any) => {
    // 1. Set Manual Parameters
    if (trial.wave_confluence !== undefined) setMetaThreshold(trial.wave_confluence);
    if (trial.validator_threshold !== undefined) setCouncilThreshold(trial.validator_threshold);
    if (trial.target_percent !== undefined) setTargetPct(trial.target_percent);
    if (trial.stop_loss_percent !== undefined) setStopLossPct(trial.stop_loss_percent);

    // 2. Set Models (Best effort extraction from OPT: Name)
    if (parentOpt) {
      if (parentOpt.model_name && parentOpt.model_name.startsWith("OPT: ")) {
        const realModel = parentOpt.model_name.replace("OPT: ", "");
        setSelectedModel(realModel);
      }
      if (parentOpt.council_model) {
        setSelectedCouncilModel(parentOpt.council_model);
      }
    }

    // 3. Switch Tab & Notify
    setActiveMainTab("manual");
    toast.success("Candidate Loaded", {
      description: "Parameters applied to Manual Backtest. Click 'Run Backtest' to verify."
    });
  };

  const handleRun = async () => {
    if (!selectedModel) return;

    setRunning(true);
    setError(null); // Clear previous errors
    setResult(null);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
      const payload = {
        exchange: selectedExchange,
        model: selectedModel,
        start_date: startDate,
        end_date: endDate,
        council_model: selectedCouncilModel,
        council_threshold: normalizeThreshold01(councilThreshold),
        meta_threshold: normalizeThreshold01(metaThreshold),
        target_pct: targetPct / 100,
        stop_loss_pct: stopLossPct / 100,
        capital: startingCapital,
        timeframe: selectedTimeframe,
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

  const handleOptimize = async () => {
    if (!selectedModel) return;
    setIsOptimizing(true);
    setError(null);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
      const res = await fetch(`${baseUrl}/backtest/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exchange: selectedExchange,
          model: selectedModel,
          start_date: startDate,
          end_date: endDate,
          wave_values: optKingValues.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
          validator_values: optWaveValues.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
          target_values: optTargets.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
          stoploss_values: optStopLosses.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
          capital: startingCapital,
          timeframe: selectedTimeframe
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Optimization failed");

      setCurrentOptId(data.id);
      setStatusMsg("Optimizing...");
      toast.info("Optimization Started", { description: data.message });
    } catch (err: any) {
      setError(err.message);
      toast.error("Optimization Failed");
      setIsOptimizing(false);
    }
  };

  const handleOpenTrades = async (bt: any) => {
    // If it's an optimization run (OPT: prefix), switch to automation tab
    if (bt.model_name?.startsWith("OPT:")) {
      setSelectedHistoryOptId(bt.id);
      setActiveMainTab("automation");
      toast.info("Viewing Optimization Results", { description: "Switched to Automation tab." });
      return;
    }

    // For Manual Backtests: Restore to Dashboard
    const toastId = toast.loading("Restoring Analysis...");
    try {
      const trades = await getBacktestTrades(bt.id);
      const safeTrades = Array.isArray(trades) ? trades : [];

      // Calculate missing metrics if needed
      let totalReturn = 0;
      safeTrades.forEach((t: any) => {
        totalReturn += (t.pnl_pct || 0);
      });
      const avgReturn = safeTrades.length > 0 ? totalReturn / safeTrades.length : 0;

      const restoredResult: BacktestResult = {
        totalTrades: bt.total_trades || safeTrades.length,
        winRate: bt.win_rate || 0,
        netProfit: bt.profit || 0, // Assuming profit % is stored here
        avgReturnPerTrade: avgReturn,
        trades: safeTrades
      };

      setResult(restoredResult);
      setActiveMainTab("manual");

      // Also restore inputs for context if possible (optional but nice)
      if (bt.model_name) setSelectedModel(bt.model_name.replace(" (Manual)", ""));
      if (bt.exchange) setSelectedExchange(bt.exchange);

      toast.dismiss(toastId);
      toast.success("Analysis Restored", { description: "Viewing historical results in Dashboard." });

    } catch (err: any) {
      toast.dismiss(toastId);
      toast.error("Failed to load history", { description: err.message });
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
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 shadow-lg shadow-emerald-500/5">
            <Activity className="h-6 w-6 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-tight uppercase">Simulation <span className="text-emerald-500">Radar</span></h2>
            <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]">Deep Learning historical engine</p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex p-1 bg-zinc-950/60 rounded-2xl border border-white/5 backdrop-blur-md">
          <button
            onClick={() => setActiveMainTab("manual")}
            className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeMainTab === "manual" ? "bg-white/10 text-white shadow-lg shadow-white/5" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            Manual Backtest
          </button>
          <button
            onClick={() => setActiveMainTab("automation")}
            className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeMainTab === "automation" ? "bg-indigo-500/20 text-indigo-400 shadow-lg shadow-indigo-500/5" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            Automation Optimizer
          </button>
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

            {/* Timeframe Selector Row */}
            <div className="space-y-3">
              <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                <Calendar className="h-3 w-3" /> Timeframe
              </label>
              <select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all cursor-pointer hover:border-emerald-500/30"
              >
                <option value="1H" className="bg-zinc-900 text-white">1H (Hourly)</option>
                <option value="4H" className="bg-zinc-900 text-white">4H (4-Hour)</option>
                <option value="1D" className="bg-zinc-900 text-white">1D (Daily)</option>
                <option value="1W" className="bg-zinc-900 text-white">1W (Weekly)</option>
              </select>
            </div>
          </div>

          {/* Section 2: Intelligence Filters & Temporal Range */}
          {activeMainTab === "manual" ? (
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

                  {/* Target & Stop Loss Section */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                        <span className="flex items-center gap-2"><Target className="h-3 w-3 text-emerald-400" /> Target %</span>
                      </label>
                      <div className="relative">
                        <input
                          type="number"
                          value={targetPct}
                          onChange={(e) => setTargetPct(Number(e.target.value))}
                          className="w-full h-14 pl-5 pr-12 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-lg focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all"
                          placeholder="15"
                        />
                        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500 font-mono text-xs">%</div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                        <span className="flex items-center gap-2"><AlertTriangle className="h-3 w-3 text-red-400" /> Stop Loss %</span>
                      </label>
                      <div className="relative">
                        <input
                          type="number"
                          value={stopLossPct}
                          onChange={(e) => setStopLossPct(Number(e.target.value))}
                          className="w-full h-14 pl-5 pr-12 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-lg focus:ring-2 focus:ring-red-500/40 outline-none transition-all"
                          placeholder="5"
                        />
                        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500 font-mono text-xs">%</div>
                      </div>
                    </div>
                  </div>

                  {/* Council Threshold & Capital */}
                  <div className="grid grid-cols-2 gap-4">
                    {activeMainTab === "manual" ? (
                      selectedCouncilModel && (
                        <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-500">
                          <label className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest flex items-center justify-between">
                            <span className="flex items-center gap-2"><ShieldCheck className="h-3 w-3" /> Council consensus</span>
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
                      )
                    ) : (
                      <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-500">
                        <label className="text-[9px] font-bold text-amber-400 uppercase tracking-widest flex items-center justify-between">
                          <span className="flex items-center gap-2"><Zap className="h-3 w-3" /> Automation Step</span>
                        </label>
                        <div className="relative">
                          <input
                            type="number"
                            min={0.01}
                            max={0.2}
                            step={0.01}
                            value={automationStep}
                            onChange={(e) => setAutomationStep(Number(e.target.value))}
                            className="w-full h-14 pl-5 pr-12 rounded-2xl border border-amber-500/20 bg-amber-500/5 text-white font-mono text-lg focus:ring-2 focus:ring-amber-500/40 outline-none transition-all"
                          />
                          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-amber-400/60 font-mono text-xs">Step</div>
                        </div>
                      </div>
                    )}
                    <div className="space-y-3">
                      <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                        <span className="flex items-center gap-2"><Wallet className="h-3 w-3 text-indigo-400" /> Starting Capital</span>
                      </label>
                      <div className="relative">
                        <input
                          type="number"
                          value={startingCapital}
                          onChange={(e) => setStartingCapital(Number(e.target.value))}
                          className="w-full h-14 pl-5 pr-12 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-lg focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                          placeholder="100000"
                        />
                        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500 font-mono text-xs">{selectedExchange === "CRYPTO" ? "USD" : "EGP"}</div>
                      </div>
                    </div>
                  </div>
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
          ) : (
            <div className="pt-6 border-t border-white/5 grid gap-12 md:grid-cols-2 animate-in fade-in slide-in-from-top-4 duration-500">
              {/* Automation Controls */}
              <div className="space-y-6">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-1 bg-indigo-500 rounded-full" />
                  <h3 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Optimization Parameters</h3>
                </div>
                <div className="grid gap-6">
                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest flex items-center justify-between">
                      <span className="flex items-center gap-2"><Target className="h-3 w-3" /> Target Pct List</span>
                      <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">e.g. 5, 10, 15</span>
                    </label>
                    <input
                      type="text"
                      value={optTargets}
                      onChange={(e) => setOptTargets(e.target.value)}
                      placeholder="5, 10, 15"
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-red-400 uppercase tracking-widest flex items-center justify-between">
                      <span className="flex items-center gap-2"><AlertTriangle className="h-3 w-3" /> Stop Loss List</span>
                      <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">e.g. 3, 5, 7</span>
                    </label>
                    <input
                      type="text"
                      value={optStopLosses}
                      onChange={(e) => setOptStopLosses(e.target.value)}
                      placeholder="3, 5, 7"
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-amber-400 uppercase tracking-widest flex items-center justify-between">
                      <span className="flex items-center gap-2"><Cpu className="h-3 w-3" /> KING (Primary) Thresholds</span>
                      <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">e.g. 0.7, 0.8, 0.9</span>
                    </label>
                    <input
                      type="text"
                      value={optKingValues}
                      onChange={(e) => setOptKingValues(e.target.value)}
                      placeholder="0.7, 0.8, 0.9"
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-amber-500/40 outline-none transition-all"
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-emerald-400 uppercase tracking-widest flex items-center justify-between">
                      <span className="flex items-center gap-2"><ShieldCheck className="h-3 w-3" /> Council / Validator Thresholds</span>
                      <span className="text-[8px] text-zinc-500 normal-case font-medium lowercase">e.g. 0.0, 0.55, 0.7</span>
                    </label>
                    <input
                      type="text"
                      value={optWaveValues}
                      onChange={(e) => setOptWaveValues(e.target.value)}
                      placeholder="0.0, 0.55, 0.7"
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-emerald-500/40 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                      <Calendar className="h-3 w-3" /> Global Start Date
                    </label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-3">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                      <Calendar className="h-3 w-3" /> Global End Date
                    </label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full h-14 px-5 rounded-2xl border border-white/5 bg-zinc-900/80 text-white font-mono text-sm focus:ring-2 focus:ring-indigo-500/40 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              {/* Automation Info */}
              <div className="bg-indigo-500/5 border border-indigo-500/10 rounded-[2rem] p-8 flex flex-col justify-center gap-4 text-center">
                <div className="w-16 h-16 bg-indigo-500/20 rounded-full flex items-center justify-center mx-auto border border-indigo-500/30">
                  <Cpu className="w-8 h-8 text-indigo-400 animate-pulse" />
                </div>
                <div>
                  <h4 className="text-sm font-black text-white mb-1 uppercase tracking-tighter">Brute-Force Optimizer</h4>
                  <p className="text-[10px] text-zinc-500 leading-relaxed font-bold">
                    This engine will run multiple backtests across a range of thresholds (0.30 to 0.90) to find the most profitable configuration for your specific model and exchange.
                  </p>
                </div>
              </div>
            </div>
          )}


          {/* Action Button and Status */}
          <div className="pt-8 border-t border-white/5 space-y-4">
            <div className="flex gap-4">
              <button
                onClick={activeMainTab === "automation" ? handleOptimize : handleRun}
                disabled={!selectedModel || running || isOptimizing}
                className={`relative flex-1 group/btn h-16 rounded-[1.25rem] transition-all duration-500 overflow-hidden shadow-2xl ${!selectedModel || running || isOptimizing
                  ? "bg-zinc-800/50 cursor-not-allowed grayscale"
                  : activeMainTab === "automation"
                    ? "bg-gradient-to-r from-indigo-600 to-indigo-500 hover:shadow-indigo-500/20 active:scale-[0.98]"
                    : "bg-gradient-to-r from-emerald-600 to-emerald-500 hover:shadow-emerald-500/20 active:scale-[0.98]"
                  }`}
              >
                {/* Button inner content */}
                <div className="flex items-center justify-center gap-3">
                  {(running || isOptimizing) ? (
                    <>
                      <div className="relative w-5 h-5">
                        <div className="absolute inset-0 rounded-full border-2 border-white/10" />
                        <div className="absolute inset-0 rounded-full border-2 border-t-white animate-spin" />
                      </div>
                      <span className="text-sm font-black text-white uppercase tracking-[0.2em]">
                        {activeMainTab === "automation" ? "Optimizing Network..." : "Executing Simulation..."}
                      </span>
                    </>
                  ) : (
                    <>
                      {activeMainTab === "automation" ? (
                        <Cpu className="h-5 w-5 text-white animate-pulse" />
                      ) : (
                        <Zap className="h-5 w-5 text-white fill-white animate-pulse" />
                      )}
                      <span className="text-sm font-black text-white uppercase tracking-[0.2em]">
                        {!selectedModel ? "Select Model to Initiate" : activeMainTab === "automation" ? "Start Golden Search" : "Initiate Full Backtest"}
                      </span>
                    </>
                  )}
                </div>
              </button>

              {result && activeMainTab === "manual" && (
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
              <div className="flex items-center gap-3 p-4 rounded-2xl bg-indigo-500/5 border border-indigo-500/10 text-indigo-400 text-[10px] font-mono animate-in fade-in slide-in-from-left-2 duration-500">
                <div className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                <span className="uppercase tracking-[0.1em] font-black opacity-60">Engine Status:</span>
                <span className="font-bold">{statusMsg}</span>
              </div>
            )}

            {error && (
              <div className="p-4 rounded-2xl bg-red-500/5 border border-red-500/10 text-red-500 text-[10px] font-mono animate-in fade-in slide-in-from-left-2 duration-500">
                <span className="uppercase tracking-[0.1em] font-black opacity-60 mr-2">Error:</span>
                {error}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Section - Active Optimization Only */}
      {activeMainTab === "automation" && currentOptId && (
        <div className="relative overflow-hidden rounded-3xl border border-indigo-500/20 bg-zinc-900/60 backdrop-blur-3xl p-8 shadow-2xl animate-in fade-in slide-in-from-bottom-8 duration-700">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-2xl bg-indigo-500/20 border border-indigo-500/30">
                <Target className="h-6 w-6 text-indigo-400" />
              </div>
              <div>
                <h3 className="text-xl font-black text-white uppercase tracking-tight">
                  Optimization Trials
                </h3>
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                  Brute-force parameter discovery
                </p>
              </div>
            </div>
            {selectedHistoryOptId && (
              <button
                onClick={() => setSelectedHistoryOptId(null)}
                className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-zinc-400 hover:text-white hover:bg-white/10 transition-all"
              >
                Close Archive
              </button>
            )}
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-zinc-950/40 overflow-hidden shadow-inner">
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-[11px] text-left border-collapse">
                <thead className="bg-zinc-950 text-zinc-500 font-black uppercase tracking-widest border-b border-white/5">
                  <tr>
                    <th className="px-8 py-5">Config</th>
                    <th className="px-6 py-5 text-center">Settings (T/S)</th>
                    <th className="px-6 py-5 text-center">Trades (Pre/Post)</th>
                    <th className="px-6 py-5 text-center">Win Rate</th>
                    <th className="px-6 py-5 text-right">Net Profit</th>
                    <th className="px-8 py-5 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-zinc-400 font-mono">
                  {(() => {
                    const targetId = selectedHistoryOptId || currentOptId;
                    const latestOpt = history.find(h => h.id === targetId || (currentOptId && h.model_name?.startsWith("OPT:")));
                    if (!latestOpt) return <tr><td colSpan={6} className="px-8 py-10 text-center text-zinc-600 font-black uppercase tracking-widest opacity-20 italic">No active optimization data stream</td></tr>;

                    let rawTrials = [];
                    try {
                      rawTrials = typeof latestOpt.trades_log === 'string' ? JSON.parse(latestOpt.trades_log) : (latestOpt.trades_log || []);
                    } catch (e) { rawTrials = []; }

                    if (!Array.isArray(rawTrials) || rawTrials.length === 0) {
                      return <tr><td colSpan={6} className="px-8 py-10 text-center text-zinc-500 font-black uppercase tracking-widest animate-pulse">Awaiting first trial result from compute engine...</td></tr>;
                    }

                    return rawTrials.map((t: any, i: number) => {
                      const profit = Number(t.profit_percent ?? t.profit ?? t.profit_pct ?? 0);
                      const winRate = Number(t.win_rate ?? t.winrate ?? 0);
                      const target = Number(t.target_percent ?? t.target_pct ?? t.target ?? 0);
                      const sl = Number(t.stop_loss_percent ?? t.stop_loss_pct ?? t.stop_loss ?? 0);
                      const meta = Number(t.wave_confluence ?? t.meta_threshold ?? t.king_threshold ?? 0);
                      const validator = Number(t.validator_threshold ?? t.council_threshold ?? 0);
                      const tradesCount = Number(t.total_trades ?? t.trades ?? 0);
                      const index = i + 1;

                      return (
                        <tr key={i} className={`group hover:bg-white/[0.03] transition-all ${t.is_best ? "bg-indigo-500/[0.08]" : ""}`}>
                          <td className="px-8 py-5">
                            <div className="flex flex-col gap-1">
                              <div className="flex items-center gap-2">
                                <span className="text-white font-black text-sm">{meta}</span>
                                <span className="text-zinc-600">/</span>
                                <span className="text-indigo-400 font-black">{validator}</span>
                              </div>
                              <span className="text-[9px] text-zinc-500 uppercase font-black tracking-tighter opacity-60">
                                Meta Confidence / Council
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-center">
                            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-900/50 border border-white/5">
                              <div className="flex flex-col items-center">
                                <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Target</span>
                                <span className="text-emerald-400 font-black">{target}%</span>
                              </div>
                              <div className="w-px h-6 bg-white/10" />
                              <div className="flex flex-col items-center">
                                <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Stop</span>
                                <span className="text-red-400 font-black">{sl}%</span>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-center">
                            <div className="flex flex-col items-center">
                              <div className="flex items-center justify-center gap-1.5 mb-1">
                                <span className="text-zinc-500 font-bold">{t.pre_council_trades || "-"}</span>
                                <span className="text-zinc-700">➜</span>
                                <span className="text-white font-black text-sm">{tradesCount}</span>
                              </div>
                              <div className="h-1 w-12 bg-zinc-900 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-indigo-500"
                                  style={{ width: `${Math.min(100, ((tradesCount / (t.pre_council_trades || tradesCount || 1)) * 100))}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-center">
                            <div className="flex flex-col items-center">
                              <span className={`text-base font-black ${winRate >= 50 ? 'text-indigo-400' : 'text-zinc-400'}`}>
                                {winRate.toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className={`px-8 py-5 text-right font-black text-base ${profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            <div className="flex flex-col items-end">
                              <span>{profit.toFixed(2)}%</span>
                              {t.profit_cash !== undefined && (
                                <span className="text-[10px] opacity-40 font-mono">
                                  {Math.round(t.profit_cash).toLocaleString()} units
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="px-8 py-5 text-center">
                            <div className="flex flex-col items-center gap-3">
                              {t.is_best ? (
                                <div className="flex flex-col items-center">
                                  <span className="px-3 py-1 rounded-full bg-indigo-500 text-white text-[9px] font-black uppercase tracking-widest shadow-lg shadow-indigo-500/20 mb-2">
                                    Best Model
                                  </span>
                                </div>
                              ) : (
                                <span className="text-[10px] text-zinc-600 font-black uppercase tracking-widest">Trial {index}</span>
                              )}
                              <button
                                onClick={() => handleVerifyCandidate(t, latestOpt)}
                                className="px-4 py-2 rounded-xl bg-white/5 hover:bg-indigo-500/20 text-zinc-500 hover:text-indigo-400 text-[10px] font-black uppercase border border-white/10 hover:border-indigo-500/30 transition-all flex items-center gap-2 group/btn"
                                title="Load parameters into Manual Backtest"
                              >
                                <Eye className="h-3.5 w-3.5 transition-transform group-hover/btn:scale-110" />
                                Load Param
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    }).reverse();
                  })()}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {result && activeMainTab === "manual" && (
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
                history
                  .filter(bt => {
                    const isOpt = bt.model_name?.startsWith("OPT:");
                    return activeMainTab === "automation" ? isOpt : !isOpt;
                  })
                  .map((bt) => {
                    const isOpt = bt.model_name?.startsWith("OPT:");
                    let trials: any[] = [];
                    let bestTrial: any = null;

                    if (isOpt) {
                      try {
                        trials = typeof bt.trades_log === 'string' ? JSON.parse(bt.trades_log) : (bt.trades_log || []);
                        bestTrial = Array.isArray(trials) ? trials.find(t => t.is_best) || trials[trials.length - 1] : null;
                      } catch (e) { trials = []; }
                    }

                    return (
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
                                <span className="text-white font-black">{bt.model_name?.replace(".pkl", "").replace("OPT: ", "OPTIMIZER: ") || "N/A"}</span>
                                {bt.status === "processing" && <Activity className="h-3 w-3 text-emerald-400 animate-spin" />}
                                {isOpt && <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 text-[8px] font-black uppercase">Auto</span>}
                              </div>
                              <span className="text-[8px] font-bold text-zinc-600 uppercase tracking-widest">{bt.exchange} • {new Date(bt.created_at).toLocaleDateString()}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-center">
                            <div className="flex flex-col items-center gap-1.5">
                              {isOpt && bestTrial ? (
                                <div className="flex flex-col items-center">
                                  <span className="px-2 py-0.5 rounded bg-indigo-500 text-white font-mono text-[8px] font-black uppercase tracking-tighter mb-1 shadow-lg shadow-indigo-500/20">
                                    BEST: {bestTrial.wave_confluence ?? bestTrial.meta_threshold} / {bestTrial.validator_threshold ?? bestTrial.council_threshold}
                                  </span>
                                  <div className="flex items-center gap-2">
                                    <span className="text-emerald-500 text-[8px] font-black">T: {Math.round((bestTrial.target_percent ?? bestTrial.target_pct ?? 0) * 100)}%</span>
                                    <span className="text-zinc-700">|</span>
                                    <span className="text-red-500 text-[8px] font-black">S: {Math.round((bestTrial.stop_loss_percent ?? bestTrial.stop_loss_pct ?? 0) * 100)}%</span>
                                  </div>
                                </div>
                              ) : bt.council_model ? (
                                <div className="flex flex-col items-center">
                                  <span className="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-mono text-[8px] uppercase tracking-tighter mb-1">
                                    {bt.council_model.replace(".pkl", "")}
                                    <span className="ml-1 opacity-60">@{bt.council_threshold ?? 0.1}</span>
                                  </span>
                                  <div className="flex items-center gap-2">
                                    <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/5 text-zinc-500 font-mono text-[8px] uppercase">
                                      Meta: {bt.meta_threshold ?? 0.4}
                                    </span>
                                    {(bt.target_pct !== undefined || bt.stop_loss_pct !== undefined) && (
                                      <div className="flex items-center gap-1 text-[8px] font-mono font-bold">
                                        <span className="text-emerald-500/80">T:{Math.round((bt.target_pct || 0) * 100)}%</span>
                                        <span className="text-zinc-700">|</span>
                                        <span className="text-red-500/80">S:{Math.round((bt.stop_loss_pct || 0) * 100)}%</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              ) : (
                                <div className="flex flex-col items-center">
                                  <span className="text-zinc-600 uppercase text-[8px] font-black tracking-widest mb-1">INDIVIDUAL</span>
                                  <div className="flex items-center gap-2">
                                    <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/5 text-zinc-500 font-mono text-[8px] uppercase">
                                      Meta: {bt.meta_threshold ?? 0.4}
                                    </span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </td>
                          <td className="px-6 py-4 text-center">
                            <div className="flex flex-col items-center">
                              <span className="font-mono text-[10px] text-zinc-400">{bt.start_date ? new Date(bt.start_date).toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }) : '—'}</span>
                              <div className="w-px h-2 bg-zinc-800 my-1" />
                              <span className="font-mono text-[10px] text-zinc-400">{bt.end_date ? new Date(bt.end_date).toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }) : '—'}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-center">
                            {isOpt ? (
                              <button
                                onClick={() => handleOpenTrades(bt)}
                                className="flex flex-col items-center group/trades hover:bg-indigo-500/10 p-2 rounded-xl transition-all border border-transparent hover:border-indigo-500/20"
                                title="View Optimization Results"
                              >
                                <span className="text-indigo-400 text-[9px] font-black uppercase tracking-widest flex items-center gap-1.5 mb-1">
                                  <Target className="h-3 w-3" /> {Array.isArray(trials) ? trials.length : 0} Trials
                                </span>
                                <span className="text-[8px] text-zinc-600 font-bold uppercase tracking-tighter group-hover/trades:text-indigo-300">Open Report</span>
                              </button>
                            ) : (
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
                            )}
                          </td>
                          <td className="px-6 py-4 text-center">
                            {isOpt && bestTrial ? (
                              <div className="flex flex-col items-center gap-1">
                                <span className="font-mono font-black text-white">{(bestTrial.win_rate ?? 0).toFixed(1)}%</span>
                                <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Best Win Rate</span>
                              </div>
                            ) : (
                              <div className="flex flex-col items-center gap-1">
                                <span className="font-mono text-[10px] text-zinc-500">{(bt.pre_council_win_rate || bt.win_rate || 0).toFixed(1)}%</span>
                                <div className="h-px w-4 bg-white/5" />
                                <span className="font-mono font-bold text-emerald-400">{(bt.post_council_win_rate || bt.win_rate || 0).toFixed(1)}%</span>
                              </div>
                            )}
                          </td>
                          <td className="px-6 py-4 text-right">
                            {isOpt && bestTrial ? (
                              <div className="flex flex-col items-end gap-1">
                                <span className={`font-mono font-black ${(bestTrial.profit_percent ?? bestTrial.profit ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                  {(Number(bestTrial.profit_percent ?? bestTrial.profit ?? 0)).toFixed(1)}%
                                </span>
                                <span className="text-[8px] text-zinc-600 uppercase font-bold tracking-tighter">Best Profit</span>
                              </div>
                            ) : (
                              (() => {
                                const tradesLog = (bt as any).trades_log as any[] | undefined;
                                let preProfit = 0;
                                let postProfit = 0;
                                const cap = bt.capital || 100000;

                                if (Array.isArray(tradesLog)) {
                                  for (const t of tradesLog) {
                                    const pc = Number((t as any).Profit_Cash ?? (t as any).profit_cash ?? (t as any).features?.profit_cash ?? 0) || 0;
                                    const isAccepted = (t as any).Status === "Accepted" || (t as any).status === "Accepted";

                                    preProfit += pc;
                                    if (isAccepted) postProfit += pc;
                                  }
                                }

                                const prePct = (preProfit / cap) * 100;
                                const postPct = (postProfit / cap) * 100;

                                return (
                                  <div className="flex flex-col items-end gap-1">
                                    <span className={`font-mono text-[9px] ${prePct >= 0 ? "text-emerald-500/60" : "text-red-500/60"}`}>
                                      {prePct.toFixed(1)}%
                                    </span>
                                    <span className={`font-mono font-bold ${postPct >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                                      {postPct.toFixed(1)}%
                                    </span>
                                  </div>
                                );
                              })()
                            )}
                          </td>
                          <td className="px-6 py-4 text-right whitespace-nowrap">
                            {isOpt && bestTrial ? (
                              <div className="flex flex-col items-end gap-1">
                                <div className="flex items-center gap-1.5 font-mono text-[9px] text-zinc-500">
                                  <span className="text-emerald-400">Profit: {Math.round(bestTrial.profit_cash ?? 0).toLocaleString()}</span>
                                </div>
                                <div className="bg-white/5 text-zinc-300 border border-white/5 px-2 py-0.5 rounded-md font-mono font-bold text-[10px]">
                                  <span className="text-[8px] opacity-60 uppercase tracking-tighter">Trials:</span>
                                  <span className="ml-1">{Array.isArray(trials) ? trials.length : 0}</span>
                                </div>
                              </div>
                            ) : (
                              (() => {
                                const tradesLog = (bt as any).trades_log as any[] | undefined;
                                let totalProfitCash = 0;
                                let requiredBalance = 0;
                                const startingCap = bt.capital || 100000;

                                if (Array.isArray(tradesLog) && tradesLog.length > 0) {
                                  const daily: Record<string, number> = {};

                                  for (const t of tradesLog) {
                                    const entryRaw = (t as any).Entry_Date || (t as any).entry_date || (t as any).features?.entry_date || (t as any).features?.trade_date;
                                    const exitRaw = (t as any).Exit_Date || (t as any).exit_date || (t as any).features?.exit_date || (t as any).features?.trade_date;

                                    const entry = new Date(entryRaw);
                                    const exit = new Date(exitRaw);
                                    if (!Number.isFinite(entry.getTime()) || !Number.isFinite(exit.getTime())) continue;

                                    const pc = Number((t as any).Profit_Cash ?? (t as any).profit_cash ?? (t as any).features?.profit_cash ?? 0) || 0;
                                    totalProfitCash += pc;

                                    const positionCash = Number((t as any).Position_Cash ?? (t as any).position_cash ?? (t as any).features?.position_cash ?? 0) || 0;
                                    if (positionCash <= 0) continue;

                                    const start = entry < exit ? entry : exit;
                                    const end = exit > entry ? exit : entry;
                                    const cursor = new Date(start);
                                    let steps = 0;
                                    while (cursor <= end && steps < 400) {
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
                                const isWarning = requiredBalance > startingCap;

                                return (
                                  <div className="flex flex-col items-end gap-1">
                                    <div className="flex items-center gap-1.5 font-mono text-[9px] text-zinc-500">
                                      <span>Cap: {startingCap.toLocaleString()}</span>
                                      <span className="opacity-20">/</span>
                                      <span className={colorClass}>Profit: {Math.round(totalProfitCash).toLocaleString()}</span>
                                    </div>
                                    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md font-mono font-bold text-[10px] ${isWarning ? 'bg-red-500/10 text-red-500 border border-red-500/20' : 'bg-white/5 text-zinc-300 border border-white/5'}`}>
                                      <span className="text-[8px] opacity-60 uppercase tracking-tighter">Max Exp:</span>
                                      <span>{requiredBalance > 0 ? Math.round(requiredBalance).toLocaleString() : "—"}</span>
                                      {isWarning && <AlertTriangle className="h-2.5 w-2.5 ml-0.5 animate-pulse" />}
                                    </div>
                                  </div>
                                );
                              })()
                            )}
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
                      </React.Fragment>
                    );
                  })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <OptimizationResultsDialog
        isOpen={!!selectedHistoryOptId}
        onClose={() => setSelectedHistoryOptId(null)}
        bt={history.find(h => h.id === selectedHistoryOptId)}
        handleVerifyCandidate={handleVerifyCandidate}
      />

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
