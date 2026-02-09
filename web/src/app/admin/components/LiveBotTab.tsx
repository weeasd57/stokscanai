"use client";

import { useState, useEffect } from "react";
import { Play, Square, Activity, Settings, Terminal, RefreshCw, Save, Coins, ShieldCheck, ShieldAlert, Plus, X, Search, Check, BarChart3, PieChart, ArrowUpRight, ArrowDownRight, Clock } from "lucide-react";
import { useRef } from "react";
import { toast } from "sonner";
import { getAlpacaAccount, getAlpacaPositions, type AlpacaAccountInfo, type AlpacaPositionInfo } from "@/lib/api";
// Radix UI Imports
import * as Dialog from "@radix-ui/react-dialog";
import * as Switch from "@radix-ui/react-switch";

// Types
interface BotConfig {
    alpaca_key_id: string;
    alpaca_secret_key: string;
    alpaca_base_url: string;
    data_source: string;
    coins: string[];
    king_threshold: number;
    council_threshold: number;
    max_notional_usd: number;
    pct_cash_per_trade: number;
    bars_limit: number;
    poll_seconds: number;
    timeframe: string;
    use_council: boolean;
    enable_sells: boolean;
    target_pct: number;
    stop_loss_pct: number;
    hold_max_bars: number;
    use_trailing: boolean;
    trail_be_pct: number;
    trail_lock_trigger_pct: number;
    trail_lock_pct: number;
    save_to_supabase: boolean;
    king_model_path: string;
    council_model_path: string;
    max_open_positions: number;
}

interface PerformanceData {
    total_trades: number;
    win_rate: number;
    profit_loss: number;
    profit_loss_pct: number;
    avg_trade_profit: number;
    max_drawdown: number;
    sharpe_ratio: number;
    exit_reasons: Record<string, number>;
    symbol_performance: Record<string, {
        trades: number;
        profit: number;
        win_rate: number;
    }>;
    open_positions: any[];
}

interface Trade {
    timestamp: string;
    symbol: string;
    action: string;
    amount: number;
    order_id: string;
}

interface BotStatus {
    status: "stopped" | "starting" | "running" | "stopping" | "error";
    config: BotConfig;
    last_scan: string | null;
    started_at: string | null;
    error: string | null;
    logs: string[];
    trades: Trade[];
}

type PollUnit = "s" | "m" | "h";

const COMMON_COINS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD",
    "DOGE/USD", "AVAX/USD", "MATIC/USD", "XRP/USD", "ADA/USD",
    "DOT/USD", "UNI/USD", "ATOM/USD", "XLM/USD", "ALGO/USD"
];

export default function LiveBotTab() {
    const [status, setStatus] = useState<BotStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [configForm, setConfigForm] = useState<Partial<BotConfig>>({});
    const [coinDialogOpen, setCoinDialogOpen] = useState(false);
    const [coinSearch, setCoinSearch] = useState("");
    const [availableCoins, setAvailableCoins] = useState<string[]>([]);
    const [alpacaAccount, setAlpacaAccount] = useState<AlpacaAccountInfo | null>(null);
    const [alpacaAccountLoading, setAlpacaAccountLoading] = useState(false);
    const [alpacaPositions, setAlpacaPositions] = useState<AlpacaPositionInfo[]>([]);
    const [alpacaPositionsLoading, setAlpacaPositionsLoading] = useState(false);

    const [pollIntervalValue, setPollIntervalValue] = useState(2);
    const [pollIntervalUnit, setPollIntervalUnit] = useState<PollUnit>("s");

    const [activeTab, setActiveTab] = useState<"control" | "performance">("control");
    const [performance, setPerformance] = useState<PerformanceData | null>(null);
    const [performanceLoading, setPerformanceLoading] = useState(false);
    const [uptime, setUptime] = useState("00:00:00");
    const logsEndRef = useRef<HTMLDivElement>(null);

    const fetchAccount = async (silent = false) => {
        if (!silent) setAlpacaAccountLoading(true);
        try {
            const acc = await getAlpacaAccount();
            setAlpacaAccount(acc);
        } catch (error) {
            console.error(error);
            if (!silent) toast.error("Failed to fetch Alpaca account");
        } finally {
            if (!silent) setAlpacaAccountLoading(false);
        }
    };

    const fetchPositions = async (silent = false) => {
        if (!silent) setAlpacaPositionsLoading(true);
        try {
            const pos = await getAlpacaPositions();
            setAlpacaPositions(Array.isArray(pos) ? pos : []);
        } catch (error) {
            console.error(error);
            if (!silent) toast.error("Failed to fetch Alpaca positions");
        } finally {
            if (!silent) setAlpacaPositionsLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        fetchAccount(true);
        fetchPositions(true);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Auto-scroll logs removed per user request

    // Uptime calculation
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (status?.status === 'running' && status?.started_at) {
            const start = new Date(status.started_at).getTime();
            interval = setInterval(() => {
                const now = Date.now();
                const diff = Math.max(0, now - start);
                const h = Math.floor(diff / 3600000);
                const m = Math.floor((diff % 3600000) / 60000);
                const s = Math.floor((diff % 60000) / 1000);
                setUptime(`${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
            }, 1000);
        } else {
            setUptime("00:00:00");
        }
        return () => clearInterval(interval);
    }, [status?.status, status?.started_at]);

    useEffect(() => {
        const secsRaw = configForm.poll_seconds;
        const secs = typeof secsRaw === "number" ? secsRaw : Number(secsRaw);
        if (!Number.isFinite(secs) || secs <= 0) return;

        let unit: PollUnit = "s";
        if (secs % 3600 === 0) unit = "h";
        else if (secs % 60 === 0) unit = "m";

        const multiplier = unit === "s" ? 1 : unit === "m" ? 60 : 3600;
        const value = Math.max(1, Math.round(secs / multiplier));

        setPollIntervalUnit(unit);
        setPollIntervalValue(value);
    }, [configForm.poll_seconds]);

    const setPollSecondsFromParts = (value: number, unit: PollUnit) => {
        const safeValue = Math.max(1, Math.floor(Number.isFinite(value) ? value : 1));
        const multiplier = unit === "s" ? 1 : unit === "m" ? 60 : 3600;
        const seconds = safeValue * multiplier;
        setPollIntervalValue(safeValue);
        setPollIntervalUnit(unit);
        setConfigForm((prev) => ({ ...prev, poll_seconds: seconds }));
    };

    const fmtUsd = (raw: string | number | undefined) => {
        const n = typeof raw === "number" ? raw : Number(raw ?? 0);
        const safe = Number.isFinite(n) ? n : 0;
        return safe.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
    };

    useEffect(() => {
        const fetchCoins = async () => {
            try {
                // Fetch coins that actually have data in the database (intraday bars)
                const res = await fetch("/api/alpaca/crypto-symbols-stats?timeframe=1h");
                if (res.ok) {
                    const data = await res.json();
                    if (Array.isArray(data) && data.length > 0) {
                        // The stats endpoint returns objects like { symbol: "BTC/USD", ... }
                        const symbols = data.map((item: any) => item.symbol).filter((s: any) => typeof s === 'string');
                        setAvailableCoins(symbols);

                        // Auto-select if bot is stopped or starting fresh
                        // We check if we are NOT running to avoid disrupting active config
                        // But since we are here on mount, status might be unknown.
                        // We rely on the useEffect below to sync.
                    }
                }
            } catch (e) {
                console.error("Failed to fetch available coins", e);
            }
        };
        fetchCoins();
    }, []);

    // Auto-select coins when available and bot is stopped
    useEffect(() => {
        if (availableCoins.length > 0 && status?.status !== 'running' && status?.status !== 'starting') {
            setConfigForm(prev => ({ ...prev, coins: availableCoins }));
        }
    }, [availableCoins, status?.status]);

    const pollSecondsRaw = configForm.poll_seconds;
    const pollSeconds = typeof pollSecondsRaw === "number" ? pollSecondsRaw : Number(pollSecondsRaw);
    const pollMs = Number.isFinite(pollSeconds) && pollSeconds > 0 ? pollSeconds * 1000 : 2000;

    const normalizePosKey = (s: string) => (s || "").toUpperCase().replace("/", "").replace("-", "").replace("_", "");

    const positionsBySymbol = new Map<string, AlpacaPositionInfo>(
        (alpacaPositions || []).map((p) => [normalizePosKey(p.symbol), p])
    );

    // Auto-refresh when running (match Poll Interval)
    useEffect(() => {
        const interval = setInterval(() => {
            if (status?.status === "running") {
                fetchStatus(true);
                fetchAccount(true);
                fetchPositions(true);
                if (activeTab === 'performance') {
                    fetchPerformance(true);
                }
            }
        }, pollMs);
        return () => clearInterval(interval);
    }, [status?.status, pollMs, activeTab]);

    const fetchStatus = async (silent = false) => {
        if (!silent) setRefreshing(true);
        try {
            const res = await fetch("/api/bot/status");
            if (res.ok) {
                const data = await res.json();
                setStatus(data);
                // Initialize config form if empty or if config updated externally
                if (!silent) {
                    setConfigForm(data.config);
                } else if (status?.status === 'running') {
                    // Don't overwrite form while typing if running, but maybe keep logs updated
                }
            }
        } catch (error) {
            console.error(error);
            if (!silent) toast.error("Failed to fetch bot status");
        } finally {
            if (!silent) {
                setLoading(false);
                setRefreshing(false);
            }
        }
    };

    const handleStart = async () => {
        try {
            const res = await fetch("/api/bot/start", { method: "POST" });
            if (res.ok) {
                toast.success("Bot started command sent");
                fetchStatus();
                fetchAccount(true);
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to start bot");
            }
        } catch (error) {
            toast.error("Failed to start bot");
        }
    };

    const handleStop = async () => {
        try {
            const res = await fetch("/api/bot/stop", { method: "POST" });
            if (res.ok) {
                toast.success("Bot stop command sent");
                fetchStatus();
                fetchAccount(true);
            } else {
                toast.error("Failed to stop bot");
            }
        } catch (error) {
            toast.error("Failed to stop bot");
        }
    };

    const handleUpdateConfig = async () => {
        try {
            const res = await fetch("/api/bot/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(configForm)
            });
            if (res.ok) {
                const data = await res.json();
                toast.success("Configuration updated");
                setStatus((prev) => prev ? { ...prev, config: data.config } : prev);
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to update config");
            }
        } catch (error) {
            toast.error("Failed to update config");
        }
    }

    const toggleCoin = (coin: string) => {
        const current = configForm.coins || [];
        if (current.includes(coin)) {
            setConfigForm({ ...configForm, coins: current.filter(c => c !== coin) });
        } else {
            setConfigForm({ ...configForm, coins: [...current, coin] });
        }
    };

    const fetchPerformance = async (silent = false) => {
        if (!silent) setPerformanceLoading(true);
        try {
            const res = await fetch("/api/bot/performance");
            if (res.ok) {
                const data = await res.json();
                setPerformance(data);
            }
        } catch (error) {
            console.error("Failed to fetch performance", error);
        } finally {
            if (!silent) setPerformanceLoading(false);
        }
    };

    useEffect(() => {
        if (activeTab === 'performance') {
            fetchPerformance();
        }
    }, [activeTab]);

    if (loading && !status) {
        return <div className="flex items-center justify-center h-96 text-indigo-400 animate-pulse font-mono tracking-widest">INITIALIZING CONNECTION...</div>;
    }

    const isRunning = status?.status === "running" || status?.status === "starting";
    const useCouncil = configForm.use_council ?? true;

    return (
        <div className="max-w-[1800px] mx-auto p-6 md:p-8 space-y-8">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative">
                <div className="relative z-10">
                    <h1 className="text-4xl font-black tracking-tighter text-white mb-2 flex items-center gap-4 drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]">
                        <Activity className="w-10 h-10 text-indigo-500 animate-pulse-slow" />
                        LIVE BOT CONTROL
                    </h1>
                    <div className="flex flex-wrap items-center gap-3">
                        <div className="flex items-center gap-3 bg-black/40 backdrop-blur-md px-4 py-2 rounded-full border border-white/5 w-fit">
                            <div className={`w-3 h-3 rounded-full shadow-[0_0_10px_currentColor] ${isRunning ? "bg-emerald-500 text-emerald-500 animate-pulse" : "bg-red-500 text-red-500"}`} />
                            <span className="text-sm font-bold text-zinc-300 uppercase tracking-widest">
                                Status: <span className={isRunning ? "text-emerald-400" : "text-red-400"}>{status?.status || "UNKNOWN"}</span>
                            </span>
                            {status?.last_scan && (
                                <>
                                    <span className="text-zinc-700">|</span>
                                    <span className="text-xs text-zinc-500 font-mono flex items-center gap-1.5">
                                        <RefreshCw className="w-3 h-3" /> {new Date(status.last_scan).toLocaleTimeString()}
                                    </span>
                                </>
                            )}
                        </div>

                        {isRunning && (
                            <div className="flex items-center gap-3 bg-indigo-500/10 backdrop-blur-md px-4 py-2 rounded-full border border-indigo-500/20 w-fit">
                                <Clock className="w-4 h-4 text-indigo-400 animate-spin-slow" />
                                <span className="text-sm font-mono font-bold text-indigo-300 tracking-wider">
                                    UPTIME: {uptime}
                                </span>
                            </div>
                        )}
                    </div>

                    <div className="mt-4 flex flex-wrap gap-3 text-xs">
                        <div className="px-3 py-2 rounded-2xl bg-zinc-900/50 border border-white/5">
                            <div className="text-zinc-500 uppercase font-bold tracking-widest">Alpaca Cash</div>
                            <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.cash)}</div>
                        </div>
                        <div className="px-3 py-2 rounded-2xl bg-zinc-900/50 border border-white/5">
                            <div className="text-zinc-500 uppercase font-bold tracking-widest">Buying Power</div>
                            <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.buying_power)}</div>
                        </div>
                        <div className="px-3 py-2 rounded-2xl bg-zinc-900/50 border border-white/5">
                            <div className="text-zinc-500 uppercase font-bold tracking-widest">Portfolio</div>
                            <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.portfolio_value)}</div>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={() => {
                            fetchStatus();
                            fetchAccount();
                            fetchPositions();
                        }}
                        className="p-4 rounded-2xl bg-zinc-900/50 border border-white/5 hover:bg-white/10 hover:border-white/10 transition-all group"
                    >
                        <RefreshCw className={`w-5 h-5 text-zinc-400 group-hover:text-white transition-colors ${refreshing ? "animate-spin" : ""}`} />
                    </button>

                    {!isRunning ? (
                        <button
                            onClick={handleStart}
                            className="px-8 py-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white font-black text-sm tracking-widest flex items-center gap-3 transition-all shadow-[0_0_30px_rgba(16,185,129,0.3)] hover:shadow-[0_0_50px_rgba(16,185,129,0.5)] transform hover:scale-105"
                        >
                            <Play className="w-5 h-5 fill-current" />
                            START SYSTEM
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            className="px-8 py-4 rounded-2xl bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-black text-sm tracking-widest flex items-center gap-3 transition-all shadow-[0_0_30px_rgba(239,68,68,0.3)] hover:shadow-[0_0_50px_rgba(239,68,68,0.5)] transform hover:scale-105"
                        >
                            <Square className="w-5 h-5 fill-current" />
                            STOP SYSTEM
                        </button>
                    )}
                </div>
            </div>

            {/* Tabs Navigation */}
            <div className="flex items-center gap-2 p-1.5 bg-zinc-900/50 border border-white/5 rounded-2xl w-full">
                <button
                    onClick={() => setActiveTab('control')}
                    className={`flex-1 px-6 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeTab === 'control'
                        ? 'bg-white text-black shadow-lg shadow-white/5'
                        : 'text-zinc-500 hover:text-zinc-300'
                        }`}
                >
                    <Settings className="w-4 h-4" />
                    System Control
                </button>
                <button
                    onClick={() => setActiveTab('performance')}
                    className={`flex-1 px-6 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeTab === 'performance'
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
                        : 'text-zinc-500 hover:text-zinc-300'
                        }`}
                >
                    <Activity className="w-4 h-4" />
                    Performance Analytics
                </button>
            </div>

            {activeTab === 'control' ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    {/* Configuration Panel */}
                    <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl lg:col-span-1 shadow-2xl relative overflow-hidden group">
                        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-transparent to-purple-500/5 pointer-events-none" />

                        <div className="flex items-center gap-3 mb-8 relative z-10">
                            <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                                <Settings className="w-5 h-5 text-indigo-400" />
                            </div>
                            <h2 className="text-xl font-bold tracking-tight text-white">SYSTEM CONFIGURATION</h2>
                        </div>

                        <div className="space-y-6 relative z-10">
                            {/* Coins Selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2">
                                    <Coins className="w-3 h-3" /> Target Assets
                                </label>
                                <div className="flex flex-wrap gap-2 mb-2 min-h-[42px] p-2 bg-black/40 border border-white/5 rounded-xl">
                                    {(configForm.coins || []).map(coin => (
                                        <span key={coin} className="px-2 py-1 rounded-lg bg-indigo-500/20 text-indigo-300 text-xs font-bold border border-indigo-500/20 flex items-center gap-1">
                                            {coin}
                                            <button onClick={() => toggleCoin(coin)} className="hover:text-white"><X className="w-3 h-3" /></button>
                                        </span>
                                    ))}
                                    <button
                                        onClick={() => setCoinDialogOpen(true)}
                                        className="px-2 py-1 rounded-lg bg-zinc-800 text-zinc-400 text-xs font-bold hover:bg-zinc-700 hover:text-white transition-colors flex items-center gap-1"
                                    >
                                        <Plus className="w-3 h-3" /> ADD
                                    </button>
                                </div>
                            </div>

                            {/* Council Validation Section */}
                            <div className="bg-black/40 rounded-2xl p-5 border border-white/5 space-y-4 shadow-xl">
                                <div className="flex items-center justify-between">
                                    <div className="space-y-1">
                                        <label className="text-sm font-black text-white flex items-center gap-2 tracking-tight">
                                            {useCouncil ? <ShieldCheck className="w-4 h-4 text-emerald-400" /> : <ShieldAlert className="w-4 h-4 text-zinc-500" />}
                                            COUNCIL VALIDATION
                                        </label>
                                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest">Multi-model consensus</p>
                                    </div>
                                    <Switch.Root
                                        checked={useCouncil}
                                        onCheckedChange={(c: boolean) => setConfigForm({ ...configForm, use_council: c })}
                                        className={`w-12 h-6 rounded-full relative shadow-inner transition-colors duration-300 ${useCouncil ? 'bg-emerald-600' : 'bg-zinc-700'}`}
                                    >
                                        <Switch.Thumb className={`block w-4 h-4 rounded-full bg-white shadow-lg transition-transform duration-300 transform translate-y-1 ${useCouncil ? 'translate-x-7' : 'translate-x-1'}`} />
                                    </Switch.Root>
                                </div>

                                {useCouncil && (
                                    <div className="space-y-4 pt-2 border-t border-white/5 animate-in fade-in duration-300">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Council Threshold</label>
                                            <input
                                                type="number" step="0.01"
                                                value={configForm.council_threshold}
                                                onChange={(e) => setConfigForm({ ...configForm, council_threshold: parseFloat(e.target.value) })}
                                                className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-indigo-500/50 transition-all text-indigo-200"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Council Validator Path</label>
                                            <input
                                                type="text"
                                                value={configForm.council_model_path}
                                                onChange={(e) => setConfigForm({ ...configForm, council_model_path: e.target.value })}
                                                className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-xs font-mono focus:outline-none focus:border-indigo-500/50 transition-all text-zinc-400"
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Model Hub Section */}
                            <div className="bg-black/40 rounded-2xl p-5 border border-white/5 space-y-4 shadow-xl">
                                <label className="text-xs font-black text-zinc-400 uppercase tracking-[0.2em] flex items-center gap-2">
                                    <ShieldCheck className="w-3.5 h-3.5 text-purple-400" /> Model Intelligence
                                </label>
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">King Model Path</label>
                                        <input
                                            type="text"
                                            value={configForm.king_model_path}
                                            onChange={(e) => setConfigForm({ ...configForm, king_model_path: e.target.value })}
                                            className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-xs font-mono focus:outline-none focus:border-purple-500/50 transition-all text-zinc-400"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">King Confidence Threshold</label>
                                        <input
                                            type="number" step="0.01"
                                            value={configForm.king_threshold}
                                            onChange={(e) => setConfigForm({ ...configForm, king_threshold: parseFloat(e.target.value) })}
                                            className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-purple-500/50 transition-all text-purple-200"
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Exit Strategy Section */}
                            <div className="bg-black/40 rounded-2xl p-5 border border-white/5 space-y-5 shadow-xl">
                                <div className="flex items-center justify-between">
                                    <div className="space-y-1">
                                        <label className="text-sm font-black text-white tracking-tight flex items-center gap-2">
                                            <ArrowDownRight className="w-4 h-4 text-orange-400" /> EXIT STRATEGY
                                        </label>
                                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest">Auto-sell logic</p>
                                    </div>
                                    <Switch.Root
                                        checked={Boolean(configForm.enable_sells ?? true)}
                                        onCheckedChange={(c: boolean) => setConfigForm({ ...configForm, enable_sells: c })}
                                        className={`w-12 h-6 rounded-full relative shadow-inner transition-colors duration-300 ${(configForm.enable_sells ?? true) ? 'bg-orange-600' : 'bg-zinc-700'}`}
                                    >
                                        <Switch.Thumb className={`block w-4 h-4 rounded-full bg-white shadow-lg transition-transform duration-300 transform translate-y-1 ${(configForm.enable_sells ?? true) ? 'translate-x-7' : 'translate-x-1'}`} />
                                    </Switch.Root>
                                </div>

                                {configForm.enable_sells && (
                                    <div className="space-y-5 animate-in fade-in duration-300">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Target %</label>
                                                <input
                                                    type="number" step="0.01"
                                                    value={configForm.target_pct}
                                                    onChange={(e) => setConfigForm({ ...configForm, target_pct: parseFloat(e.target.value) })}
                                                    className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-emerald-500/50 transition-all text-emerald-200"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Stop Loss %</label>
                                                <input
                                                    type="number" step="0.01"
                                                    value={configForm.stop_loss_pct}
                                                    onChange={(e) => setConfigForm({ ...configForm, stop_loss_pct: parseFloat(e.target.value) })}
                                                    className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-red-500/50 transition-all text-red-200"
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-4 pt-4 border-t border-white/5">
                                            <div className="flex items-center justify-between">
                                                <div className="space-y-1">
                                                    <label className="text-xs font-bold text-white tracking-wider flex items-center gap-2">
                                                        <Activity className="w-3 h-3 text-blue-400" /> Trailing Stop
                                                    </label>
                                                </div>
                                                <Switch.Root
                                                    checked={Boolean(configForm.use_trailing ?? true)}
                                                    onCheckedChange={(c: boolean) => setConfigForm({ ...configForm, use_trailing: c })}
                                                    className={`w-10 h-5 rounded-full relative shadow-inner transition-colors duration-300 ${(configForm.use_trailing ?? true) ? 'bg-blue-600' : 'bg-zinc-700'}`}
                                                >
                                                    <Switch.Thumb className={`block w-3 h-3 rounded-full bg-white shadow-lg transition-transform duration-300 transform translate-y-1 ${(configForm.use_trailing ?? true) ? 'translate-x-6' : 'translate-x-1'}`} />
                                                </Switch.Root>
                                            </div>

                                            {configForm.use_trailing && (
                                                <div className="grid grid-cols-3 gap-2 animate-in slide-in-from-top-2 duration-300">
                                                    <div className="space-y-1">
                                                        <label className="text-[8px] font-black text-zinc-600 uppercase">BE %</label>
                                                        <input
                                                            type="number" step="0.01"
                                                            value={configForm.trail_be_pct}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_be_pct: parseFloat(e.target.value) })}
                                                            className="w-full bg-black/60 border border-white/5 rounded-lg px-2 py-1.5 text-[10px] font-mono focus:outline-none transition-all"
                                                        />
                                                    </div>
                                                    <div className="space-y-1">
                                                        <label className="text-[8px] font-black text-zinc-600 uppercase">Lock Trigg %</label>
                                                        <input
                                                            type="number" step="0.01"
                                                            value={configForm.trail_lock_trigger_pct}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_lock_trigger_pct: parseFloat(e.target.value) })}
                                                            className="w-full bg-black/60 border border-white/5 rounded-lg px-2 py-1.5 text-[10px] font-mono focus:outline-none transition-all"
                                                        />
                                                    </div>
                                                    <div className="space-y-1">
                                                        <label className="text-[8px] font-black text-zinc-600 uppercase">Lock Profit %</label>
                                                        <input
                                                            type="number" step="0.01"
                                                            value={configForm.trail_lock_pct}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_lock_pct: parseFloat(e.target.value) })}
                                                            className="w-full bg-black/60 border border-white/5 rounded-lg px-2 py-1.5 text-[10px] font-mono focus:outline-none transition-all"
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Hold Max Bars</label>
                                            <input
                                                type="number" min={1}
                                                value={configForm.hold_max_bars}
                                                onChange={(e) => setConfigForm({ ...configForm, hold_max_bars: parseInt(e.target.value) })}
                                                className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none border-white/5 transition-all"
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Risk Management Section */}
                            <div className="bg-black/40 rounded-2xl p-5 border border-white/5 space-y-5 shadow-xl font-mono">
                                <label className="text-xs font-black text-red-400 uppercase tracking-[0.2em] flex items-center gap-2">
                                    <ShieldAlert className="w-3.5 h-3.5" /> Risk Firewall
                                </label>
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center bg-red-500/5 rounded-xl p-3 border border-red-500/10">
                                        <label className="text-[10px] font-black text-red-500/70 uppercase">Max Open Trades</label>
                                        <input
                                            type="number" min={1}
                                            value={configForm.max_open_positions}
                                            onChange={(e) => setConfigForm({ ...configForm, max_open_positions: parseInt(e.target.value) })}
                                            className="w-16 bg-transparent text-right text-sm font-black text-red-400 focus:outline-none"
                                        />
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-zinc-600 uppercase">Per-Trade Notional ($)</label>
                                            <input
                                                type="number"
                                                value={configForm.max_notional_usd}
                                                onChange={(e) => setConfigForm({ ...configForm, max_notional_usd: parseFloat(e.target.value) })}
                                                className="w-full bg-black/40 border border-white/5 rounded-xl px-3 py-2 text-sm text-zinc-300 focus:outline-none border-white/10"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-zinc-600 uppercase">% Cash Allocate</label>
                                            <input
                                                type="number" step="0.01"
                                                value={configForm.pct_cash_per_trade}
                                                onChange={(e) => setConfigForm({ ...configForm, pct_cash_per_trade: parseFloat(e.target.value) })}
                                                className="w-full bg-black/40 border border-white/5 rounded-xl px-3 py-2 text-sm text-zinc-300 focus:outline-none border-white/10"
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Poll Interval</label>
                                    <div className="flex gap-2">
                                        <input
                                            type="number"
                                            min={1}
                                            value={pollIntervalValue}
                                            onChange={(e) => setPollSecondsFromParts(Number(e.target.value), pollIntervalUnit)}
                                            className="flex-1 min-w-0 bg-black/40 border border-white/5 rounded-xl px-3 py-2 text-sm font-mono focus:outline-none focus:border-white/20 transition-all"
                                        />
                                        <select
                                            value={pollIntervalUnit}
                                            onChange={(e) => {
                                                const nextUnit = e.target.value as PollUnit;
                                                setPollSecondsFromParts(pollIntervalValue, nextUnit);
                                            }}
                                            className="shrink-0 w-[5.5rem] bg-black/40 border border-white/5 rounded-xl px-3 py-2 text-sm font-mono focus:outline-none focus:border-white/20 transition-all text-white"
                                        >
                                            <option value="s" className="bg-zinc-950 text-white">sec</option>
                                            <option value="m" className="bg-zinc-950 text-white">min</option>
                                            <option value="h" className="bg-zinc-950 text-white">hour</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Bars Limit</label>
                                    <input
                                        type="number"
                                        value={configForm.bars_limit}
                                        onChange={(e) => setConfigForm({ ...configForm, bars_limit: parseInt(e.target.value) })}
                                        className="w-full bg-black/40 border border-white/5 rounded-xl px-3 py-2 text-sm font-mono focus:outline-none focus:border-white/20 transition-all"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-xs font-bold text-zinc-500 uppercase">Timeframe</label>
                                <div className="relative">
                                    <select
                                        value={configForm.timeframe || "1Hour"}
                                        onChange={(e) => setConfigForm({ ...configForm, timeframe: e.target.value })}
                                        className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-indigo-500/50 focus:bg-indigo-500/5 transition-all appearance-none cursor-pointer text-white"
                                    >
                                        <option value="1Min" className="bg-zinc-950 text-white">1 Minute</option>
                                        <option value="5Min" className="bg-zinc-950 text-white">5 Minutes</option>
                                        <option value="15Min" className="bg-zinc-950 text-white">15 Minutes</option>
                                        <option value="30Min" className="bg-zinc-950 text-white">30 Minutes</option>
                                        <option value="1Hour" className="bg-zinc-950 text-white">1 Hour (Default)</option>
                                        <option value="4Hour" className="bg-zinc-950 text-white">4 Hours</option>
                                        <option value="1Day" className="bg-zinc-950 text-white">1 Day</option>
                                    </select>
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                                        <Settings className="w-4 h-4" />
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-xs font-bold text-zinc-500 uppercase">Data Source</label>
                                <select
                                    value={(configForm.data_source as string) || "alpaca"}
                                    onChange={(e) => setConfigForm({ ...configForm, data_source: e.target.value })}
                                    className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-white/20 transition-all text-white"
                                >
                                    <option value="alpaca" className="bg-zinc-950 text-white">Alpaca</option>
                                    <option value="binance" className="bg-zinc-950 text-white">Binance</option>
                                </select>
                            </div>

                            {/* Save to Supabase Toggle */}
                            <div className="bg-black/20 rounded-xl p-4 border border-white/5 flex items-center justify-between">
                                <div className="space-y-1">
                                    <label className="text-sm font-bold text-white flex items-center gap-2">
                                        <Save className="w-4 h-4 text-indigo-400" />
                                        Save to Supabase
                                    </label>
                                    <p className="text-xs text-zinc-500">Persist poll data to DB</p>
                                </div>
                                <Switch.Root
                                    checked={configForm.save_to_supabase ?? true}
                                    onCheckedChange={(c: boolean) => setConfigForm({ ...configForm, save_to_supabase: c })}
                                    className={`w-12 h-6 rounded-full relative shadow-inner transition-colors duration-300 ${configForm.save_to_supabase !== false ? 'bg-indigo-600' : 'bg-zinc-700'}`}
                                >
                                    <Switch.Thumb className={`block w-4 h-4 rounded-full bg-white shadow-lg transition-transform duration-300 transform translate-y-1 ${configForm.save_to_supabase !== false ? 'translate-x-7' : 'translate-x-1'}`} />
                                </Switch.Root>
                            </div>

                            <button
                                onClick={handleUpdateConfig}
                                disabled={isRunning}
                                className={`w-full py-4 rounded-xl font-black tracking-wider flex items-center justify-center gap-2 transition-all ${isRunning
                                    ? "bg-zinc-800/50 text-zinc-600 cursor-not-allowed border border-zinc-800"
                                    : "bg-white text-black hover:bg-indigo-50 shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(99,102,241,0.3)]"
                                    }`}
                            >
                                <Save className="w-4 h-4" />
                                {isRunning ? "STOP SYSTEM TO UPDATE" : "SAVE CONFIGURATION"}
                            </button>
                        </div>
                    </div>

                    {/* Right Column: Logs & Trades */}
                    <div className="lg:col-span-2 flex flex-col gap-6 h-full">
                        {/* Live Logs */}
                        <div className="flex-1 bg-black border border-zinc-800 rounded-3xl p-1 shadow-2xl overflow-hidden flex flex-col min-h-[400px]">
                            <div className="flex items-center justify-between px-6 py-4 bg-zinc-900/50 border-b border-zinc-800">
                                <div className="flex items-center gap-3">
                                    <Terminal className="w-5 h-5 text-emerald-400" />
                                    <h2 className="text-sm font-bold tracking-widest text-zinc-300">SYSTEM LOGS</h2>
                                </div>
                                <div className="flex gap-2">
                                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50" />
                                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/20 border border-emerald-500/50" />
                                </div>
                            </div>

                            <div className="flex-1 p-6 overflow-y-auto font-mono text-xs space-y-1.5 custom-scrollbar">
                                {status?.logs && status.logs.length > 0 ? (
                                    status.logs.map((log, i) => (
                                        <div key={i} className={`break-all border-l-2 pl-3 py-0.5 ${log.includes("BUY") ? "border-emerald-500 text-emerald-400 bg-emerald-500/5" :
                                            log.includes("ERROR") ? "border-red-500 text-red-500 bg-red-500/5" :
                                                log.includes("SIGNAL") ? "border-indigo-500 text-indigo-300" :
                                                    "border-zinc-800 text-zinc-400"
                                            }`}>
                                            <span className="opacity-50 mr-2">{log.split("]")[0]}]</span>
                                            {log.split("]")[1]}
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-zinc-700 italic flex items-center justify-center h-full">Waiting for data stream...</div>
                                )}
                                <div ref={logsEndRef} />
                            </div>
                        </div>

                        {/* Recent Trades */}
                        <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl">
                            <div className="flex items-center gap-3 mb-4">
                                <Activity className="w-5 h-5 text-purple-400" />
                                <h2 className="text-sm font-bold tracking-widest text-white">RECENT EXECUTIONS</h2>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-[10px] font-black text-zinc-500 uppercase bg-black/20 tracking-wider">
                                        <tr>
                                            <th className="px-4 py-3 rounded-l-lg">Time</th>
                                            <th className="px-4 py-3">Symbol</th>
                                            <th className="px-4 py-3">Action</th>
                                            <th className="px-4 py-3">Amount</th>
                                            <th className="px-4 py-3">Today's P/L ($)</th>
                                            <th className="px-4 py-3">Market Value</th>
                                            <th className="px-4 py-3 rounded-r-lg">Order ID</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-white/5">
                                        {status?.trades && status.trades.length > 0 ? (
                                            status.trades.map((trade, i) => {
                                                const pos = positionsBySymbol.get(normalizePosKey(trade.symbol));
                                                return (
                                                    <tr key={i} className="hover:bg-white/5 transition-colors group">
                                                        <td className="px-4 py-4 font-mono text-zinc-400 group-hover:text-white transition-colors">
                                                            {new Date(trade.timestamp).toLocaleTimeString()}
                                                        </td>
                                                        <td className="px-4 py-4 font-bold text-white shadow-[0_0_10px_transparent] group-hover:shadow-[0_0_15px_rgba(255,255,255,0.1)] transition-all">{trade.symbol}</td>
                                                        <td className="px-4 py-4">
                                                            <span className={`px-2 py-1 rounded text-[10px] font-black uppercase ${trade.action === 'BUY' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/20 text-red-400 border border-red-500/20'
                                                                }`}>
                                                                {trade.action}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-4 text-zinc-300 font-mono">${trade.amount.toFixed(2)}</td>
                                                        <td className="px-4 py-4 text-zinc-300 font-mono">{alpacaPositionsLoading ? "…" : fmtUsd(pos?.unrealized_intraday_pl || 0)}</td>
                                                        <td className="px-4 py-4 text-zinc-300 font-mono">{alpacaPositionsLoading ? "…" : fmtUsd(pos?.market_value || 0)}</td>
                                                        <td className="px-4 py-4 font-mono text-xs text-zinc-600">{trade.order_id}</td>
                                                    </tr>
                                                );
                                            })
                                        ) : (
                                            <tr>
                                                <td colSpan={7} className="px-4 py-8 text-center text-zinc-600 italic">
                                                    No trades executed yet
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    {performanceLoading && !performance ? (
                        <div className="flex flex-col items-center justify-center h-96 space-y-4">
                            <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin" />
                            <p className="text-zinc-500 font-mono text-sm tracking-widest">ANALYZING MARKET DATA...</p>
                        </div>
                    ) : (
                        <>
                            {/* Summary Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl group hover:border-indigo-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                            <Activity className="w-5 h-5" />
                                        </div>
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Total Trades</span>
                                    </div>
                                    <div className="text-3xl font-black text-white">{performance?.total_trades || 0}</div>
                                    <div className="mt-2 text-xs text-zinc-500">System executions</div>
                                </div>

                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl group hover:border-emerald-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                                            <BarChart3 className="w-5 h-5" />
                                        </div>
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Win Rate</span>
                                    </div>
                                    <div className="text-3xl font-black text-emerald-400">{(performance?.win_rate || 0).toFixed(1)}%</div>
                                    <div className="mt-2 flex items-center gap-1 text-xs text-zinc-500">
                                        <ArrowUpRight className="w-3 h-3" /> Consensus Alpha
                                    </div>
                                </div>

                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl group hover:border-indigo-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                            <PieChart className="w-5 h-5" />
                                        </div>
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Total P/L</span>
                                    </div>
                                    <div className={`text-3xl font-black ${(performance?.profit_loss || 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                        ${(performance?.profit_loss || 0).toFixed(2)}
                                    </div>
                                    <div className={`mt-2 text-xs font-bold ${(performance?.profit_loss_pct || 0) >= 0 ? "text-emerald-500/70" : "text-red-500/70"}`}>
                                        {(performance?.profit_loss_pct || 0).toFixed(2)}% ROI
                                    </div>
                                </div>

                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl group hover:border-indigo-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                            <Activity className="w-5 h-5" />
                                        </div>
                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Avg Profit</span>
                                    </div>
                                    <div className={`text-3xl font-black ${(performance?.avg_trade_profit || 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                        ${(performance?.avg_trade_profit || 0).toFixed(2)}
                                    </div>
                                    <div className="mt-2 text-xs text-zinc-500">Per trade average</div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                {/* Exit Reasons */}
                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl">
                                    <div className="flex items-center gap-3 mb-6">
                                        <PieChart className="w-5 h-5 text-indigo-400" />
                                        <h2 className="text-lg font-bold text-white tracking-tight">EXIT REASONS</h2>
                                    </div>
                                    <div className="space-y-4">
                                        {performance?.exit_reasons && Object.entries(performance.exit_reasons).length > 0 ? (
                                            Object.entries(performance.exit_reasons).map(([reason, count]) => {
                                                const total = Object.values(performance.exit_reasons).reduce((a, b) => a + b, 0);
                                                const pct = (count / total) * 100;
                                                return (
                                                    <div key={reason} className="space-y-1.5">
                                                        <div className="flex justify-between text-xs font-bold uppercase tracking-wider">
                                                            <span className="text-zinc-400">{reason.replace(/_/g, ' ')}</span>
                                                            <span className="text-white">{count} ({pct.toFixed(0)}%)</span>
                                                        </div>
                                                        <div className="h-1.5 bg-black/40 rounded-full overflow-hidden border border-white/5">
                                                            <div
                                                                className={`h-full rounded-full transition-all duration-1000 ${reason.includes('target') ? 'bg-emerald-500' :
                                                                    reason.includes('stop') ? 'bg-red-500' :
                                                                        'bg-indigo-500'
                                                                    }`}
                                                                style={{ width: `${pct}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                );
                                            })
                                        ) : (
                                            <div className="text-center py-12 text-zinc-600 italic">No exit data recorded</div>
                                        )}
                                    </div>
                                </div>

                                {/* Asset Performance */}
                                <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl">
                                    <div className="flex items-center gap-3 mb-6">
                                        <BarChart3 className="w-5 h-5 text-emerald-400" />
                                        <h2 className="text-lg font-bold text-white tracking-tight">ASSET PERFORMANCE</h2>
                                    </div>
                                    <div className="overflow-hidden rounded-2xl border border-white/5">
                                        <table className="w-full text-left text-sm">
                                            <thead className="bg-black/40 text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                                <tr>
                                                    <th className="px-4 py-3">Symbol</th>
                                                    <th className="px-4 py-3 text-center">Trades</th>
                                                    <th className="px-4 py-3 text-center">Win Rate</th>
                                                    <th className="px-4 py-3 text-right">Profit ($)</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5 font-mono text-xs">
                                                {performance?.symbol_performance && Object.entries(performance.symbol_performance).length > 0 ? (
                                                    Object.entries(performance.symbol_performance)
                                                        .sort(([, a], [, b]) => b.profit - a.profit)
                                                        .map(([symbol, stats]) => (
                                                            <tr key={symbol} className="hover:bg-white/5 transition-colors">
                                                                <td className="px-4 py-3 font-bold text-white">{symbol}</td>
                                                                <td className="px-4 py-3 text-center text-zinc-400">{stats.trades}</td>
                                                                <td className="px-4 py-3 text-center">
                                                                    <span className={stats.win_rate >= 50 ? 'text-emerald-400' : 'text-zinc-500'}>
                                                                        {stats.win_rate.toFixed(0)}%
                                                                    </span>
                                                                </td>
                                                                <td className={`px-4 py-3 text-right font-bold ${stats.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                    {stats.profit >= 0 ? '+' : ''}{stats.profit.toFixed(2)}
                                                                </td>
                                                            </tr>
                                                        ))
                                                ) : (
                                                    <tr>
                                                        <td colSpan={4} className="px-4 py-12 text-center text-zinc-600 italic">No asset data available</td>
                                                    </tr>
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

                            {/* Open Positions */}
                            <div className="bg-zinc-900/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl">
                                <div className="flex items-center gap-3 mb-6">
                                    <ArrowUpRight className="w-5 h-5 text-indigo-400" />
                                    <h2 className="text-lg font-bold text-white tracking-tight">ACTIVE POSITIONS</h2>
                                </div>
                                <div className="overflow-x-auto rounded-2xl border border-white/5">
                                    <table className="w-full text-left text-sm">
                                        <thead className="bg-black/40 text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                                            <tr>
                                                <th className="px-4 py-4">Symbol</th>
                                                <th className="px-4 py-4">Status</th>
                                                <th className="px-4 py-4">Entry</th>
                                                <th className="px-4 py-4">Current</th>
                                                <th className="px-4 py-4">P/L %</th>
                                                <th className="px-4 py-4 text-right">P/L $</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5 font-mono text-xs">
                                            {performance?.open_positions && performance.open_positions.length > 0 ? (
                                                performance.open_positions.map((pos, i) => (
                                                    <tr key={i} className="hover:bg-white/5 transition-colors group">
                                                        <td className="px-4 py-4 font-bold text-white flex items-center gap-2">
                                                            {pos.symbol}
                                                        </td>
                                                        <td className="px-4 py-4">
                                                            <span className="px-2 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/10 text-[10px] font-black uppercase tracking-tighter">
                                                                In Progress
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-4 text-zinc-400">${(pos.entry_price || 0).toFixed(4)}</td>
                                                        <td className="px-4 py-4 text-white">${(pos.current_price || 0).toFixed(4)}</td>
                                                        <td className={`px-4 py-4 font-bold ${(pos.pl_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            {pos.pl_pct >= 0 ? '+' : ''}{pos.pl_pct.toFixed(2)}%
                                                        </td>
                                                        <td className={`px-4 py-4 text-right font-bold ${(pos.pl_usd || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            ${(pos.pl_usd || 0).toFixed(2)}
                                                        </td>
                                                    </tr>
                                                ))
                                            ) : (
                                                <tr>
                                                    <td colSpan={6} className="px-4 py-12 text-center text-zinc-600 italic font-mono uppercase tracking-[0.2em] text-[10px]">
                                                        Scan in progress... No active targets locked.
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* Coin Selection Dialog */}
            <Dialog.Root open={coinDialogOpen} onOpenChange={setCoinDialogOpen}>
                <Dialog.Portal>
                    <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 transition-opacity" />
                    <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-zinc-900 border border-zinc-800 rounded-3xl p-6 z-50 shadow-2xl">
                        <div className="flex items-center justify-between mb-6">
                            <Dialog.Title className="text-xl font-bold text-white flex items-center gap-2">
                                <Coins className="w-6 h-6 text-indigo-500" />
                                Manage Assets
                            </Dialog.Title>
                            <Dialog.Close className="p-2 rounded-full hover:bg-white/10 text-zinc-400 hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </Dialog.Close>
                        </div>

                        <div className="relative mb-4">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                            <input
                                type="text"
                                placeholder="Search coins..."
                                value={coinSearch}
                                onChange={(e) => setCoinSearch(e.target.value)}
                                className="w-full bg-black/50 border border-zinc-800 rounded-xl pl-10 pr-4 py-3 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-2 max-h-[300px] overflow-y-auto custom-scrollbar pr-2">
                            {(availableCoins.length > 0 ? availableCoins : COMMON_COINS).filter(c => c.toLowerCase().includes(coinSearch.toLowerCase())).map(coin => {
                                const active = (configForm.coins || []).includes(coin);
                                return (
                                    <button
                                        key={coin}
                                        onClick={() => toggleCoin(coin)}
                                        className={`px-4 py-3 rounded-xl border flex items-center justify-between group transition-all ${active
                                            ? "bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                                            : "bg-zinc-800/50 border-transparent text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                                            }`}
                                    >
                                        <span className="font-bold text-sm">{coin}</span>
                                        {active && <Check className="w-4 h-4" />}
                                    </button>
                                )
                            })}
                        </div>

                        <div className="mt-6 flex justify-end">
                            <Dialog.Close className="px-6 py-2.5 rounded-xl bg-white text-black font-bold hover:bg-zinc-200 transition-colors">
                                Done
                            </Dialog.Close>
                        </div>
                    </Dialog.Content>
                </Dialog.Portal>
            </Dialog.Root>

            <style jsx global>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.05);
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
            `}</style>
        </div>
    );
}
