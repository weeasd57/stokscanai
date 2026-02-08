"use client";

import { useState, useEffect } from "react";
import { Play, Square, Activity, Settings, Terminal, RefreshCw, Save } from "lucide-react";
import { toast } from "sonner";
import AdminHeader from "../components/AdminHeader";
import { getAlpacaAccount, type AlpacaAccountInfo } from "@/lib/api";

// Types
interface BotConfig {
    alpaca_key_id: string;
    alpaca_secret_key: string;
    alpaca_base_url: string;
    coins: string[];
    king_threshold: number;
    council_threshold: number;
    max_notional_usd: number;
    pct_cash_per_trade: number;
    bars_limit: number;
    poll_seconds: number;
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
    error: string | null;
    logs: string[];
    trades: Trade[];
}

type PollUnit = "s" | "m" | "h";

export default function BotPage() {
    const [status, setStatus] = useState<BotStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [configForm, setConfigForm] = useState<Partial<BotConfig>>({});
    const [alpacaAccount, setAlpacaAccount] = useState<AlpacaAccountInfo | null>(null);
    const [alpacaAccountLoading, setAlpacaAccountLoading] = useState(false);

    const [pollIntervalValue, setPollIntervalValue] = useState(2);
    const [pollIntervalUnit, setPollIntervalUnit] = useState<PollUnit>("s");

    // Auto-refresh logs when running
    useEffect(() => {
        fetchStatus();
        const interval = setInterval(() => {
            if (status?.status === "running") {
                fetchStatus(true);
            }
        }, 2000);
        return () => clearInterval(interval);
    }, [status?.status]);

    useEffect(() => {
        fetchAccount(true);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

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

    const fetchStatus = async (silent = false) => {
        if (!silent) setRefreshing(true);
        try {
            const res = await fetch("/api/bot/status");
            if (res.ok) {
                const data = await res.json();
                setStatus(data);
                // Initialize config form if empty
                if (Object.keys(configForm).length === 0 && data.config) {
                    setConfigForm(data.config);
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

    const setPollSecondsFromParts = (value: number, unit: PollUnit) => {
        const safeValue = Math.max(1, Math.floor(Number.isFinite(value) ? value : 1));
        const multiplier = unit === "s" ? 1 : unit === "m" ? 60 : 3600;
        const seconds = safeValue * multiplier;
        setPollIntervalValue(safeValue);
        setPollIntervalUnit(unit);
        setConfigForm((prev) => ({ ...prev, poll_seconds: seconds }));
    };

    const handleRefresh = async () => {
        await Promise.all([fetchStatus(), fetchAccount()]);
    };

    const fmtUsd = (raw: string | number | undefined) => {
        const n = typeof raw === "number" ? raw : Number(raw ?? 0);
        const safe = Number.isFinite(n) ? n : 0;
        return safe.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
    };

    if (loading && !status) {
        return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading...</div>;
    }

    const isRunning = status?.status === "running" || status?.status === "starting";

    return (
        <div className="min-h-screen bg-black text-zinc-100 flex flex-col selection:bg-indigo-500/30">
            <AdminHeader
                activeMainTab="ai" // Highlight AI tab for now or add a new one?
                setActiveMainTab={() => { }} // No-op as this is a separate page
            />

            <main className="flex-1 w-full max-w-[1600px] mx-auto p-6 md:p-8 overflow-y-auto">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-black tracking-tighter text-white mb-2 flex items-center gap-3">
                            <Activity className="w-8 h-8 text-indigo-500" />
                            LIVE BOT CONTROL
                        </h1>
                        <div className="flex items-center gap-3">
                            <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
                            <span className="text-sm font-bold text-zinc-400 uppercase tracking-widest">
                                Status: <span className={isRunning ? "text-emerald-400" : "text-red-400"}>{status?.status || "UNKNOWN"}</span>
                            </span>
                            {status?.last_scan && (
                                <span className="text-xs text-zinc-600 ml-2">Last Scan: {status.last_scan}</span>
                            )}
                        </div>

                        <div className="mt-3 flex flex-wrap gap-3 text-xs">
                            <div className="px-3 py-2 rounded-xl bg-zinc-900/60 border border-zinc-800/60">
                                <div className="text-zinc-500 uppercase font-bold tracking-widest">Alpaca Cash</div>
                                <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.cash)}</div>
                            </div>
                            <div className="px-3 py-2 rounded-xl bg-zinc-900/60 border border-zinc-800/60">
                                <div className="text-zinc-500 uppercase font-bold tracking-widest">Buying Power</div>
                                <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.buying_power)}</div>
                            </div>
                            <div className="px-3 py-2 rounded-xl bg-zinc-900/60 border border-zinc-800/60">
                                <div className="text-zinc-500 uppercase font-bold tracking-widest">Portfolio</div>
                                <div className="text-white font-mono">{alpacaAccountLoading ? "…" : fmtUsd(alpacaAccount?.portfolio_value)}</div>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleRefresh}
                            className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 transition-colors"
                        >
                            <RefreshCw className={`w-5 h-5 text-zinc-400 ${refreshing ? "animate-spin" : ""}`} />
                        </button>

                        {!isRunning ? (
                            <button
                                onClick={handleStart}
                                className="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center gap-2 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)]"
                            >
                                <Play className="w-5 h-5 fill-current" />
                                START BOT
                            </button>
                        ) : (
                            <button
                                onClick={handleStop}
                                className="px-6 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold flex items-center gap-2 transition-all shadow-[0_0_20px_rgba(239,68,68,0.3)] hover:shadow-[0_0_30px_rgba(239,68,68,0.5)]"
                            >
                                <Square className="w-5 h-5 fill-current" />
                                STOP BOT
                            </button>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Configuration */}
                    <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl p-6 backdrop-blur-sm lg:col-span-1">
                        <div className="flex items-center gap-2 mb-6">
                            <Settings className="w-5 h-5 text-indigo-400" />
                            <h2 className="text-lg font-bold">Configuration</h2>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-zinc-500 uppercase">Coins (comma separated)</label>
                                <input
                                    type="text"
                                    value={Array.isArray(configForm.coins) ? configForm.coins.join(",") : configForm.coins || ""}
                                    onChange={(e) => setConfigForm({ ...configForm, coins: e.target.value.split(",") })}
                                    className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">King Threshold</label>
                                    <input
                                        type="number" step="0.01"
                                        value={configForm.king_threshold}
                                        onChange={(e) => setConfigForm({ ...configForm, king_threshold: parseFloat(e.target.value) })}
                                        className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Council Threshold</label>
                                    <input
                                        type="number" step="0.01"
                                        value={configForm.council_threshold}
                                        onChange={(e) => setConfigForm({ ...configForm, council_threshold: parseFloat(e.target.value) })}
                                        className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Max Notional ($)</label>
                                    <input
                                        type="number"
                                        value={configForm.max_notional_usd}
                                        onChange={(e) => setConfigForm({ ...configForm, max_notional_usd: parseFloat(e.target.value) })}
                                        className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">% Cash/Trade</label>
                                    <input
                                        type="number" step="0.01"
                                        value={configForm.pct_cash_per_trade}
                                        onChange={(e) => setConfigForm({ ...configForm, pct_cash_per_trade: parseFloat(e.target.value) })}
                                        className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Poll Interval</label>
                                    <div className="flex gap-2">
                                        <input
                                            type="number"
                                            min={1}
                                            value={pollIntervalValue}
                                            onChange={(e) => setPollSecondsFromParts(Number(e.target.value), pollIntervalUnit)}
                                            className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                        />
                                        <select
                                            value={pollIntervalUnit}
                                            onChange={(e) => {
                                                const nextUnit = e.target.value as PollUnit;
                                                const currentSeconds =
                                                    typeof configForm.poll_seconds === "number"
                                                        ? configForm.poll_seconds
                                                        : Number(configForm.poll_seconds);
                                                const multiplier = nextUnit === "s" ? 1 : nextUnit === "m" ? 60 : 3600;
                                                const nextValue =
                                                    Number.isFinite(currentSeconds) && currentSeconds > 0
                                                        ? Math.max(1, Math.round(currentSeconds / multiplier))
                                                        : pollIntervalValue;
                                                setPollSecondsFromParts(nextValue, nextUnit);
                                            }}
                                            className="bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                        >
                                            <option value="s">sec</option>
                                            <option value="m">min</option>
                                            <option value="h">hour</option>
                                        </select>
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Bars Limit</label>
                                    <input
                                        type="number"
                                        value={configForm.bars_limit}
                                        onChange={(e) => setConfigForm({ ...configForm, bars_limit: parseInt(e.target.value) })}
                                        className="w-full bg-black/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={handleUpdateConfig}
                                disabled={isRunning}
                                className={`w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${isRunning
                                        ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                                        : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                                    }`}
                            >
                                <Save className="w-4 h-4" />
                                {isRunning ? "STOP BOT TO UPDATE" : "SAVE CONFIGURATION"}
                            </button>
                        </div>
                    </div>

                    {/* Right Column: Logs & Trades */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Live Logs */}
                        <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl p-6 backdrop-blur-sm flex flex-col h-[500px]">
                            <div className="flex items-center gap-2 mb-4 shrink-0">
                                <Terminal className="w-5 h-5 text-emerald-400" />
                                <h2 className="text-lg font-bold">Live Logs</h2>
                            </div>

                            <div className="flex-1 bg-black/80 rounded-xl border border-zinc-800 p-4 overflow-y-auto font-mono text-xs text-zinc-300 space-y-1">
                                {status?.logs && status.logs.length > 0 ? (
                                    status.logs.map((log, i) => (
                                        <div key={i} className="break-all border-b border-zinc-900/50 pb-0.5 last:border-0">
                                            {log}
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-zinc-600 italic">No logs available...</div>
                                )}
                            </div>
                        </div>

                        {/* Recent Trades */}
                        <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl p-6 backdrop-blur-sm">
                            <div className="flex items-center gap-2 mb-4">
                                <Activity className="w-5 h-5 text-purple-400" />
                                <h2 className="text-lg font-bold">Recent Trades</h2>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-xs text-zinc-500 uppercase bg-black/20">
                                        <tr>
                                            <th className="px-4 py-3 rounded-l-lg">Time</th>
                                            <th className="px-4 py-3">Symbol</th>
                                            <th className="px-4 py-3">Action</th>
                                            <th className="px-4 py-3">Amount</th>
                                            <th className="px-4 py-3 rounded-r-lg">Order ID</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-zinc-800/50">
                                        {status?.trades && status.trades.length > 0 ? (
                                            status.trades.map((trade, i) => (
                                                <tr key={i} className="hover:bg-white/5 transition-colors">
                                                    <td className="px-4 py-3 font-mono text-zinc-400">
                                                        {new Date(trade.timestamp).toLocaleTimeString()}
                                                    </td>
                                                    <td className="px-4 py-3 font-bold text-white">{trade.symbol}</td>
                                                    <td className={`px-4 py-3 font-bold ${trade.action === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                                                        {trade.action}
                                                    </td>
                                                    <td className="px-4 py-3 text-zinc-300">${trade.amount.toFixed(2)}</td>
                                                    <td className="px-4 py-3 font-mono text-xs text-zinc-600">{trade.order_id}</td>
                                                </tr>
                                            ))
                                        ) : (
                                            <tr>
                                                <td colSpan={5} className="px-4 py-8 text-center text-zinc-600 italic">
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
            </main>
        </div>
    );
}
