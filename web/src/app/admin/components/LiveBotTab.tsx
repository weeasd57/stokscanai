"use client";

import { useState, useEffect } from "react";
import { Play, Square, Activity, Settings, Terminal, RefreshCw, Save, Coins, ShieldCheck, ShieldAlert, Plus, X, Search, Check, BarChart3, PieChart, ArrowUpRight, ArrowDownRight, Clock, Globe, Target, Trash2, History as HistoryIcon, Maximize2, ChevronDown, ChevronUp, Copy, CheckCheck, LayoutGrid, Cpu, Zap, BarChartHorizontal } from "lucide-react";
import { useRef } from "react";
import { toast } from "sonner";
import LiveCandleChart from "./LiveCandleChart";
// Radix UI Imports
import * as Dialog from "@radix-ui/react-dialog";
import * as Switch from "@radix-ui/react-switch";

// Types
interface BotConfig {
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
    use_quality_score?: boolean;
    min_quality_score?: number;
    use_auto_tune?: boolean;
    use_trailing: boolean;
    trail_be_pct: number;
    trail_lock_trigger_pct: number;
    trail_lock_pct: number;
    save_to_supabase: boolean;
    king_model_path: string;
    council_model_path: string;
    max_open_positions: number;
    name: string;
    execution_mode: "VIRTUAL" | "TELEGRAM" | "BOTH";
    trading_mode: "defensive" | "aggressive" | "hybrid";
}

interface BotListItem {
    id: string;
    name: string;
    status: string;
    mode: string;
    uptime?: string;
    current_activity?: string;
    active_positions_count?: number;
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
    open_positions: Array<{
        symbol: string;
        entry_price: number;
        current_price: number;
        target_price?: number;
        stop_price?: number;
        pl_pct: number;
        pl_usd: number;
        entry_time?: string;
        bars_held: number;
        trail_mode: string;
        amount: number;
    }>;
    trades?: Trade[];
}

interface Trade {
    timestamp: string;
    symbol: string;
    action: string;
    amount: number;
    price?: number;
    entry_price?: number;
    pnl?: number;
    order_id: string;
    status?: string;
}

interface BotStatus {
    status: "stopped" | "starting" | "running" | "stopping" | "error";
    config: BotConfig;
    last_scan: string | null;
    started_at: string | null;
    error: string | null;
    logs: string[];
    trades: Trade[];
    active_positions_count?: number;
    current_activity?: string;
    data_stream?: Record<string, {
        source: string;
        count: number;
        timestamp: string;
        status: string;
        has_volume?: boolean;
        error?: string;
    }>;
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
    // Multi-Bot State
    const [botList, setBotList] = useState<BotListItem[]>([]);
    const [selectedBotId, setSelectedBotId] = useState<string>("primary");
    const [createBotDialogOpen, setCreateBotDialogOpen] = useState(false);
    const [newBotName, setNewBotName] = useState("");
    const [isCreatingBot, setIsCreatingBot] = useState(false);
    const [renameBotDialogOpen, setRenameBotDialogOpen] = useState(false);
    const [renameBotName, setRenameBotName] = useState("");
    const [isRenamingBot, setIsRenamingBot] = useState(false);
    const [modelHubOpen, setModelHubOpen] = useState(true);

    // Command States
    const [isStarting, setIsStarting] = useState(false);
    const [isStopping, setIsStopping] = useState(false);

    // Asset Selection State
    const [assetTab, setAssetTab] = useState<"CRYPTO" | "STOCKS" | "GLOBAL">("CRYPTO");
    const [cryptoFilter, setCryptoFilter] = useState<"ALL" | "USD" | "USDT">("ALL");
    const [countries, setCountries] = useState<{ name: string, count: number }[]>([]);
    const [selectedCountry, setSelectedCountry] = useState<string>("USA");
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    const [pollIntervalValue, setPollIntervalValue] = useState(2);
    const [pollIntervalUnit, setPollIntervalUnit] = useState<PollUnit>("s");

    const [activeTab, setActiveTab] = useState<"control" | "performance">("control");
    const [performance, setPerformance] = useState<PerformanceData | null>(null);
    const [performanceLoading, setPerformanceLoading] = useState(false);
    const [uptime, setUptime] = useState("00:00:00");
    const [logFilter, setLogFilter] = useState<"ALL" | "ACCEPTED" | "REJECTED" | "ERROR">("ALL");
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Alpaca integration removed (virtual execution only).
    // Logs & Analytics State
    const [selectedChartSymbol, setSelectedChartSymbol] = useState<string | null>(null);
    const [thresholdStats, setThresholdStats] = useState<Record<string, number>>({});
    const [logsCollapsed, setLogsCollapsed] = useState(false);
    const [marketDataCollapsed, setMarketDataCollapsed] = useState(false);
    const [copyingLogs, setCopyingLogs] = useState(false);

    const handleCopyLogs = () => {
        if (!status?.logs) return;
        const logText = status.logs.join("\n");
        navigator.clipboard.writeText(logText).then(() => {
            setCopyingLogs(true);
            toast.success("Logs copied to clipboard");
            setTimeout(() => setCopyingLogs(false), 2000);
        }).catch(err => {
            toast.error("Failed to copy logs");
            console.error(err);
        });
    };

    const handleClearLogs = async () => {
        try {
            const res = await fetch(`/api/ai_bot/clear_logs?bot_id=${selectedBotId}`, { method: "POST" });
            if (!res.ok) throw new Error("Failed to clear logs");

            // Optimistically clear local state
            setStatus(prev => prev ? { ...prev, logs: [] } : null);
            toast.success("Logs cleared");
        } catch (error) {
            console.error(error);
            toast.error("Failed to clear logs");
        }
    };


    useEffect(() => {
        fetchBotList();
        fetchStatus();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedBotId]);

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

    const [coinSource, setCoinSource] = useState<"database" | "alpaca">("alpaca");
    const [coinLimit, setCoinLimit] = useState(100);
    const autoSelectRef = useRef(false);

    const fetchModels = async () => {
        try {
            const res = await fetch("/api/ai_bot/models");
            if (res.ok) {
                const data = await res.json();
                setAvailableModels(data);
            }
        } catch (e) {
            console.error("Failed to fetch models", e);
        }
    };

    const fetchCountries = async () => {
        try {
            const res = await fetch("/api/ai_bot/countries");
            if (res.ok) {
                const data = await res.json();
                setCountries(data);
                if (data.length > 0 && !data.some((c: { name: string }) => c.name === selectedCountry)) {
                    setSelectedCountry(data[0].name);
                }
            }
        } catch (e) {
            console.error("Failed to fetch countries", e);
        }
    };

    const fetchCoins = async () => {
        try {
            let source = "";
            let url = "";

            if (assetTab === "CRYPTO") {
                source = coinSource === "alpaca" ? "alpaca" : "database";
                url = `/api/ai_bot/available_coins?source=${source}&limit=${coinLimit}&pair_type=${cryptoFilter}`;
            } else if (assetTab === "STOCKS") {
                source = "alpaca_stocks";
                url = `/api/ai_bot/available_coins?source=${source}&limit=${coinLimit}`;
            } else if (assetTab === "GLOBAL") {
                source = "global";
                url = `/api/ai_bot/available_coins?source=${source}&country=${selectedCountry}&limit=${coinLimit}`;
            }

            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                if (Array.isArray(data)) {
                    setAvailableCoins(data);
                    if (autoSelectRef.current) {
                        // Apply the cryptoFilter so only matching pairs are auto-selected
                        const filtered = assetTab === "CRYPTO" && cryptoFilter !== "ALL"
                            ? data.filter((c: string) => c.endsWith(`/${cryptoFilter}`))
                            : data;
                        setConfigForm(prev => ({ ...prev, coins: filtered }));
                        autoSelectRef.current = false;
                    }
                }
            }
        } catch (e) {
            console.error("Failed to fetch coins", e);
        }
    };

    useEffect(() => {
        if (assetTab === "GLOBAL") {
            setConfigForm(prev => ({ ...prev, data_source: "tvdata" }));
        } else if (assetTab === "CRYPTO" && configForm.data_source === "tvdata") {
            setConfigForm(prev => ({ ...prev, data_source: "binance" }));
        } else if (assetTab === "STOCKS" && configForm.data_source === "tvdata") {
            setConfigForm(prev => ({ ...prev, data_source: "yfinance" }));
        }
    }, [assetTab, configForm.data_source]);

    useEffect(() => {
        fetchModels();
        if (assetTab === "GLOBAL") {
            fetchCountries();
        }
    }, [assetTab]);

    useEffect(() => {
        fetchCoins();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [coinSource, coinLimit, assetTab, selectedCountry, cryptoFilter]);

    const fetchBotList = async () => {
        try {
            const res = await fetch("/api/ai_bot/list");
            if (res.ok) {
                const data = await res.json();
                setBotList(data.bots || []);
            }
        } catch (e) {
            console.error("Failed to fetch bot list", e);
        }
    };

    const handleCreateBot = async () => {
        if (!newBotName.trim()) return;
        setIsCreatingBot(true);
        try {
            const bot_id = newBotName.trim().toLowerCase().replace(/\s+/g, '_');
            const res = await fetch("/api/ai_bot/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    bot_id,
                    name: newBotName.trim(),
                })
            });
            if (res.ok) {
                toast.success(`Bot "${newBotName}" created`);
                setNewBotName("");
                setCreateBotDialogOpen(false);
                await fetchBotList();
                setSelectedBotId(bot_id);
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to create bot");
            }
        } catch (e) {
            toast.error("Error creating bot");
        } finally {
            setIsCreatingBot(false);
        }
    };

    const handleRenameBot = async () => {
        if (!renameBotName.trim()) return;
        setIsRenamingBot(true);
        try {
            const res = await fetch(`/api/ai_bot/config?bot_id=${selectedBotId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: renameBotName.trim() })
            });
            if (res.ok) {
                toast.success(`Bot renamed to "${renameBotName}"`);
                setRenameBotName("");
                setRenameBotDialogOpen(false);
                await fetchBotList();
                // Optionally update local state too if we want immediate feedback without full fetch
                setStatus(prev => prev ? { ...prev, config: { ...prev.config, name: renameBotName.trim() } } : prev);
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to rename bot");
            }
        } catch (e) {
            toast.error("Error renaming bot");
        } finally {
            setIsRenamingBot(false);
        }
    };

    const handleDeleteBot = async (id: string, name: string) => {
        if (id === "primary") {
            toast.error("Cannot delete primary bot");
            return;
        }
        const running = status?.status === "running" && selectedBotId === id;
        const prompt = running
            ? `Bot "${name}" is running.\n\nStop and delete it?`
            : `Are you sure you want to delete bot "${name}"?`;
        if (!confirm(prompt)) return;

        try {
            if (running) {
                await fetch(`/api/ai_bot/stop?bot_id=${id}`, { method: "POST" });
            }
            const res = await fetch(`/api/ai_bot/delete/${id}`, { method: "DELETE" });
            if (res.ok) {
                toast.success("Bot deleted");
                if (selectedBotId === id) setSelectedBotId("primary");
                fetchBotList();
            } else {
                toast.error("Failed to delete bot");
            }
        } catch (e) {
            toast.error("Error deleting bot");
        }
    };

    // Auto-save configuration with debounce
    useEffect(() => {
        if (!configForm || status?.status === 'running' || status?.status === 'starting') return;

        const timeout = setTimeout(() => {
            handleUpdateConfig(true); // pass silent=true to suppress toast
        }, 1000); // Debounce 1s

        return () => clearTimeout(timeout);
    }, [configForm, status?.status]);

    const pollSecondsRaw = configForm.poll_seconds;
    const pollSeconds = typeof pollSecondsRaw === "number" ? pollSecondsRaw : Number(pollSecondsRaw);
    const pollMs = Number.isFinite(pollSeconds) && pollSeconds > 0 ? pollSeconds * 1000 : 2000;

    const normalizePosKey = (s: string) => (s || "").toUpperCase().replace("/", "").replace("-", "").replace("_", "");

    const configuredBotCoins = ((configForm.coins && configForm.coins.length > 0)
        ? configForm.coins
        : (status?.config?.coins || [])) as string[];
    const botCoinNorms = new Set(configuredBotCoins.map(normalizePosKey));
    const isBotSymbol = (symbol: string) => {
        const n = normalizePosKey(symbol);
        return !n || botCoinNorms.size === 0 || botCoinNorms.has(n);
    };

    const scopedOpenPositions = (performance?.open_positions || [])
        .filter(p => isBotSymbol(p.symbol))
        .map(p => ({
            symbol: p.symbol,
            unrealized_pl: String(p.pl_usd ?? 0),
        }));

    const positionsBySymbol = new Map<string, { symbol: string; unrealized_pl: string }>(
        scopedOpenPositions.map((p) => [normalizePosKey(p.symbol), p])
    );

    const toNum = (v: any): number | null => {
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
    };

    const computeSellPnl = (trade: Trade): number | null => {
        const direct = toNum(trade.pnl);
        if (direct !== null && Math.abs(direct) > 1e-12) return direct;

        const price = toNum(trade.price);
        const entry = toNum(trade.entry_price);
        const qty = toNum(trade.amount);
        if (price !== null && entry !== null && qty !== null && qty > 0) {
            return (price - entry) * qty;
        }
        return direct;
    };

    // Auto-refresh when running (Fixed 3s interval for UI responsiveness)
    useEffect(() => {
        const interval = setInterval(() => {
            if (status?.status === "running") {
                fetchStatus(true);
                fetchPerformance(true);
            }
        }, 8000); // Increased to 8s for lower load
        return () => clearInterval(interval);
    }, [status?.status, selectedBotId]); // Refresh when botId changes as well

    // Initial load and refresh on bot switch
    useEffect(() => {
        fetchStatus(false);
        fetchPerformance(false);
    }, [selectedBotId]);

    const fetchStatus = async (silent = false) => {
        if (!silent) setRefreshing(true);
        try {
            const res = await fetch(`/api/ai_bot/status?bot_id=${selectedBotId}`);
            if (res.ok) {
                const data = await res.json();

                // Keep more history (1000 items)
                if (data.logs && data.logs.length > 1000) {
                    data.logs = data.logs.slice(-1000);
                }

                // Compute Threshold Stats from logs
                // Log format example: "[2024-02-09 10:00:00] [INFO] [AAPL] Score: 0.85 (Threshold 0.80) -> ACCEPTED"
                // or just extracting any number after "Score:"
                const stats: Record<string, number> = {};
                for (let i = 4; i <= 10; i++) {
                    stats[(i / 10).toFixed(1)] = 0;
                }

                if (data.logs && Array.isArray(data.logs)) {
                    data.logs.forEach((l: string) => {
                        const match = l.match(/(?:KING|COUNCIL)(?:=|\s*pass\s*\(|:\s*)(\d+(\.\d+)?)/i);
                        if (match && match[1]) {
                            const score = parseFloat(match[1]);
                            // Increment all buckets that this score clears
                            for (let t = 4; t <= 10; t++) {
                                const thresh = t / 10;
                                if (score >= thresh) {
                                    stats[thresh.toFixed(1)] = (stats[thresh.toFixed(1)] || 0) + 1;
                                }
                            }
                        }
                    });
                }
                setThresholdStats(stats);

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
            const res = await fetch(`/api/ai_bot/start?bot_id=${selectedBotId}`, { method: "POST" });
            if (res.ok) {
                toast.success("Bot started command sent");
                fetchStatus();
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to start bot");
            }
        } catch (error) {
            toast.error("Failed to start bot");
        }
    };

    const handleStop = async () => {
        setIsStopping(true);
        try {
            const res = await fetch(`/api/ai_bot/stop?bot_id=${selectedBotId}`, { method: "POST" });
            if (res.ok) {
                toast.success("Bot stop command sent");
                fetchStatus();
            } else {
                toast.error("Failed to stop bot");
            }
        } catch (error) {
            toast.error("Failed to stop bot");
        } finally {
            setIsStopping(false);
        }
    };

    const handleTestNotification = async (type: string) => {
        try {
            const res = await fetch(`/api/ai_bot/test_notification?notify_type=${type}&bot_id=${selectedBotId}`, { method: "POST" });
            if (res.ok) {
                toast.success(`Test ${type} notification sent`);
            } else {
                const err = await res.json();
                toast.error(err.detail || "Failed to send test notification");
            }
        } catch (error) {
            toast.error("Network error");
        }
    };


    const handleUpdateConfig = async (silent: boolean = false) => {
        try {
            const res = await fetch(`/api/ai_bot/config?bot_id=${selectedBotId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(configForm)
            });
            if (res.ok) {
                const data = await res.json();
                if (!silent) toast.success("Configuration updated");
                setStatus((prev) => prev ? { ...prev, config: data.config } : prev);
            } else {
                const err = await res.json();
                if (!silent) toast.error(err.detail || "Failed to update config");
            }
        } catch (error) {
            if (!silent) toast.error("Failed to update config");
        }
    };

    const toggleCoin = (coin: string) => {
        const current = configForm.coins || [];
        if (current.includes(coin)) {
            setConfigForm({ ...configForm, coins: current.filter(c => c !== coin) });
        } else {
            setConfigForm({ ...configForm, coins: [...current, coin] });
        }
    };

    // Alpaca watchlist sync removed (virtual execution only).

    const fetchPerformance = async (silent = false) => {
        if (!silent) setPerformanceLoading(true);
        console.log(`[Dashboard] Fetching performance for ${selectedBotId}`);
        try {
            const res = await fetch(`/api/ai_bot/performance?bot_id=${selectedBotId}`);
            if (res.ok) {
                const data = await res.json();
                console.log(`[Dashboard] Performance data received: ${data.trades?.length || 0} trades found`);
                setPerformance(data);
            } else {
                console.error(`[Dashboard] Performance fetch failed with status ${res.status}`);
            }
        } catch (error) {
            console.error("[Dashboard] Failed to fetch performance", error);
        } finally {
            if (!silent) setPerformanceLoading(false);
        }
    };

    useEffect(() => {
        fetchPerformance(true); // Fetch silently on tab change to keep dashboard cards fresh
    }, [activeTab]);

    if (loading && !status) {
        return <div className="flex items-center justify-center h-96 text-indigo-400 animate-pulse font-mono tracking-widest">INITIALIZING CONNECTION...</div>;
    }

    const isRunning = status?.status === "running" || status?.status === "starting";
    const useCouncil = configForm.use_council ?? true;

    return (
        <div className="space-y-6 max-w-[1600px] mx-auto animate-in fade-in zoom-in-95 duration-500 pb-20">
            {/* NEW REDESIGNED COMMAND CENTER */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 items-stretch">
                {/* Main Terminal Identity Card */}
                <div className="xl:col-span-1 bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-white/10 rounded-3xl p-6 backdrop-blur-xl flex flex-col justify-between group relative overflow-hidden shadow-2xl">
                    <div className="absolute top-0 right-0 p-8 bg-indigo-500/10 rounded-full blur-3xl group-hover:bg-indigo-500/20 transition-all duration-700" />

                    <div className="relative z-10">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="bg-indigo-500 p-3 rounded-2xl shadow-lg shadow-indigo-500/40">
                                <Cpu className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl font-black text-white tracking-tighter leading-none">
                                    AI TRADING<br />TERMINAL
                                </h1>
                                <span className="text-[10px] font-bold text-indigo-300 tracking-widest uppercase">Precision Engine V2</span>
                            </div>
                        </div>

                        <div className="flex items-center gap-2 mb-6">
                            <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[9px] font-bold text-zinc-400 uppercase tracking-widest">
                                Instance: {status?.config?.name || "PRIMARY"}
                            </span>
                            <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-emerald-500 animate-pulse shadow-[0_0_12px_rgba(16,185,129,0.5)]" : "bg-zinc-600"}`} />
                        </div>
                    </div>

                    <div className="relative z-10 flex gap-2 pt-4">
                        <button
                            onClick={() => fetchStatus()}
                            disabled={refreshing}
                            className="p-3 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all group/refresh"
                            title="Refresh Status"
                        >
                            <RefreshCw className={`w-4 h-4 text-zinc-400 group-hover/refresh:text-white transition-colors ${refreshing ? "animate-spin" : ""}`} />
                        </button>
                        <Dialog.Root open={renameBotDialogOpen} onOpenChange={(open: boolean) => {
                            if (open && status?.config?.name) setRenameBotName(status.config.name);
                            setRenameBotDialogOpen(open);
                        }}>
                            <Dialog.Trigger asChild>
                                <button className="p-3 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all group/settings" title="Bot Configuration">
                                    <Settings className="w-4 h-4 text-zinc-400 group-hover/settings:text-white" />
                                </button>
                            </Dialog.Trigger>
                            <Dialog.Portal>
                                <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300 z-50" />
                                <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md bg-zinc-950 border border-zinc-800 p-8 rounded-3xl shadow-2xl animate-in zoom-in-95 duration-300 z-50">
                                    <div className="space-y-6">
                                        <div className="space-y-2">
                                            <h2 className="text-xl font-black text-white">Rename Bot</h2>
                                            <p className="text-sm text-zinc-500 font-medium">Update the display name for this instance.</p>
                                        </div>
                                        <div className="space-y-4">
                                            <div className="space-y-2">
                                                <label className="text-xs font-bold text-zinc-500 uppercase">New Name</label>
                                                <input
                                                    type="text"
                                                    value={renameBotName}
                                                    onChange={(e) => setRenameBotName(e.target.value)}
                                                    placeholder="e.g., Scalper Pro"
                                                    className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-bold focus:outline-none focus:border-indigo-500 transition-all"
                                                    onKeyDown={(e) => e.key === 'Enter' && handleRenameBot()}
                                                />
                                            </div>
                                            <button
                                                onClick={handleRenameBot}
                                                disabled={isRenamingBot || !renameBotName.trim()}
                                                className="w-full bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-black py-4 rounded-xl shadow-lg shadow-indigo-500/20 transition-all"
                                            >
                                                {isRenamingBot ? <RefreshCw className="w-5 h-5 animate-spin mx-auto" /> : "UPDATE NAME"}
                                            </button>
                                        </div>
                                    </div>
                                </Dialog.Content>
                            </Dialog.Portal>
                        </Dialog.Root>

                        <div className="flex bg-black/60 p-1 rounded-2xl border border-white/5 shadow-inner flex-1">
                            <button
                                onClick={handleStart}
                                disabled={isRunning || isStarting}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all ${isRunning || isStarting
                                    ? "text-emerald-500/30 cursor-not-allowed"
                                    : "text-emerald-400 hover:bg-emerald-500/10 hover:text-emerald-300"
                                    }`}
                            >
                                <Play className={`w-3.5 h-3.5 ${isStarting ? "animate-pulse" : ""}`} />
                                {isStarting ? "BUSY" : "START"}
                            </button>
                            <button
                                onClick={handleStop}
                                disabled={!isRunning || isStopping}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all ${!isRunning || isStopping
                                    ? "text-red-500/30 cursor-not-allowed"
                                    : "text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                    }`}
                            >
                                <Square className={`w-3.5 h-3.5 ${isStopping ? "animate-pulse" : ""}`} />
                                {isStopping ? "HALTING" : "STOP"}
                            </button>
                        </div>

                        {/* Test Notification Section */}
                        <div className="flex bg-black/60 p-1 rounded-2xl border border-white/5 shadow-inner ml-2">
                            <button
                                onClick={() => handleTestNotification("buy")}
                                className="px-3 py-2 rounded-xl text-[10px] font-black text-emerald-500 hover:bg-emerald-500/10 transition-all"
                                title="Test Buy Notification"
                            >
                                <ArrowUpRight className="w-3.5 h-3.5" />
                            </button>
                            <button
                                onClick={() => handleTestNotification("sell")}
                                className="px-3 py-2 rounded-xl text-[10px] font-black text-rose-500 hover:bg-rose-500/10 transition-all"
                                title="Test Sell Notification"
                            >
                                <ArrowDownRight className="w-3.5 h-3.5" />
                            </button>
                            <button
                                onClick={() => handleTestNotification("signal")}
                                className="px-3 py-2 rounded-xl text-[10px] font-black text-indigo-500 hover:bg-indigo-500/10 transition-all"
                                title="Test Signal Notification"
                            >
                                <Zap className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    </div>

                    <div className="mt-4 pt-4 border-t border-white/5 space-y-2">
                        <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest block mb-2">Notification Testing (Legacy)</span>
                        <div className="flex bg-black/60 p-1 rounded-2xl border border-white/5 shadow-inner">
                            <button
                                onClick={() => handleTestNotification("buy")}
                                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black text-emerald-500 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all group/test"
                            >
                                <ArrowUpRight className="w-3.5 h-3.5 group-hover/test:scale-110 transition-transform" />
                                BUY
                            </button>
                            <button
                                onClick={() => handleTestNotification("sell")}
                                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black text-rose-500 hover:bg-rose-500/10 hover:text-rose-400 transition-all group/test"
                            >
                                <ArrowDownRight className="w-3.5 h-3.5 group-hover/test:scale-110 transition-transform" />
                                SELL
                            </button>
                            <button
                                onClick={() => handleTestNotification("signal")}
                                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black text-indigo-500 hover:bg-indigo-500/10 hover:text-indigo-400 transition-all group/test"
                            >
                                <Zap className="w-3.5 h-3.5 group-hover/test:scale-110 transition-transform" />
                                SIGNAL
                            </button>
                        </div>
                    </div>
                </div>

                {/* Status Cards Grid */}
                <div className="xl:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* BOT HEALTH CARD */}
                    <div className="bg-black/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl relative overflow-hidden group/card shadow-xl flex flex-col justify-between">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500/50 to-transparent" />
                        <div className="flex items-center justify-between mb-8">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                                    <Zap className="w-4 h-4 text-emerald-400" />
                                </div>
                                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Bot Health</span>
                            </div>
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${isRunning ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-zinc-800 text-zinc-500"}`}>
                                {status?.status?.toUpperCase() || "OFFLINE"}
                            </span>
                        </div>
                        <div className="space-y-1">
                            <div className="text-3xl font-black text-white font-mono tracking-tighter">
                                {isRunning ? uptime : "00:00:00"}
                            </div>
                            <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Continuous Uptime</div>
                        </div>
                        <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-[10px]">
                            <span className="text-zinc-600 font-bold">SYSTEM THREADS</span>
                            <span className="text-emerald-400/80 font-mono">ACTIVE</span>
                        </div>
                    </div>

                    {/* ACTIVE EXPOSURE CARD */}
                    <div className="bg-black/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl relative overflow-hidden group/card shadow-xl flex flex-col justify-between">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500/50 to-transparent" />
                        <div className="flex items-center justify-between mb-8">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                                    <BarChartHorizontal className="w-4 h-4 text-indigo-400" />
                                </div>
                                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Active Exposure</span>
                            </div>
                            <span className="text-[10px] font-bold text-zinc-400 bg-zinc-800 px-2 py-0.5 rounded-full">
                                LIMIT: {status?.config?.max_open_positions || 3}
                            </span>
                        </div>
                        <div className="space-y-1">
                            <div className="text-3xl font-black text-white font-mono tracking-tighter flex items-baseline gap-2">
                                {status?.active_positions_count || 0}
                                <span className="text-sm text-zinc-600">POSITIONS</span>
                            </div>
                            <div className="h-1.5 w-full bg-zinc-900 rounded-full mt-2 overflow-hidden border border-white/5">
                                <div
                                    className="h-full bg-indigo-500 transition-all duration-1000 shadow-[0_0_10px_rgba(99,102,241,0.5)]"
                                    style={{ width: `${Math.min(((status?.active_positions_count || 0) / (status?.config?.max_open_positions || 1)) * 100, 100)}%` }}
                                />
                            </div>
                        </div>
                        <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-[10px]">
                            <span className="text-zinc-600 font-bold">MODE</span>
                            <span className="text-indigo-400/80 font-bold uppercase tracking-widest">{configForm.trading_mode || "HYBRID"}</span>
                        </div>
                    </div>

                    {/* SCANNER ENGINE CARD */}
                    <div className="bg-black/40 border border-white/5 rounded-3xl p-6 backdrop-blur-xl relative overflow-hidden group/card shadow-xl flex flex-col justify-between">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500/50 to-transparent" />
                        <div className="flex items-center justify-between mb-8">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20">
                                    <Terminal className="w-4 h-4 text-purple-400" />
                                </div>
                                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Scanner Engine</span>
                            </div>
                            <div className="flex items-center gap-1.5 p-1 bg-zinc-900 rounded-lg">
                                <div className={`w-1.5 h-1.5 rounded-full ${isRunning ? "bg-purple-500 animate-pulse" : "bg-zinc-700"}`} />
                            </div>
                        </div>
                        <div className="space-y-1 overflow-hidden">
                            <div className="text-xs font-black text-white truncate uppercase tracking-tight">
                                {status?.current_activity || "IDLE ENGINE"}
                            </div>
                            <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-2 flex items-center gap-2">
                                <Clock className="w-3 h-3" />
                                Next Scan: ~{status?.config?.poll_seconds || 0}s
                            </div>
                        </div>
                        <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-[10px]">
                            <span className="text-zinc-600 font-bold">SOURCE</span>
                            <span className="text-purple-400/80 font-bold truncate max-w-[100px] text-right uppercase">{status?.config?.data_source || "ALPACA"}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* BOT SELECTOR TABS */}
            <div className="flex items-center gap-2 p-1 bg-black/40 border border-white/5 rounded-3xl w-full backdrop-blur-sm overflow-x-auto no-scrollbar">
                {botList.map(bot => {
                    const isActive = selectedBotId === bot.id;
                    return (
                        <button
                            key={bot.id}
                            onClick={() => setSelectedBotId(bot.id)}
                            className={`flex-none px-6 py-2.5 rounded-[22px] text-xs font-black transition-all flex items-center gap-2 border ${isActive
                                ? "bg-white text-black border-white shadow-xl shadow-white/5"
                                : "text-zinc-500 border-transparent hover:text-zinc-300 hover:bg-white/5"
                                }`}
                        >
                            <LayoutGrid className={`w-3.5 h-3.5 ${isActive ? "text-indigo-600 font-black" : "text-zinc-600"}`} />
                            {bot.name.toUpperCase()}
                            {isActive && (
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse ml-1" />
                            )}
                        </button>
                    )
                })}
                {selectedBotId !== "primary" && (
                    <button
                        onClick={() => handleDeleteBot(selectedBotId, (configForm.name || status?.config?.name || "Bot"))}
                        className="flex-none p-2.5 rounded-[22px] bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all"
                        title="Delete bot"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                )}
                <Dialog.Root open={createBotDialogOpen} onOpenChange={setCreateBotDialogOpen}>
                    <Dialog.Trigger asChild>
                        <button className="flex-none p-2.5 rounded-[22px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all ml-auto mr-1">
                            <Plus className="w-4 h-4" />
                        </button>
                    </Dialog.Trigger>
                    <Dialog.Portal>
                        <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300 z-50" />
                        <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md bg-zinc-950 border border-zinc-800 p-8 rounded-3xl shadow-2xl animate-in zoom-in-95 duration-300 z-50">
                            <div className="space-y-6">
                                <div className="space-y-2">
                                    <h2 className="text-xl font-black text-white">Create New Bot</h2>
                                    <p className="text-sm text-zinc-500 font-medium">Add a new specialized trading instance.</p>
                                </div>
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-bold text-zinc-500 uppercase">Bot Name</label>
                                        <input
                                            type="text"
                                            value={newBotName}
                                            onChange={(e) => setNewBotName(e.target.value)}
                                            placeholder="Trade ID (Optional)"
                                            className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-bold focus:outline-none focus:border-indigo-500 transition-all"
                                            onKeyDown={(e) => e.key === 'Enter' && handleCreateBot()}
                                        />
                                    </div>
                                    <button
                                        onClick={handleCreateBot}
                                        disabled={isCreatingBot || !newBotName.trim()}
                                        className="w-full bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-black py-4 rounded-xl shadow-lg shadow-emerald-500/20 transition-all"
                                    >
                                        {isCreatingBot ? <RefreshCw className="w-5 h-5 animate-spin mx-auto" /> : "INITIALIZE BOT"}
                                    </button>
                                </div>
                            </div>
                        </Dialog.Content>
                    </Dialog.Portal>
                </Dialog.Root>
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

            {
                activeTab === 'control' ? (
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
                                {/* Asset Selection Panel */}
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase">Execution Mode</label>
                                    <select
                                        value={configForm.execution_mode || "BOTH"}
                                        onChange={(e) => setConfigForm({ ...configForm, execution_mode: e.target.value as any })}
                                        className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-indigo-500/50 transition-all text-white border-white/10"
                                    >
                                        <option value="BOTH" className="bg-zinc-950 text-white">Together (Virtual + Telegram)</option>
                                        <option value="VIRTUAL" className="bg-zinc-950 text-white">Virtual Only (Simulated)</option>
                                        <option value="TELEGRAM" className="bg-zinc-950 text-white">Telegram Only (Signal Only)</option>
                                    </select>
                                </div>

                                {/* Trading Mode Selector */}
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2">
                                        <ShieldCheck className="w-3.5 h-3.5" /> Trading Mode
                                    </label>
                                    <div className="grid grid-cols-3 gap-2">
                                        {([
                                            {
                                                value: "defensive", emoji: "🛡️", label: "Defensive",
                                                active: "bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-lg shadow-emerald-500/10"
                                            },
                                            {
                                                value: "aggressive", emoji: "⚔️", label: "Aggressive",
                                                active: "bg-red-500/20 border-red-500 text-red-400 shadow-lg shadow-red-500/10"
                                            },
                                            {
                                                value: "hybrid", emoji: "🔄", label: "Hybrid",
                                                active: "bg-indigo-500/20 border-indigo-500 text-indigo-400 shadow-lg shadow-indigo-500/10"
                                            },
                                        ] as const).map((m) => {
                                            const isActive = (configForm.trading_mode || "hybrid") === m.value;
                                            return (
                                                <button
                                                    key={m.value}
                                                    onClick={() => setConfigForm({ ...configForm, trading_mode: m.value as any })}
                                                    className={`flex flex-col items-center gap-1 p-3 rounded-xl border text-center transition-all ${isActive
                                                        ? m.active
                                                        : "bg-black/40 border-white/5 text-zinc-500 hover:border-white/10 hover:text-zinc-300"
                                                        }`}
                                                >
                                                    <span className="text-lg">{m.emoji}</span>
                                                    <span className="text-[10px] font-black uppercase tracking-wider">{m.label}</span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                    <p className="text-[9px] text-zinc-600 leading-tight">
                                        {(configForm.trading_mode || "hybrid") === "defensive" && "🛡️ Capital preservation: strict filters, higher thresholds, smaller positions"}
                                        {(configForm.trading_mode || "hybrid") === "aggressive" && "⚔️ Early entry: relaxed filters, lower thresholds, trades in BEAR markets"}
                                        {(configForm.trading_mode || "hybrid") === "hybrid" && "🔄 Regime-based: adapts automatically using current config values"}
                                    </p>
                                </div>

                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2">
                                            <Target className="w-3.5 h-3.5" /> Target Assets
                                        </label>
                                        <div className="flex bg-black/40 rounded-lg p-0.5 border border-white/5">
                                            <button
                                                onClick={() => setAssetTab("CRYPTO")}
                                                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${assetTab === "CRYPTO" ? "bg-indigo-500 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"}`}
                                            >
                                                CRYPTO
                                            </button>
                                            <button
                                                onClick={() => setAssetTab("STOCKS")}
                                                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${assetTab === "STOCKS" ? "bg-indigo-500 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"}`}
                                            >
                                                US STOCKS
                                            </button>
                                            <button
                                                onClick={() => setAssetTab("GLOBAL")}
                                                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${assetTab === "GLOBAL" ? "bg-indigo-500 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"}`}
                                            >
                                                GLOBAL
                                            </button>
                                        </div>
                                    </div>

                                    {/* Asset Filters */}


                                    {assetTab === "STOCKS" && (
                                        <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-xl text-xs text-yellow-200 flex items-center gap-2">
                                            <Globe className="w-4 h-4" />
                                            <span>US Stock Market (Alpaca)</span>
                                        </div>
                                    )}

                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setConfigForm({ ...configForm, coins: [] })}
                                            disabled={isRunning || (configForm.coins || []).length === 0}
                                            className="text-[10px] font-bold text-red-400 hover:text-red-300 disabled:opacity-30 transition-colors"
                                        >
                                            CLEAR ALL
                                        </button>
                                        {assetTab === "GLOBAL" && (
                                            <button
                                                onClick={() => {
                                                    if (availableCoins.length > 0) {
                                                        setConfigForm({ ...configForm, coins: availableCoins });
                                                        toast.success(`Added all ${availableCoins.length} symbols for ${selectedCountry}`);
                                                    }
                                                }}
                                                disabled={isRunning || availableCoins.length === 0}
                                                className="text-[10px] font-black bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-1 rounded hover:bg-emerald-500/20 transition-all disabled:opacity-50"
                                            >
                                                SYNC ALL {selectedCountry}
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* Smart Add Interface */}
                                <div className="bg-black/40 border border-white/5 rounded-xl p-3 space-y-3">
                                    {/* Source & Filter Controls */}
                                    <div className="flex flex-wrap items-center gap-2">
                                        {assetTab === "CRYPTO" ? (
                                            <div className="flex bg-black/40 rounded-lg p-0.5 border border-white/5">
                                                <button
                                                    onClick={() => setCoinSource("database")}
                                                    className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${coinSource === "database" ? "bg-indigo-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
                                                >
                                                    MY ASSETS
                                                </button>
                                                <button
                                                    onClick={() => setCoinSource("alpaca")}
                                                    className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${coinSource === "alpaca" ? "bg-indigo-600 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
                                                >
                                                    ALPACA
                                                </button>
                                            </div>
                                        ) : null}

                                        {assetTab === "CRYPTO" && (
                                            <select
                                                value={cryptoFilter}
                                                onChange={(e) => setCryptoFilter(e.target.value as any)}
                                                className="bg-zinc-900 border border-zinc-800 rounded-lg px-2 py-1 text-[10px] font-bold text-zinc-400 focus:outline-none focus:border-indigo-500 transition-colors"
                                            >
                                                <option value="ALL">All Pairs</option>
                                                <option value="USD">USD Pairs</option>
                                                <option value="USDT">USDT Pairs</option>
                                            </select>
                                        )}

                                        {assetTab === "GLOBAL" && (
                                            <select
                                                value={selectedCountry}
                                                onChange={(e) => setSelectedCountry(e.target.value)}
                                                className="bg-zinc-900 border border-zinc-800 rounded-lg px-2 py-1 text-[10px] font-bold text-zinc-400 focus:outline-none focus:border-indigo-500 transition-colors"
                                            >
                                                {countries.map(c => (
                                                    <option key={c.name} value={c.name}>{c.name} ({c.count})</option>
                                                ))}
                                            </select>
                                        )}

                                        {coinSource === "alpaca" && (
                                            <div className="flex items-center gap-1 overflow-x-auto custom-scrollbar pb-1 max-w-full">
                                                {[10, 50, 100, 0].map(lim => {
                                                    const countryObj = countries.find(c => c.name === selectedCountry);
                                                    const totalCount = countryObj?.count || 0;
                                                    const isHuge = assetTab === "GLOBAL" && lim === 0 && totalCount > 1000;

                                                    return (
                                                        <button
                                                            key={lim}
                                                            onClick={() => {
                                                                if (isHuge) {
                                                                    toast.error(`Too many symbols (${totalCount}). Please use search or Top limits.`);
                                                                    return;
                                                                }
                                                                autoSelectRef.current = true;
                                                                setCoinLimit(lim);
                                                            }}
                                                            className={`px-2 py-1 text-[9px] font-bold rounded-md border whitespace-nowrap transition-all ${coinLimit === lim
                                                                ? "bg-emerald-500/20 border-emerald-500 text-emerald-400"
                                                                : "bg-zinc-800 border-zinc-800 text-zinc-500 hover:border-zinc-600"
                                                                }`}
                                                        >
                                                            {lim === 0 ? "ALL" : `TOP ${lim}`}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>

                                    {/* Search & Bulk Add */}
                                    <div className="relative group">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 group-focus-within:text-indigo-400 transition-colors" />
                                        <input
                                            type="text"
                                            placeholder={`Search to add from ${availableCoins.length} available...`}
                                            value={coinSearch}
                                            onChange={(e) => setCoinSearch(e.target.value)}
                                            className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-9 pr-32 py-2.5 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-all text-white placeholder:text-zinc-600"
                                        />
                                        <div className="absolute right-1 top-1/2 -translate-y-1/2">
                                            {coinSearch && (
                                                <button
                                                    onClick={() => {
                                                        const filtered = availableCoins.filter(c => {
                                                            const matchesSearch = c.toLowerCase().includes(coinSearch.toLowerCase());
                                                            if (assetTab === "CRYPTO") {
                                                                if (!c.includes("/")) return false;
                                                                if (cryptoFilter === "USD" && !c.endsWith("/USD")) return false;
                                                                if (cryptoFilter === "USDT" && !c.endsWith("/USDT")) return false;
                                                            } else if (assetTab === "STOCKS") {
                                                                if (c.includes("/")) return false;
                                                            } else if (assetTab === "GLOBAL") {
                                                                // Assuming global assets might have a country prefix or suffix, or are just symbols
                                                                // For now, no specific filtering logic beyond search for global, as `availableCoins` should already be filtered by country
                                                            }
                                                            return matchesSearch;
                                                        });
                                                        const current = new Set(configForm.coins || []);
                                                        const toAdd = filtered.filter(c => !current.has(c));
                                                        if (toAdd.length > 0) {
                                                            setConfigForm(prev => ({ ...prev, coins: [...(prev.coins || []), ...toAdd] }));
                                                            setCoinSearch(""); // Clear on add
                                                        }
                                                    }}
                                                    className="px-2 py-1 text-[9px] font-bold bg-indigo-500/20 text-indigo-300 rounded border border-indigo-500/20 hover:bg-indigo-500 hover:text-white transition-all"
                                                >
                                                    ADD FILTERED
                                                </button>
                                            )}
                                        </div>

                                        {/* Autocomplete Dropdown */}
                                        {coinSearch && (
                                            <div className="absolute top-full left-0 right-0 mt-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-2xl z-50 max-h-48 overflow-y-auto custom-scrollbar">
                                                {availableCoins.filter(c => {
                                                    const matchesSearch = c.toLowerCase().includes(coinSearch.toLowerCase());
                                                    if (assetTab === "CRYPTO") {
                                                        if (!c.includes("/")) return false;
                                                        if (cryptoFilter === "USD" && !c.endsWith("/USD")) return false;
                                                        if (cryptoFilter === "USDT" && !c.endsWith("/USDT")) return false;
                                                    } else if (assetTab === "STOCKS") {
                                                        if (c.includes("/")) return false;
                                                    } else if (assetTab === "GLOBAL") {
                                                        // For global, we assume any symbol returned by the backend is fine
                                                        // Usually global symbols don't have a slash
                                                        if (c.includes("/")) return false;
                                                    }
                                                    return matchesSearch;
                                                }).slice(0, 50).map(coin => {
                                                    const isSelected = (configForm.coins || []).includes(coin);
                                                    return (
                                                        <button
                                                            key={coin}
                                                            disabled={isSelected}
                                                            onClick={() => {
                                                                if (!isSelected) {
                                                                    setConfigForm(prev => ({ ...prev, coins: [...(prev.coins || []), coin] }));
                                                                    setCoinSearch("");
                                                                }
                                                            }}
                                                            className={`w-full text-left px-4 py-2 text-xs font-mono flex items-center justify-between hover:bg-white/5 transition-colors ${isSelected ? "opacity-50 cursor-default" : ""}`}
                                                        >
                                                            <span className={isSelected ? "text-indigo-400" : "text-white"}>{coin}</span>
                                                            {isSelected && <Check className="w-3 h-3 text-indigo-500" />}
                                                        </button>
                                                    )
                                                })}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Active Assets Grid */}
                                <div className="bg-black/20 rounded-xl p-3 border border-white/5 min-h-[80px]">
                                    {(configForm.coins || []).length === 0 ? (
                                        <div className="h-full flex flex-col items-center justify-center text-zinc-600 gap-2 py-4">
                                            <Target className="w-8 h-8 opacity-20" />
                                            <span className="text-xs font-bold opacity-50">NO ASSETS TARGETED</span>
                                            <span className="text-[10px] opacity-40">Search and add symbols above</span>
                                        </div>
                                    ) : (
                                        <div className="flex flex-wrap gap-2">
                                            {(configForm.coins || [])
                                                .filter(coin => {
                                                    if (assetTab === "CRYPTO") {
                                                        if (cryptoFilter === "USD" && !coin.endsWith("/USD")) return false;
                                                        if (cryptoFilter === "USDT" && !coin.endsWith("/USDT")) return false;
                                                    } else if (assetTab === "STOCKS") {
                                                        if (coin.includes("/")) return false;
                                                    } else if (assetTab === "GLOBAL") {
                                                        if (coin.includes("/")) return false;
                                                        // ideally we'd check if it's in the global list, but this is a good first pass
                                                    }
                                                    return true;
                                                })
                                                .slice().reverse().map(coin => (
                                                    <span key={coin} className="group px-2.5 py-1.5 rounded-lg bg-indigo-500/10 text-indigo-300 text-xs font-bold border border-indigo-500/10 flex items-center gap-2 hover:bg-indigo-500/20 transition-all select-none">
                                                        {coin}
                                                        <button
                                                            onClick={() => toggleCoin(coin)}
                                                            className="text-indigo-400/50 hover:text-white transition-colors p-0.5 rounded-md hover:bg-white/10"
                                                        >
                                                            <X className="w-3 h-3" />
                                                        </button>
                                                    </span>
                                                ))}
                                        </div>
                                    )}
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
                                            <select
                                                value={configForm.council_model_path}
                                                onChange={(e) => setConfigForm({ ...configForm, council_model_path: e.target.value })}
                                                className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-xs font-mono focus:outline-none focus:border-indigo-500/50 transition-all text-zinc-400"
                                            >
                                                <option value="">Select Model...</option>
                                                {availableModels.map(m => (
                                                    <option key={m} value={m}>{m.split('/').pop()}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Model Hub Section */}
                            <div className="bg-black/40 rounded-2xl p-5 border border-white/5 space-y-4 shadow-xl">
                                <button
                                    type="button"
                                    onClick={() => setModelHubOpen(v => !v)}
                                    className="w-full flex items-center justify-between"
                                >
                                    <span className="text-xs font-black text-purple-400 uppercase tracking-[0.2em] flex items-center gap-2">
                                        <ShieldCheck className="w-3.5 h-3.5" /> Model Intelligence
                                    </span>
                                    <span className="text-zinc-500">
                                        {modelHubOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    </span>
                                </button>
                                {modelHubOpen && (
                                <div className="space-y-4 animate-in fade-in duration-200">
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">King Model Path</label>
                                        <select
                                            value={configForm.king_model_path}
                                            onChange={(e) => setConfigForm({ ...configForm, king_model_path: e.target.value })}
                                            className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-xs font-mono focus:outline-none focus:border-purple-500/50 transition-all text-zinc-400"
                                        >
                                            <option value="">Select Model...</option>
                                            {availableModels.map(m => (
                                                <option key={m} value={m}>{m.split('/').pop()}</option>
                                            ))}
                                        </select>
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
                                    <div className="flex items-center justify-between pt-4 border-t border-white/5">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-white tracking-wider flex items-center gap-2">
                                                <Zap className="w-3 h-3 text-yellow-400" /> Auto-Tune Control
                                            </label>
                                            <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest">Self-Learning Optimizer</p>
                                        </div>
                                        <Switch.Root
                                            checked={configForm.use_auto_tune ?? true}
                                            onCheckedChange={(c: boolean) => setConfigForm({ ...configForm, use_auto_tune: c })}
                                            className={`w-10 h-5 rounded-full relative shadow-inner transition-colors duration-300 ${configForm.use_auto_tune !== false ? 'bg-yellow-600' : 'bg-zinc-700'}`}
                                        >
                                            <Switch.Thumb className={`block w-3 h-3 rounded-full bg-white shadow-lg transition-transform duration-300 transform translate-y-1 ${configForm.use_auto_tune !== false ? 'translate-x-6' : 'translate-x-1'}`} />
                                        </Switch.Root>
                                    </div>
                                </div>
                                )}
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
                                                    type="number" step="0.1"
                                                    value={Math.round((configForm.target_pct || 0) * 100 * 100) / 100}
                                                    onChange={(e) => setConfigForm({ ...configForm, target_pct: parseFloat(e.target.value) / 100 })}
                                                    className="w-full bg-black/60 border border-white/5 rounded-xl px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-emerald-500/50 transition-all text-emerald-200"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Stop Loss %</label>
                                                <input
                                                    type="number" step="0.1"
                                                    value={Math.round((configForm.stop_loss_pct || 0) * 100 * 100) / 100}
                                                    onChange={(e) => setConfigForm({ ...configForm, stop_loss_pct: parseFloat(e.target.value) / 100 })}
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
                                                            type="number" step="0.1"
                                                            value={Math.round((configForm.trail_be_pct || 0) * 100 * 100) / 100}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_be_pct: parseFloat(e.target.value) / 100 })}
                                                            className="w-full bg-black/60 border border-white/5 rounded-lg px-2 py-1.5 text-[10px] font-mono focus:outline-none transition-all"
                                                        />
                                                    </div>
                                                    <div className="space-y-1">
                                                        <label className="text-[8px] font-black text-zinc-600 uppercase">Lock Trigg %</label>
                                                        <input
                                                            type="number" step="0.1"
                                                            value={Math.round((configForm.trail_lock_trigger_pct || 0) * 100 * 100) / 100}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_lock_trigger_pct: parseFloat(e.target.value) / 100 })}
                                                            className="w-full bg-black/60 border border-white/5 rounded-lg px-2 py-1.5 text-[10px] font-mono focus:outline-none transition-all"
                                                        />
                                                    </div>
                                                    <div className="space-y-1">
                                                        <label className="text-[8px] font-black text-zinc-600 uppercase">Lock Profit %</label>
                                                        <input
                                                            type="number" step="0.1"
                                                            value={Math.round((configForm.trail_lock_pct || 0) * 100 * 100) / 100}
                                                            onChange={(e) => setConfigForm({ ...configForm, trail_lock_pct: parseFloat(e.target.value) / 100 })}
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
                                                type="number" step="0.1"
                                                value={Math.round((configForm.pct_cash_per_trade || 0) * 100 * 100) / 100}
                                                onChange={(e) => setConfigForm({ ...configForm, pct_cash_per_trade: parseFloat(e.target.value) / 100 })}
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
                                    value={(configForm.data_source as string) || "binance"}
                                    onChange={(e) => setConfigForm({ ...configForm, data_source: e.target.value })}
                                    className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-white/20 transition-all text-white"
                                >
                                    <option value="binance" className="bg-zinc-950 text-white">Binance</option>
                                    <option value="yfinance" className="bg-zinc-950 text-white">Yahoo Finance</option>
                                    {assetTab === "GLOBAL" && (
                                        <option value="tvdata" className="bg-zinc-950 text-white">TradingView (tvdata)</option>
                                    )}
                                </select>
                            </div>

                            {selectedBotId !== "primary" && (
                                <div className="pt-4 border-t border-white/5">
                                    <button
                                        onClick={() => handleDeleteBot(selectedBotId, configForm.name || "Bot")}
                                        className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all text-xs font-black uppercase tracking-widest group"
                                    >
                                        <Trash2 className="w-4 h-4 group-hover:scale-110 transition-transform" />
                                        Delete Bot Instance
                                    </button>
                                </div>
                            )}

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

                            <div className="flex items-center justify-center gap-2 w-full py-4 rounded-xl bg-zinc-900/50 border border-white/5 text-xs font-bold text-zinc-500 uppercase tracking-widest">
                                <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-zinc-700" : "bg-emerald-500 animate-pulse"}`} />
                                {isRunning ? "Stop to Edit Config" : "Configuration Auto-Saves"}
                            </div>
                        </div>

                        {/* Right Column: Performance Cards + Logs & Trades */}
                        <div className="lg:col-span-2 flex flex-col gap-6 h-full">
                            {/* Performance Cards */}
                            <div className="grid grid-cols-2 gap-4">
                                {/* Total Profit Card */}
                                <div className="bg-zinc-900/40 border border-white/5 rounded-2xl p-5 backdrop-blur-xl group hover:border-emerald-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                                            <BarChart3 className="w-4 h-4" />
                                        </div>
                                        <span className="text-[9px] font-black text-zinc-500 uppercase tracking-widest">Total Profit</span>
                                    </div>
                                    <div className={`text-2xl font-black ${(performance?.profit_loss || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        ${(performance?.profit_loss || 0).toFixed(2)}
                                    </div>
                                    <div className="mt-2 text-[10px] text-zinc-500 flex items-center gap-1">
                                        <ArrowUpRight className="w-3 h-3" />
                                        {performance?.total_trades || 0} trades
                                    </div>
                                </div>

                                {/* Win Rate Card */}
                                <div className="bg-zinc-900/40 border border-white/5 rounded-2xl p-5 backdrop-blur-xl group hover:border-indigo-500/30 transition-all">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                                            <PieChart className="w-4 h-4" />
                                        </div>
                                        <span className="text-[9px] font-black text-zinc-500 uppercase tracking-widest">Win Rate</span>
                                    </div>
                                    <div className="text-2xl font-black text-indigo-400">
                                        {(performance?.win_rate || 0).toFixed(1)}%
                                    </div>
                                    <div className="mt-2 text-[10px] text-zinc-500">
                                        Consensus Alpha
                                    </div>
                                </div>
                            </div>

                            {/* Live Logs */}
                            <div className={`flex-1 bg-black border border-zinc-800 rounded-3xl shadow-2xl overflow-hidden flex flex-col transition-all duration-500 ease-in-out ${logsCollapsed ? "min-h-[64px] h-[64px] flex-none" : "min-h-[600px]"}`}>
                                <div
                                    className="flex items-center justify-between px-6 py-4 bg-zinc-900/50 border-b border-zinc-800 cursor-pointer hover:bg-zinc-900/70 transition-colors"
                                    onClick={() => setLogsCollapsed(!logsCollapsed)}
                                >
                                    <div className="flex items-center gap-3">
                                        <Terminal className="w-5 h-5 text-emerald-400" />
                                        <h2 className="text-sm font-bold tracking-widest text-zinc-300">SYSTEM LOGS</h2>
                                        {logsCollapsed && (
                                            <span className="text-[10px] text-zinc-500 font-mono animate-pulse">
                                                {status?.current_activity || "IDLE"}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="flex gap-2 items-center" onClick={(e) => e.stopPropagation()}>
                                            <button
                                                onClick={handleCopyLogs}
                                                className={`p-1.5 rounded-lg transition-all ${copyingLogs ? 'bg-emerald-500/20 text-emerald-400' : 'hover:bg-white/10 text-zinc-500'}`}
                                                title="Copy all logs"
                                            >
                                                {copyingLogs ? <CheckCheck className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                                            </button>

                                            <button
                                                onClick={handleClearLogs}
                                                className="p-1.5 rounded-lg transition-all hover:bg-red-500/20 text-zinc-500 hover:text-red-400"
                                                title="Clear logs"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>

                                            <div className="w-[1px] h-4 bg-zinc-800 mx-1" />

                                            {(['ALL', 'ACCEPTED', 'REJECTED', 'ERROR'] as const).map(f => (
                                                <button
                                                    key={f}
                                                    onClick={() => setLogFilter(f)}
                                                    className={`px-3 py-1 rounded-full text-[10px] font-bold transition-all ${logFilter === f
                                                        ? 'bg-white text-black shadow-lg shadow-white/10'
                                                        : 'bg-zinc-800 text-zinc-500 hover:bg-zinc-700'
                                                        }`}
                                                >
                                                    {f}
                                                </button>
                                            ))}
                                        </div>
                                        <div className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                                            {logsCollapsed ? <ChevronDown className="w-5 h-5 text-zinc-500" /> : <ChevronUp className="w-5 h-5 text-zinc-500" />}
                                        </div>
                                    </div>
                                </div>

                                {!logsCollapsed && (
                                    <>
                                        {/* Threshold Dashboard */}
                                        <div className="px-6 py-3 border-b border-zinc-800 bg-black/40 flex items-center justify-between gap-4 overflow-x-auto animate-in slide-in-from-top-2 duration-300">
                                            {Object.entries(thresholdStats).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0])).map(([thresh, count]) => (
                                                <div key={thresh} className="flex flex-col items-center min-w-[3rem]">
                                                    <span className="text-[9px] font-black text-zinc-600 uppercase">TH {thresh}</span>
                                                    <span className={`text-xs font-mono font-bold ${count > 0 ? 'text-indigo-400' : 'text-zinc-700'}`}>
                                                        {count}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>

                                        <div className="flex-1 p-6 overflow-y-auto font-mono text-xs space-y-1.5 custom-scrollbar max-h-[1200px] animate-in fade-in duration-500">
                                            {status?.logs && status.logs.length > 0 ? (
                                                status?.logs
                                                    .filter(log => {
                                                        if (logFilter === 'ALL') return true;
                                                        if (logFilter === 'ACCEPTED') return log.includes("ACCEPTED") || log.includes("BUY");
                                                        if (logFilter === 'REJECTED') return log.includes("REJECTED");
                                                        if (logFilter === 'ERROR') return log.includes("ERROR") || log.includes("Exception");
                                                        return true;
                                                    })
                                                    .map((log, i) => {
                                                        const parts = log.split("]");
                                                        const header = parts.slice(0, 2).join("]") + (parts.length > 1 ? "]" : "");
                                                        const message = parts.slice(2).join("]");
                                                        return (
                                                            <div key={i} className={`break-all border-l-2 pl-3 py-0.5 ${log.includes("BUY") ? "border-emerald-500 text-emerald-400 bg-emerald-500/5" :
                                                                log.includes("ERROR") || log.includes("DATA ERROR") ? "border-red-500 text-red-500 bg-red-500/5" :
                                                                    log.includes("SIGNAL") ? "border-indigo-500 text-indigo-300" :
                                                                        log.includes("DEBUG") ? "border-zinc-700 text-zinc-500 text-[10px]" :
                                                                            "border-zinc-800 text-zinc-400"
                                                                }`}>
                                                                <span className="opacity-50 mr-2 font-bold">{header}</span>
                                                                {message}
                                                            </div>
                                                        );
                                                    })
                                            ) : (
                                                <div className="text-zinc-700 italic flex items-center justify-center h-full">Waiting for data stream...</div>
                                            )}
                                            <div ref={logsEndRef} />
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* ACTIVE CHARTS SECTION */}
                            {positionsBySymbol.size > 0 && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-top-4 duration-700">
                                    <div className="flex items-center justify-between px-2">
                                        <div className="flex items-center gap-3">
                                            <Activity className="w-5 h-5 text-indigo-400" />
                                            <h2 className="text-[10px] font-black tracking-[0.2em] text-zinc-400 uppercase">Active Position Monitors</h2>
                                        </div>
                                        <div className="px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-[9px] font-black text-indigo-400 uppercase tracking-widest">
                                            {positionsBySymbol.size} Active
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-4">
                                        {Array.from(positionsBySymbol.values()).map((pos) => (
                                            <LiveCandleChart
                                                key={pos.symbol}
                                                symbol={pos.symbol}
                                                botId={selectedBotId}
                                                height={350}
                                                showControls={true}
                                                autoRefresh={true}
                                                paused={!!selectedChartSymbol}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Recent Executions Section - Twin Tables */}
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                                {/* BUY SIGNALS & ORDERS */}
                                <div className="bg-black/80 border border-zinc-800 rounded-3xl p-1 shadow-2xl flex flex-col min-h-[400px]">
                                    <div className="flex items-center justify-between px-6 py-4 bg-zinc-900/50 border-b border-zinc-800">
                                        <div className="flex items-center gap-3">
                                            <ArrowUpRight className="w-5 h-5 text-emerald-400" />
                                            <h2 className="text-[10px] font-black tracking-[0.2em] text-zinc-400">BUY SIGNALS & ORDERS</h2>
                                        </div>
                                    </div>
                                    <div className="flex-1 overflow-x-auto">
                                        <table className="w-full text-[11px] text-left border-collapse">
                                            <thead className="bg-zinc-900/50 text-[10px] font-black text-zinc-500 uppercase tracking-wider border-b border-zinc-800">
                                                <tr>
                                                    <th className="px-4 py-3">Time</th>
                                                    <th className="px-4 py-3">Symbol</th>
                                                    <th className="px-4 py-3">Price</th>
                                                    <th className="px-4 py-3">Total P/L ($)</th>
                                                    <th className="px-4 py-3">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5 font-mono">
                                                {(() => {
                                                    // Bot-scoped trades only (no account-wide Alpaca merge).
                                                    const botTrades = (performance?.trades || status?.trades || [])
                                                        .filter(t => isBotSymbol(t.symbol));
                                                    const buyTrades = botTrades
                                                        .filter(t => t.action === 'BUY' || t.action === 'SIGNAL' || t.action.startsWith('ALPACA:'));

                                                    console.log(`[Dashboard] Buy Table - Displaying from ${performance?.trades ? 'Performance (Supabase)' : 'Status (Memory)'}`);

                                                    // Deduplicate: keep only the latest trade per symbol
                                                    const seen = new Set<string>();
                                                    const deduped = buyTrades.slice().reverse().filter(t => {
                                                        const key = normalizePosKey(t.symbol);
                                                        if (seen.has(key)) return false;
                                                        seen.add(key);
                                                        return true;
                                                    });
                                                    if (deduped.length === 0) {
                                                        return <tr><td colSpan={4} className="px-4 py-8 text-center text-zinc-700 italic">No buy signals recorded</td></tr>;
                                                    }
                                                    return deduped.slice(0, 30).map((trade, i) => {
                                                        const isActive = positionsBySymbol.has(normalizePosKey(trade.symbol));
                                                        return (
                                                            <tr key={i} className={`hover:bg-white/5 transition-colors group ${isActive ? 'bg-emerald-500/5' : ''}`}>
                                                                <td className="px-4 py-3 text-zinc-600 whitespace-nowrap text-[10px]">
                                                                    {new Date(trade.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                                </td>
                                                                <td className={`px-4 py-3 font-bold transition-all ${isActive ? 'text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.3)]' : 'text-zinc-300'}`}>
                                                                    {trade.symbol}
                                                                </td>
                                                                <td className="px-4 py-3 text-zinc-400">${(trade.price || 0).toFixed(2)}</td>
                                                                <td className="px-4 py-3">
                                                                    {(() => {
                                                                        const pos = positionsBySymbol.get(normalizePosKey(trade.symbol));
                                                                        const pl = pos ? parseFloat(pos.unrealized_pl || "0") : (trade.pnl || 0);
                                                                        if (pl === 0 && !isActive) return <span className="text-zinc-600">--</span>;
                                                                        return (
                                                                            <span className={`font-mono font-bold ${pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                                {pl >= 0 ? '+' : ''}${pl.toFixed(2)}
                                                                            </span>
                                                                        );
                                                                    })()}
                                                                </td>
                                                                <td className="px-4 py-3">
                                                                    <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${trade.action === 'BUY' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/10' :
                                                                        trade.action.startsWith('ALPACA:') ? 'bg-orange-500/10 text-orange-400 border border-orange-500/10' :
                                                                            'bg-indigo-500/10 text-indigo-400 border border-indigo-500/10'
                                                                        }`}>
                                                                        {trade.action}
                                                                    </span>
                                                                </td>
                                                            </tr>
                                                        );
                                                    });
                                                })()}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* SELL EXECUTIONS & PROFITS */}
                                <div className="bg-black/80 border border-zinc-800 rounded-3xl p-1 shadow-2xl flex flex-col min-h-[400px]">
                                    <div className="flex items-center justify-between px-6 py-4 bg-zinc-900/50 border-b border-zinc-800">
                                        <div className="flex items-center gap-3">
                                            <ArrowDownRight className="w-5 h-5 text-red-400" />
                                            <h2 className="text-[10px] font-black tracking-[0.2em] text-zinc-400">SELL EXECUTIONS & PROFITS</h2>
                                        </div>
                                    </div>
                                    <div className="flex-1 overflow-x-auto">
                                        <table className="w-full text-[11px] text-left border-collapse">
                                            <thead className="bg-zinc-900/50 text-[10px] font-black text-zinc-500 uppercase tracking-wider border-b border-zinc-800">
                                                <tr>
                                                    <th className="px-4 py-3">Time</th>
                                                    <th className="px-4 py-3">Symbol</th>
                                                    <th className="px-4 py-3 text-right">Profit</th>
                                                    <th className="px-4 py-3 text-right">Price</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5 font-mono">
                                                {(() => {
                                                    const allTrades = (performance?.trades || status?.trades || [])
                                                        .filter(t => isBotSymbol(t.symbol));
                                                    const sellTrades = allTrades.filter(t => t.action === 'SELL');

                                                    console.log(`[Dashboard] Sell Table - Displaying from ${performance?.trades ? 'Performance (Supabase)' : 'Status (Memory)'}`);

                                                    if (sellTrades.length === 0) {
                                                        return <tr><td colSpan={4} className="px-4 py-8 text-center text-zinc-700 italic">No sell executions yet</td></tr>;
                                                    }

                                                    return sellTrades.slice().reverse().slice(0, 30).map((trade, i) => {
                                                        const pnl = computeSellPnl(trade);
                                                        const price = toNum(trade.price);
                                                        const isActive = positionsBySymbol.has(normalizePosKey(trade.symbol));
                                                        return (
                                                            <tr key={i} className={`hover:bg-white/5 transition-colors group ${isActive ? 'bg-indigo-500/5' : ''}`}>
                                                                <td className="px-4 py-3 text-zinc-600 whitespace-nowrap text-[10px]">
                                                                    {new Date(trade.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                                </td>
                                                                <td
                                                                    className={`px-4 py-3 font-bold transition-all cursor-pointer hover:scale-105 ${isActive ? 'text-indigo-400 drop-shadow-[0_0_8px_rgba(129,140,248,0.3)]' : 'text-zinc-300 hover:text-indigo-400'}`}
                                                                    onClick={() => {
                                                                        // Set the chart symbol to show trade details
                                                                        setSelectedChartSymbol(trade.symbol);
                                                                        // Scroll to chart section
                                                                        document.getElementById('live-chart-section')?.scrollIntoView({ behavior: 'smooth' });
                                                                    }}
                                                                    title={`Click to view ${trade.symbol} trade chart`}
                                                                >
                                                                    {trade.symbol}
                                                                </td>
                                                                <td className={`px-4 py-3 text-right font-black ${(pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                    {pnl === null ? <span className="text-zinc-600">--</span> : `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`}
                                                                </td>
                                                                <td className="px-4 py-3 text-right text-zinc-500">
                                                                    {price === null || Math.abs(price) < 1e-12 ? <span className="text-zinc-600">--</span> : `$${price.toFixed(2)}`}
                                                                </td>
                                                            </tr>
                                                        );
                                                    });
                                                })()}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

                            {/* Market Data Stream - moved below BUY/SELL tables */}
                            <div className={`bg-black/80 border border-zinc-800 rounded-3xl shadow-2xl flex flex-col transition-all duration-500 ease-in-out relative group/stream ${marketDataCollapsed ? "min-h-[64px] h-[64px]" : "min-h-[250px]"}`}>
                                <div
                                    className="flex items-center justify-between px-6 py-4 bg-zinc-900/30 border-b border-zinc-800 cursor-pointer hover:bg-zinc-900/50 transition-colors"
                                    onClick={() => setMarketDataCollapsed(!marketDataCollapsed)}
                                >
                                    <div className="flex items-center gap-3">
                                        <Globe className="w-5 h-5 text-indigo-400" />
                                        <h2 className="text-sm font-bold tracking-widest text-zinc-300">MARKET DATA STREAM</h2>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-[10px] font-black text-emerald-500/70 uppercase flex items-center gap-1.5 backdrop-blur-md bg-emerald-500/5 px-3 py-1 rounded-full border border-emerald-500/10">
                                            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                                            Live Processing
                                        </div>
                                        {marketDataCollapsed ? <ChevronDown className="w-5 h-5 text-zinc-500" /> : <ChevronUp className="w-5 h-5 text-zinc-500" />}
                                    </div>
                                </div>

                                {!marketDataCollapsed && (
                                    <div className="flex-1 overflow-x-auto animate-in slide-in-from-top-2 duration-500">
                                        <table className="w-full text-[11px] text-left border-collapse">
                                            <thead className="bg-zinc-900/50 text-[10px] font-black text-zinc-500 uppercase tracking-wider border-b border-zinc-800">
                                                <tr>
                                                    <th className="px-6 py-3">Symbol</th>
                                                    <th className="px-6 py-3">Source</th>
                                                    <th className="px-6 py-3">Bars</th>
                                                    <th className="px-6 py-3">Volume</th>
                                                    <th className="px-6 py-3">Status</th>
                                                    <th className="px-6 py-3 text-right">Last Update</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5 font-mono">
                                                {status?.data_stream && Object.keys(status.data_stream).length > 0 ? (
                                                    Object.entries(status?.data_stream || {}).map(([sym, data]: [string, any]) => (
                                                        <tr key={sym} className="hover:bg-indigo-500/5 transition-colors font-mono">
                                                            <td className="px-6 py-3 font-bold text-zinc-200">{sym}</td>
                                                            <td className="px-6 py-3 text-zinc-500">{data.source}</td>
                                                            <td className="px-6 py-3">
                                                                <span className={data.count > 0 ? "text-emerald-400 font-bold" : "text-red-400 font-bold"}>{data.count}</span>
                                                            </td>
                                                            <td className="px-6 py-3">
                                                                {data.has_volume ? (
                                                                    <span className="text-emerald-500 flex items-center gap-1">
                                                                        <Check className="w-3 h-3" /> YES
                                                                    </span>
                                                                ) : (
                                                                    <span className="text-red-500 flex items-center gap-1">
                                                                        <X className="w-3 h-3" /> NO
                                                                    </span>
                                                                )}
                                                            </td>
                                                            <td className="px-6 py-3">
                                                                <span className={`px-2 py-0.5 rounded-full text-[9px] font-black ${data.status === 'OK' ? 'bg-emerald-500/10 text-emerald-500' :
                                                                    data.status === 'EMPTY' ? 'bg-amber-500/10 text-amber-500' : 'bg-red-500/10 text-red-500'
                                                                    }`}>
                                                                    {data.status}
                                                                </span>
                                                            </td>
                                                            <td className="px-6 py-3 text-right text-zinc-600">
                                                                {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'N/A'}
                                                            </td>
                                                        </tr>
                                                    ))
                                                ) : (
                                                    <tr>
                                                        <td colSpan={6} className="px-6 py-8 text-center text-zinc-700 italic">No data stream initialized</td>
                                                    </tr>
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
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
                                            {performance?.exit_reasons && Object.entries(performance!.exit_reasons).length > 0 ? (
                                                Object.entries(performance!.exit_reasons).map(([reason, count]) => {
                                                    const total = Object.values(performance!.exit_reasons).reduce((a, b) => a + b, 0);
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
                                                    {performance?.symbol_performance && Object.entries(performance!.symbol_performance).length > 0 ? (
                                                        Object.entries(performance!.symbol_performance)
                                                            .sort(([, a], [, b]) => b.profit - a.profit)
                                                            .map(([symbol, stats]) => (
                                                                <tr key={symbol} className="hover:bg-white/5 transition-colors group">
                                                                    <td className="px-4 py-3">
                                                                        <button
                                                                            onClick={() => setSelectedChartSymbol(symbol)}
                                                                            className="flex items-center gap-2 font-bold text-zinc-300 hover:text-indigo-400 transition-colors group/btn"
                                                                        >
                                                                            {symbol}
                                                                            <Maximize2 className="w-3 h-3 opacity-0 group-hover/btn:opacity-100" />
                                                                        </button>
                                                                    </td>
                                                                    <td className="px-4 py-3 text-center text-zinc-400">{stats.trades}</td>
                                                                    <td className="px-4 py-3 text-center">
                                                                        <span className={(stats.win_rate || 0) >= 50 ? 'text-emerald-400' : 'text-zinc-500'}>
                                                                            {(stats.win_rate || 0).toFixed(0)}%
                                                                        </span>
                                                                    </td>
                                                                    <td className={`px-4 py-3 text-right font-bold ${(stats.profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                        {(stats.profit || 0) >= 0 ? '+' : ''}{(stats.profit || 0).toFixed(2)}
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
                                                    <th className="px-4 py-4 text-center">Entry</th>
                                                    <th className="px-4 py-4 text-center">Target</th>
                                                    <th className="px-4 py-4 text-center">Current</th>
                                                    <th className="px-4 py-4 text-center">Stop</th>
                                                    <th className="px-4 py-4 text-center">P/L %</th>
                                                    <th className="px-4 py-4 text-right">P/L $</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5 font-mono text-xs">
                                                {performance?.open_positions && performance!.open_positions.length > 0 ? (
                                                    performance!.open_positions.map((pos, i) => (
                                                        <tr key={i} className="hover:bg-white/5 transition-colors group">
                                                            <td className="px-4 py-4 font-bold text-white">
                                                                <button
                                                                    onClick={() => setSelectedChartSymbol(pos.symbol)}
                                                                    className="flex items-center gap-2 font-bold text-zinc-300 hover:text-indigo-400 transition-colors group/btn"
                                                                >
                                                                    {pos.symbol}
                                                                    <Maximize2 className="w-3 h-3 opacity-0 group-hover/btn:opacity-100" />
                                                                </button>
                                                            </td>
                                                            <td className="px-4 py-4">
                                                                <span className="px-2 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/10 text-[10px] font-black uppercase tracking-tighter">
                                                                    In Progress
                                                                </span>
                                                            </td>
                                                            <td className="px-4 py-4 text-zinc-400 text-center">${(pos.entry_price || 0).toFixed(4)}</td>
                                                            <td className="px-4 py-4 text-emerald-400 font-bold text-center">
                                                                {pos.target_price ? `$${pos.target_price.toFixed(4)}` : '--'}
                                                            </td>
                                                            <td className="px-4 py-4 text-white text-center">${(pos.current_price || 0).toFixed(4)}</td>
                                                            <td className="px-4 py-4 text-red-400 font-bold text-center">
                                                                {pos.stop_price ? `$${pos.stop_price.toFixed(4)}` : '--'}
                                                            </td>
                                                            <td className={`px-4 py-4 text-center font-bold ${(pos.pl_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {pos.pl_pct !== undefined ? (pos.pl_pct >= 0 ? '+' : '') + pos.pl_pct.toFixed(2) + '%' : 'N/A'}
                                                            </td>
                                                            <td className={`px-4 py-4 text-right font-bold ${(pos.pl_usd || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                ${(pos.pl_usd || 0).toFixed(2)}
                                                            </td>
                                                        </tr>
                                                    ))
                                                ) : (
                                                    <tr>
                                                        <td colSpan={8} className="px-4 py-12 text-center text-zinc-600 italic font-mono uppercase tracking-[0.2em] text-[10px]">
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
                )
            }

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

                        <div className="flex gap-2 mb-4 p-1 bg-black/40 rounded-lg border border-white/5">
                            <button
                                onClick={() => setCoinSource("database")}
                                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${coinSource === "database"
                                    ? "bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                                    : "text-zinc-500 hover:text-zinc-300"
                                    }`}
                            >
                                My Symbols
                            </button>
                            <button
                                onClick={() => setCoinSource("alpaca")}
                                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${coinSource === "alpaca"
                                    ? "bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                                    : "text-zinc-500 hover:text-zinc-300"
                                    }`}
                            >
                                All Alpaca
                            </button>
                        </div>

                        {coinSource === "alpaca" && (
                            <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[10px] text-amber-200/70 leading-relaxed animate-in fade-in duration-500">
                                <Activity className="w-3 h-3 mb-1 text-amber-500" />
                                <strong>Important:</strong> Trading <code>/USDC</code> or <code>/USDT</code> pairs requires having those specific assets. If you only have USD, please use <code>/USD</code> pairs (recommended).
                            </div>
                        )}

                        {coinSource === "alpaca" && (
                            <div className="flex gap-2 mb-4 overflow-x-auto custom-scrollbar pb-2">
                                {[10, 20, 50, 100, 200, 500, 1000, 0].map(lim => (
                                    <button
                                        key={lim}
                                        onClick={() => setCoinLimit(lim)}
                                        className={`px-3 py-1 text-[10px] font-bold rounded-full whitespace-nowrap transition-all border ${coinLimit === lim
                                            ? "bg-emerald-500/20 border-emerald-500 text-emerald-400"
                                            : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-500"
                                            }`}
                                    >
                                        {lim === 0 ? "ALL" : `Top ${lim}`}
                                    </button>
                                ))}
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-2 max-h-[300px] overflow-y-auto custom-scrollbar pr-2">
                            {(availableCoins.length > 0 ? availableCoins : COMMON_COINS).filter(c => c && typeof c === "string" && c.toLowerCase().includes(coinSearch.toLowerCase())).map(coin => {
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

            {/* Historical Chart Modal */}
            <Dialog.Root open={!!selectedChartSymbol} onOpenChange={(open: boolean) => !open && setSelectedChartSymbol(null)}>
                <Dialog.Portal>
                    <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-md z-[100] transition-opacity" />
                    <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[95vw] h-[90vh] bg-zinc-950 border border-zinc-800 rounded-3xl p-1 z-[101] shadow-2xl flex flex-col">
                        <div className="flex-1 min-h-0">
                            {selectedChartSymbol && (
                                <LiveCandleChart
                                    symbol={selectedChartSymbol!}
                                    botId={selectedBotId}
                                    height="100%"
                                    onClose={() => setSelectedChartSymbol(null)}
                                    showControls={true}
                                    autoRefresh={false} // Static historical view
                                />
                            )}
                        </div>
                    </Dialog.Content>
                </Dialog.Portal>
            </Dialog.Root>

            <style jsx global>{`
                    .custom-scrollbar::-webkit-scrollbar {
                        width: 6px;
                        height: 6px;
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
        </div >
    )
}
