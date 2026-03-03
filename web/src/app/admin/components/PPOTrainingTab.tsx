"use client";

import { Zap, ChevronDown, Loader2, Database, History, TrendingUp, TrendingDown, Clock, Play, StopCircle, Brain, Layers, Cpu, Activity } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { useState, useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";

interface PPOTrainingTabProps {
    dbInventory: any[];
    trainingExchange: string;
    setTrainingExchange: (ex: string) => void;
    isExchangeDropdownOpen: boolean;
    setIsExchangeDropdownOpen: (open: boolean) => void;
    isTraining: boolean;
    setIsTraining: (training: boolean) => void;
}

export default function PPOTrainingTab({
    dbInventory,
    trainingExchange,
    setTrainingExchange,
    isExchangeDropdownOpen,
    setIsExchangeDropdownOpen,
    isTraining,
    setIsTraining
}: PPOTrainingTabProps) {
    // PPO Hyperparameters
    const [modelName, setModelName] = useState<string>("");
    const [totalTimesteps, setTotalTimesteps] = useState<number>(200000);
    const [learningRate, setLearningRate] = useState<number>(0.0003);
    const [nSteps, setNSteps] = useState<number>(2048);
    const [batchSize, setBatchSize] = useState<number>(64);
    const [nEpochs, setNEpochs] = useState<number>(10);
    const [gamma, setGamma] = useState<number>(0.99);
    const [clipRange, setClipRange] = useState<number>(0.2);
    const [entCoef, setEntCoef] = useState<number>(0.01);
    const [vfCoef, setVfCoef] = useState<number>(0.5);
    const [netArch, setNetArch] = useState<"small" | "medium" | "large">("medium");

    // Helpers for dynamic NN visualization
    const nodesPerLayer = netArch === "small" ? 5 : netArch === "medium" ? 8 : 12;
    const inputY = (i: number) => 65 + i * 32;
    const hiddenY = (i: number) => {
        const totalH = (nodesPerLayer - 1) * (400 / nodesPerLayer);
        const startY = (450 - totalH) / 2;
        return startY + i * (400 / nodesPerLayer);
    };
    const outputY = (i: number) => 110 + i * 55;

    // Environment parameters  
    const [initialBalance, setInitialBalance] = useState<number>(10000);
    const [maxSteps, setMaxSteps] = useState<number>(1000);
    const [rewardMode, setRewardMode] = useState<"pnl" | "sharpe" | "sortino">("pnl");

    // Training status
    const [trainingStatus, setTrainingStatus] = useState<{
        running: boolean;
        phase?: string | null;
        last_message?: string | null;
        started_at?: string | null;
        completed_at?: string | null;
        error?: string | null;
        stats?: Record<string, any> | null;
    } | null>(null);

    // Training logs & metrics
    const [trainingLogs, setTrainingLogs] = useState<Array<{ ts: string; msg: string }>>([]);
    const [lastLoggedMessage, setLastLoggedMessage] = useState<string | null>(null);
    const [policyLossCurve, setPolicyLossCurve] = useState<Array<{ step: number; loss: number }>>([]);
    const [rewardCurve, setRewardCurve] = useState<Array<{ step: number; reward: number }>>([]);
    const [valueLossCurve, setValueLossCurve] = useState<Array<{ step: number; loss: number }>>([]);
    const [triggeringTraining, setTriggeringTraining] = useState(false);
    const logEndRef = useRef<HTMLDivElement>(null);

    const exchangeOptions = useMemo(() => {
        const options = dbInventory.filter(i => i.priceCount > 0);
        if (!options.find(o => o.exchange === "CRYPTO" || o.exchange === "BINANCE")) {
            // Add a virtual Crypto entry if not found, to allow users to select it
            options.push({ exchange: "CRYPTO", priceCount: 1, symbols: ["BTC-USD"] });
        }
        return options;
    }, [dbInventory]);

    const selectedExchangeCount = dbInventory.find(i => i.exchange === trainingExchange)?.priceCount || 0;

    const netArchConfig = {
        small: { layers: [32, 32], params: "~2,200", label: "32 × 32" },
        medium: { layers: [64, 64], params: "~8,500", label: "64 × 64" },
        large: { layers: [128, 128], params: "~33,000", label: "128 × 128" },
    };

    // Estimated training time
    const estimatedTime = useMemo(() => {
        const base = totalTimesteps / 1000; // rough seconds per 1k steps
        const archMultiplier = netArch === "small" ? 0.5 : netArch === "large" ? 2 : 1;
        const secs = base * archMultiplier;
        return {
            minutes: Math.ceil(secs / 60),
            descriptor: secs < 300 ? "Quick Run" : secs < 1800 ? "Standard Run" : "Deep Training",
            color: secs < 300 ? "text-emerald-400" : secs < 1800 ? "text-indigo-400" : "text-purple-400"
        };
    }, [totalTimesteps, netArch]);

    // Log accumulation
    useEffect(() => {
        if (!trainingStatus?.last_message) return;
        const msg = String(trainingStatus.last_message);
        if (!msg.trim() || msg === lastLoggedMessage) return;
        setLastLoggedMessage(msg);
        setTrainingLogs(prev => {
            const next = [...prev, { ts: new Date().toISOString(), msg }];
            return next.length > 300 ? next.slice(next.length - 300) : next;
        });
    }, [trainingStatus?.last_message, lastLoggedMessage]);

    // Auto-scroll logs - Removed per user request
    // useEffect(() => {
    //     logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    // }, [trainingLogs]);

    // Metric curve updates from live stats
    useEffect(() => {
        if (!trainingStatus?.running || !trainingStatus.stats) return;
        const stats = trainingStatus.stats;
        const now = Date.now();

        if (typeof stats.policy_loss === "number") {
            setPolicyLossCurve(prev => {
                const last = prev[prev.length - 1];
                if (last && Math.abs(last.loss - stats.policy_loss) < 0.0001) return prev;
                const next = [...prev, { step: now, loss: stats.policy_loss }];
                return next.length > 200 ? next.slice(-200) : next;
            });
        }
        if (typeof stats.value_loss === "number") {
            setValueLossCurve(prev => {
                const last = prev[prev.length - 1];
                if (last && Math.abs(last.loss - stats.value_loss) < 0.0001) return prev;
                const next = [...prev, { step: now, loss: stats.value_loss }];
                return next.length > 200 ? next.slice(-200) : next;
            });
        }
        if (typeof stats.ep_rew_mean === "number") {
            setRewardCurve(prev => {
                const last = prev[prev.length - 1];
                if (last && Math.abs(last.reward - stats.ep_rew_mean) < 0.01) return prev;
                const next = [...prev, { step: now, reward: stats.ep_rew_mean }];
                return next.length > 200 ? next.slice(-200) : next;
            });
        }
    }, [trainingStatus?.running, trainingStatus?.stats]);

    // Polling for training status - Optimized to poll only when visible and less frequently when idle
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const res = await fetch("/api/admin/ppo/status");
                if (!res.ok) return;
                const data = await res.json();
                setTrainingStatus(data);
                if (typeof data.running === "boolean") {
                    setIsTraining(data.running);
                }
            } catch { /* ignore */ }
        };

        // Initial fetch
        fetchStatus();

        const pollInterval = isTraining ? 2000 : 30000; // 2s active, 30s idle
        const poll = setInterval(() => {
            if (document.visibilityState === 'visible') {
                fetchStatus();
            }
        }, pollInterval);

        const handleVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                fetchStatus();
            }
        };
        document.addEventListener("visibilitychange", handleVisibilityChange);

        return () => {
            clearInterval(poll);
            document.removeEventListener("visibilitychange", handleVisibilityChange);
        };
    }, [isTraining]);

    const startPPOTraining = async () => {
        if (!trainingExchange) {
            toast.error("Select an exchange first");
            return;
        }
        setTriggeringTraining(true);
        try {
            const res = await fetch("/api/admin/ppo/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    exchange: trainingExchange,
                    model_name: modelName || undefined,
                    total_timesteps: totalTimesteps,
                    learning_rate: learningRate,
                    n_steps: nSteps,
                    batch_size: batchSize,
                    n_epochs: nEpochs,
                    gamma,
                    clip_range: clipRange,
                    ent_coef: entCoef,
                    vf_coef: vfCoef,
                    net_arch: netArchConfig[netArch].layers,
                    initial_balance: initialBalance,
                    max_steps: maxSteps,
                    reward_mode: rewardMode
                })
            });

            if (res.ok) {
                toast.success("PPO training started");
                setIsTraining(true);
                // Clear old curves
                setPolicyLossCurve([]);
                setRewardCurve([]);
                setValueLossCurve([]);
                setTrainingLogs([]);
            } else {
                const err = await res.json().catch(() => null);
                let errorMsg = "Failed to start";
                if (err?.detail) {
                    if (Array.isArray(err.detail)) {
                        errorMsg = err.detail.map((e: any) => `${e.msg} (${e.loc?.join('.')})`).join(", ");
                    } else if (typeof err.detail === "object") {
                        errorMsg = JSON.stringify(err.detail);
                    } else {
                        errorMsg = String(err.detail);
                    }
                }
                toast.error(errorMsg);
            }
        } catch {
            toast.error("Connection error");
        } finally {
            setTriggeringTraining(false);
        }
    };

    const stopTraining = async () => {
        try {
            await fetch("/api/admin/ppo/stop", { method: "POST" });
            toast.success("Stopping...");
        } catch {
            toast.error("Failed to stop");
        }
    };

    return (
        <div className="p-8 max-w-[1800px] mx-auto w-full space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="flex flex-col gap-2">
                <h1 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
                    <Brain className="h-8 w-8 text-purple-500" />
                    PPO Reinforcement Learning
                </h1>
                <p className="text-sm text-zinc-500 font-medium">
                    Train PPO agents with full control over network architecture, hyperparameters, and environment settings.
                </p>
            </header>

            {/* Neural Network Architecture Visualizer */}
            <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="p-3 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
                            <Layers className="w-6 h-6" />
                        </div>
                        <div>
                            <h2 className="text-xl font-black text-white">MLP Policy Network Architecture</h2>
                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">
                                Input(11) → Dense({netArchConfig[netArch].layers[0]}) → Dense({netArchConfig[netArch].layers[1]}) → Actions
                            </p>
                        </div>
                    </div>
                    <div className="flex gap-3">
                        <span className="px-4 py-2 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-black border border-purple-500/20 flex items-center gap-2">
                            <Cpu className="w-3 h-3" /> {netArchConfig[netArch].params} Params
                        </span>
                        <span className="px-4 py-2 rounded-full bg-indigo-500/10 text-indigo-400 text-[10px] font-black border border-indigo-500/20 flex items-center gap-2">
                            <Activity className="w-3 h-3" /> Actor-Critic
                        </span>
                    </div>
                </div>

                {/* SVG Network Diagram */}
                <div className="w-full overflow-x-auto">
                    <svg viewBox="0 0 900 420" className="w-full min-w-[600px]" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="ppo-conn-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
                                <stop offset="50%" stopColor="#6366f1" stopOpacity="0.4" />
                                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.3" />
                            </linearGradient>
                            <linearGradient id="ppo-conn-grad2" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                                <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.4" />
                            </linearGradient>
                            <filter id="ppo-glow-p">
                                <feGaussianBlur stdDeviation="3" result="g" />
                                <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
                            </filter>
                            <filter id="ppo-glow-i">
                                <feGaussianBlur stdDeviation="3" result="g" />
                                <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
                            </filter>
                            <filter id="ppo-glow-b">
                                <feGaussianBlur stdDeviation="3" result="g" />
                                <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
                            </filter>
                            <filter id="ppo-glow-o">
                                <feGaussianBlur stdDeviation="3" result="g" />
                                <feMerge><feMergeNode in="g" /><feMergeNode in="SourceGraphic" /></feMerge>
                            </filter>
                        </defs>

                        {/* Layer labels */}
                        <text x="100" y="22" textAnchor="middle" fill="#8b5cf6" fontSize="10" fontWeight="900" letterSpacing="0.1em">INPUT LAYER</text>
                        <text x="100" y="38" textAnchor="middle" fill="#52525b" fontSize="8" fontWeight="bold">11 Features</text>
                        <text x="350" y="22" textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="900" letterSpacing="0.1em">HIDDEN 1</text>
                        <text x="350" y="38" textAnchor="middle" fill="#52525b" fontSize="8" fontWeight="bold">{netArchConfig[netArch].layers[0]} Neurons (ReLU)</text>
                        <text x="570" y="22" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="900" letterSpacing="0.1em">HIDDEN 2</text>
                        <text x="570" y="38" textAnchor="middle" fill="#52525b" fontSize="8" fontWeight="bold">{netArchConfig[netArch].layers[1]} Neurons (ReLU)</text>
                        <text x="790" y="22" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="900" letterSpacing="0.1em">OUTPUT</text>
                        <text x="790" y="38" textAnchor="middle" fill="#52525b" fontSize="8" fontWeight="bold">Actions</text>

                        {/* Connections: Input → H1 */}
                        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(i =>
                            [...Array(nodesPerLayer)].map((_, j) => (
                                <line key={`ih-${i}-${j}`} x1="118" y1={inputY(i)} x2="332" y2={hiddenY(j)}
                                    stroke="url(#ppo-conn-grad)"
                                    strokeWidth={netArch === 'large' ? 0.6 : 0.4}
                                    opacity={isTraining ? 0.35 : 0.15}
                                    className={isTraining ? "ppo-animate-flow" : ""}
                                    strokeDasharray="4 4" />
                            ))
                        )}

                        {/* Connections: H1 → H2 */}
                        {[...Array(nodesPerLayer)].map((_, i) =>
                            [...Array(nodesPerLayer)].map((_, j) => (
                                <line key={`hh-${i}-${j}`} x1="368" y1={hiddenY(i)} x2="552" y2={hiddenY(j)}
                                    stroke="url(#ppo-conn-grad)"
                                    strokeWidth={netArch === 'large' ? 0.6 : 0.4}
                                    opacity={isTraining ? 0.4 : 0.2}
                                    className={isTraining ? "ppo-animate-flow" : ""}
                                    strokeDasharray="4 4" />
                            ))
                        )}

                        {/* Connections: H2 → Output */}
                        {[...Array(nodesPerLayer)].map((_, i) =>
                            [0, 1, 2, 3, 4].map(j => (
                                <line key={`ho-${i}-${j}`} x1="588" y1={hiddenY(i)} x2="772" y2={outputY(j)}
                                    stroke="url(#ppo-conn-grad2)"
                                    strokeWidth={netArch === 'large' ? 0.8 : 0.5}
                                    opacity={isTraining ? 0.4 : 0.2}
                                    className={isTraining ? "ppo-animate-flow" : ""}
                                    strokeDasharray="4 4" />
                            ))
                        )}

                        {/* Input nodes */}
                        {['RSI', 'ATR', 'MACD', 'SIGN', 'SMA50', 'SMA200', 'HiDst', 'LoDst', 'RSIdf', 'V20', 'Day'].map((label, i) => (
                            <g key={`in-${i}`}>
                                <circle cx="100" cy={inputY(i)} r="13" fill="#8b5cf6" fillOpacity="0.12" stroke="#8b5cf6" strokeWidth="1.5" filter="url(#ppo-glow-p)" />
                                <circle cx="100" cy={inputY(i)} r="5" fill="#8b5cf6" fillOpacity={isTraining ? 1 : 0.8}>
                                    {isTraining && <animate attributeName="opacity" values="0.6;1;0.6" dur={`${1.5 + i * 0.1}s`} repeatCount="indefinite" />}
                                </circle>
                                <text x="68" y={inputY(i) + 4} fill="#8b5cf6" fontSize="7" fontWeight="bold" textAnchor="end" fontFamily="monospace">{label}</text>
                            </g>
                        ))}

                        {/* Hidden 1 nodes */}
                        {[...Array(nodesPerLayer)].map((_, i) => (
                            <g key={`h1-${i}`}>
                                <circle cx="350" cy={hiddenY(i)} r="13" fill="#6366f1" fillOpacity="0.12" stroke="#6366f1" strokeWidth="1.5" filter="url(#ppo-glow-i)" />
                                <circle cx="350" cy={hiddenY(i)} r="5" fill="#6366f1" fillOpacity={isTraining ? 1 : 0.8}>
                                    {isTraining && <animate attributeName="opacity" values="0.5;1;0.5" dur={`${2 + i * 0.2}s`} repeatCount="indefinite" />}
                                </circle>
                            </g>
                        ))}
                        <text x="350" y="385" fill="#52525b" fontSize="8" textAnchor="middle" fontWeight="bold" fontStyle="italic">... {netArchConfig[netArch].layers[0]} neurons total</text>

                        {/* Hidden 2 nodes */}
                        {[...Array(nodesPerLayer)].map((_, i) => (
                            <g key={`h2-${i}`}>
                                <circle cx="570" cy={hiddenY(i)} r="13" fill="#3b82f6" fillOpacity="0.12" stroke="#3b82f6" strokeWidth="1.5" filter="url(#ppo-glow-b)" />
                                <circle cx="570" cy={hiddenY(i)} r="5" fill="#3b82f6" fillOpacity={isTraining ? 1 : 0.8}>
                                    {isTraining && <animate attributeName="opacity" values="0.5;1;0.5" dur={`${2.5 + i * 0.2}s`} repeatCount="indefinite" />}
                                </circle>
                            </g>
                        ))}
                        <text x="570" y="385" fill="#52525b" fontSize="8" textAnchor="middle" fontWeight="bold" fontStyle="italic">... {netArchConfig[netArch].layers[1]} neurons total</text>

                        {/* Output nodes */}
                        {['HOLD', 'CLOSE', 'BUY', 'SELL', '...'].map((label, i) => (
                            <g key={`out-${i}`}>
                                <circle cx="790" cy={outputY(i)} r="14" fill="#f59e0b" fillOpacity="0.12" stroke="#f59e0b" strokeWidth="1.5" filter="url(#ppo-glow-o)" />
                                <circle cx="790" cy={outputY(i)} r="5" fill="#f59e0b" fillOpacity={isTraining ? 1 : 0.8}>
                                    {isTraining && <animate attributeName="opacity" values="0.8;1;0.8" dur="2s" repeatCount="indefinite" />}
                                </circle>
                                <text x="822" y={outputY(i) + 4} fill="#f59e0b" fontSize="7" fontWeight="bold" fontFamily="monospace">{label}</text>
                            </g>
                        ))}
                    </svg>
                </div>

                {/* Architecture Info Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'Input Features', value: '11', sub: '8 indicators + 3 state', color: 'text-purple-400', bg: 'bg-purple-500/10' },
                        { label: 'Hidden Neurons', value: String(netArchConfig[netArch].layers[0] * 2), sub: `2 × ${netArchConfig[netArch].layers[0]} Dense`, color: 'text-indigo-400', bg: 'bg-indigo-500/10' },
                        { label: 'Action Space', value: '130+', sub: 'HOLD + CLOSE + OPENs', color: 'text-blue-400', bg: 'bg-blue-500/10' },
                        { label: 'Activation', value: 'ReLU', sub: 'Non-linear Transform', color: 'text-amber-400', bg: 'bg-amber-500/10' }
                    ].map((card, i) => (
                        <div key={i} className={`${card.bg} p-5 rounded-2xl border border-white/5 text-center`}>
                            <p className={`text-3xl font-black tracking-tighter ${card.color}`}>{card.value}</p>
                            <p className="text-[10px] font-black uppercase tracking-widest text-zinc-500 mt-1">{card.label}</p>
                            <p className="text-[8px] font-bold text-zinc-600 mt-1">{card.sub}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Training Configuration + Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left: Config Panel */}
                <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-6">
                    <div className="flex items-center gap-4">
                        <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                            <Cpu className="w-6 h-6" />
                        </div>
                        <div>
                            <h2 className="text-xl font-black text-white">Training Config</h2>
                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">PPO Hyperparameters</p>
                        </div>
                    </div>

                    <div className="space-y-4">
                        {/* Model Name */}
                        <div className="space-y-1">
                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Model Name</label>
                            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)}
                                placeholder="ppo_alpha_v1"
                                className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                        </div>

                        {/* Network Architecture */}
                        <div className="space-y-1">
                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Network Architecture</label>
                            <div className="flex gap-2">
                                {(["small", "medium", "large"] as const).map(arch => (
                                    <button key={arch} onClick={() => setNetArch(arch)}
                                        className={`flex-1 h-9 rounded-xl text-[11px] font-bold border ${netArch === arch
                                            ? "bg-zinc-100 text-black border-zinc-100"
                                            : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"}`}>
                                        {arch === "small" ? "32×32" : arch === "medium" ? "64×64" : "128×128"}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Exchange Selector */}
                        <div className="space-y-1">
                            <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Target Exchange</label>
                            <div className="relative">
                                <button onClick={() => setIsExchangeDropdownOpen(!isExchangeDropdownOpen)}
                                    className={`w-full bg-black border ${isExchangeDropdownOpen ? 'border-indigo-500 ring-1 ring-indigo-500/50' : 'border-zinc-800'} rounded-xl p-3 text-sm text-left transition-all flex items-center justify-between`}>
                                    <span className={trainingExchange ? 'text-white font-medium' : 'text-zinc-500'}>
                                        {trainingExchange ? (
                                            <span className="flex items-center gap-2">
                                                <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                                                {trainingExchange}
                                                <span className="text-zinc-600 text-xs ml-1">({selectedExchangeCount} symbols)</span>
                                            </span>
                                        ) : "Select exchange..."}
                                    </span>
                                    <ChevronDown className={`w-4 h-4 text-zinc-500 transition-transform ${isExchangeDropdownOpen ? 'rotate-180' : ''}`} />
                                </button>
                                {isExchangeDropdownOpen && (
                                    <div className="absolute top-full mt-2 w-full bg-black border border-zinc-800 rounded-2xl shadow-xl z-50 max-h-64 overflow-y-auto">
                                        {exchangeOptions.map(inv => (
                                            <button key={inv.exchange}
                                                onClick={() => { setTrainingExchange(inv.exchange); setIsExchangeDropdownOpen(false); }}
                                                className={`w-full px-4 py-3 text-sm text-left hover:bg-zinc-900 flex items-center justify-between ${trainingExchange === inv.exchange ? 'text-indigo-400' : 'text-zinc-400'}`}>
                                                <span>{inv.exchange}</span>
                                                <span className="text-zinc-600 text-xs">{inv.priceCount} symbols</span>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Timesteps */}
                        <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Total Timesteps</label>
                                <select value={totalTimesteps} onChange={(e) => setTotalTimesteps(Number(e.target.value))}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500">
                                    <option value={50000}>50K (Quick)</option>
                                    <option value={100000}>100K</option>
                                    <option value={200000}>200K (Default)</option>
                                    <option value={500000}>500K</option>
                                    <option value={1000000}>1M (Deep)</option>
                                    <option value={2000000}>2M (Maximum)</option>
                                </select>
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Learning Rate</label>
                                <input type="number" min={0.00001} max={0.01} step={0.00001} value={learningRate}
                                    onChange={(e) => setLearningRate(Number(e.target.value) || 0.0003)}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                            </div>
                        </div>

                        {/* PPO-specific params */}
                        <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Batch Size</label>
                                <select value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500">
                                    <option value={32}>32</option>
                                    <option value={64}>64</option>
                                    <option value={128}>128</option>
                                    <option value={256}>256</option>
                                </select>
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">N Epochs</label>
                                <input type="number" min={1} max={30} value={nEpochs}
                                    onChange={(e) => setNEpochs(Number(e.target.value) || 10)}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-3">
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Gamma (γ)</label>
                                <input type="number" min={0.9} max={0.9999} step={0.001} value={gamma}
                                    onChange={(e) => setGamma(Number(e.target.value) || 0.99)}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Clip Range</label>
                                <input type="number" min={0.05} max={0.5} step={0.01} value={clipRange}
                                    onChange={(e) => setClipRange(Number(e.target.value) || 0.2)}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Entropy Coef</label>
                                <input type="number" min={0} max={0.1} step={0.001} value={entCoef}
                                    onChange={(e) => setEntCoef(Number(e.target.value) || 0.01)}
                                    className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                            </div>
                        </div>

                        {/* Environment Settings */}
                        <div className="pt-4 border-t border-zinc-800">
                            <span className="text-[10px] text-zinc-500 uppercase font-black">Environment Settings</span>
                            <div className="grid grid-cols-3 gap-3 mt-3">
                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Balance</label>
                                    <input type="number" min={1000} max={1000000} step={1000} value={initialBalance}
                                        onChange={(e) => setInitialBalance(Number(e.target.value) || 10000)}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500 outline-none" />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Max Steps</label>
                                    <select value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500">
                                        <option value={500}>500</option>
                                        <option value={1000}>1000</option>
                                        <option value={2000}>2000</option>
                                        <option value={5000}>5000</option>
                                    </select>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] text-zinc-500 uppercase font-black px-1">Reward</label>
                                    <select value={rewardMode} onChange={(e) => setRewardMode(e.target.value as any)}
                                        className="w-full h-11 rounded-xl border border-zinc-800 bg-black px-3 text-sm text-zinc-300 focus:border-indigo-500">
                                        <option value="pnl">PnL</option>
                                        <option value="sharpe">Sharpe</option>
                                        <option value="sortino">Sortino</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Estimated Impact */}
                        <div className="rounded-xl bg-zinc-900/50 border border-zinc-800 p-3 flex flex-col gap-2">
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] uppercase font-black text-zinc-500">Estimated Impact</span>
                                <span className={`text-[10px] font-bold ${estimatedTime.color}`}>{estimatedTime.descriptor}</span>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-zinc-300">
                                <div className="flex items-center gap-1.5">
                                    <Clock className="w-3 h-3 text-zinc-500" />
                                    <span>~{estimatedTime.minutes} min</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <Database className="w-3 h-3 text-zinc-500" />
                                    <span>{(totalTimesteps / 1000).toFixed(0)}K steps</span>
                                </div>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-3 pt-4">
                            <button onClick={startPPOTraining}
                                disabled={isTraining || triggeringTraining || !trainingExchange}
                                className="flex-1 py-4 rounded-2xl bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-[10px] font-black uppercase tracking-widest hover:opacity-90 transition-all flex items-center justify-center gap-2 disabled:opacity-50 shadow-[0_8px_25px_rgba(99,102,241,0.3)]">
                                {triggeringTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                                Start PPO Training
                            </button>
                            <button onClick={stopTraining}
                                disabled={!isTraining}
                                className="py-4 px-6 rounded-2xl bg-red-500/10 text-red-400 text-[10px] font-black uppercase tracking-widest hover:bg-red-500/20 transition-all flex items-center gap-2 disabled:opacity-30 border border-red-500/20">
                                <StopCircle className="w-4 h-4" />
                            </button>
                        </div>

                        {/* Live Status */}
                        {isTraining && (
                            <div className="p-4 rounded-2xl bg-indigo-500/5 border border-indigo-500/20 flex items-center gap-3">
                                <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                                <span className="text-xs font-black text-indigo-400">{trainingStatus?.last_message || "Training..."}</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right: Charts + Console */}
                <div className="lg:col-span-2 space-y-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Policy Loss Chart */}
                        <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800">
                            <h4 className="text-lg font-black text-white tracking-tight mb-1 flex items-center gap-2">
                                <TrendingDown className="w-5 h-5 text-rose-400" /> Policy Loss
                            </h4>
                            <p className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-6">Policy Gradient Loss Over Time</p>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={policyLossCurve}>
                                        <CartesianGrid strokeDasharray="8 8" vertical={false} stroke="rgba(255,255,255,0.03)" />
                                        <XAxis dataKey="step" hide />
                                        <YAxis hide domain={['auto', 'auto']} />
                                        <Tooltip contentStyle={{ backgroundColor: '#09090b', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '12px' }} itemStyle={{ color: '#f43f5e', fontWeight: '900', fontSize: '11px' }} />
                                        <Line type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2} dot={false} isAnimationActive={false} name="Policy Loss" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Reward Curve */}
                        <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800">
                            <h4 className="text-lg font-black text-white tracking-tight mb-1 flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-emerald-400" /> Mean Reward
                            </h4>
                            <p className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-6">Episode Reward Mean</p>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={rewardCurve}>
                                        <defs>
                                            <linearGradient id="rewardFill" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                                                <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="8 8" vertical={false} stroke="rgba(255,255,255,0.03)" />
                                        <XAxis dataKey="step" hide />
                                        <YAxis hide domain={['auto', 'auto']} />
                                        <Tooltip contentStyle={{ backgroundColor: '#09090b', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '12px' }} itemStyle={{ color: '#10b981', fontWeight: '900', fontSize: '11px' }} />
                                        <Area type="monotone" dataKey="reward" stroke="#10b981" strokeWidth={2} fill="url(#rewardFill)" isAnimationActive={false} name="Reward" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Value Loss Chart */}
                    <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800">
                        <h4 className="text-lg font-black text-white tracking-tight mb-1 flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-400" /> Value Network Loss
                        </h4>
                        <p className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-6">Critic Value Function Convergence</p>
                        <div className="h-[180px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={valueLossCurve}>
                                    <CartesianGrid strokeDasharray="8 8" vertical={false} stroke="rgba(255,255,255,0.03)" />
                                    <XAxis dataKey="step" hide />
                                    <YAxis hide domain={['auto', 'auto']} />
                                    <Tooltip contentStyle={{ backgroundColor: '#09090b', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '12px' }} itemStyle={{ color: '#3b82f6', fontWeight: '900', fontSize: '11px' }} />
                                    <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} name="Value Loss" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Live Training Console */}
                    <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800">
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="text-lg font-black text-white tracking-tight flex items-center gap-2">
                                <Database className="w-5 h-5 text-indigo-400" /> Training Console
                            </h4>
                            {isTraining && (
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-indigo-500/10 rounded-full border border-indigo-500/20">
                                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                                    <span className="text-[9px] font-black text-indigo-400 uppercase tracking-widest">Live</span>
                                </div>
                            )}
                        </div>
                        <div className="bg-black rounded-2xl border border-zinc-800 p-6 h-[260px] overflow-y-auto font-mono text-xs leading-relaxed">
                            {trainingLogs.length > 0 ? (
                                <div className="space-y-1">
                                    {trainingLogs.map((log, i) => (
                                        <div key={i} className="flex gap-3 opacity-70 hover:opacity-100 transition-opacity">
                                            <span className="text-zinc-700 select-none min-w-[3ch] text-right">{i + 1}</span>
                                            <span className={
                                                log.msg.includes('ERROR') || log.msg.includes('error') ? 'text-rose-400' :
                                                    log.msg.includes('Training') || log.msg.includes('training') ? 'text-indigo-400' :
                                                        log.msg.includes('complete') || log.msg.includes('saved') || log.msg.includes('done') ? 'text-emerald-400' :
                                                            'text-zinc-400'
                                            }>{log.msg}</span>
                                        </div>
                                    ))}
                                    <div ref={logEndRef} />
                                </div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center">
                                    <Loader2 className="w-8 h-8 mb-4 text-zinc-800 animate-spin" />
                                    <p className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-700">Awaiting PPO Training Session...</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
