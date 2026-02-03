"use client";

import { Database, Zap, TrendingUp, Beaker, Brain, LineChart } from "lucide-react";

interface AdminHeaderProps {
    activeMainTab: "data" | "ai" | "test" | "scan" | "backtest";
    setActiveMainTab: (tab: "data" | "ai" | "test" | "scan" | "backtest") => void;
}

export default function AdminHeader({ activeMainTab, setActiveMainTab }: AdminHeaderProps) {
    return (
        <header className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-black/80 backdrop-blur-md">
            <div className="max-w-[1600px] mx-auto px-4 md:px-8 h-20 flex items-center justify-between">
                <div className="flex items-center gap-8">
                    <div className="flex items-center gap-3 group">
                        <div className="p-2.5 rounded-2xl bg-indigo-600 shadow-[0_0_20px_rgba(99,102,241,0.4)] group-hover:scale-110 transition-transform duration-300">
                            <Database className="h-6 w-6 text-white" />
                        </div>
                        <div>
                            <span className="text-xl font-black tracking-tighter text-white">ADMIN<span className="text-indigo-500">PANEL</span></span>
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                                <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">System Online</span>
                            </div>
                        </div>
                    </div>

                    <nav className="hidden md:flex items-center bg-zinc-900/50 p-1.5 rounded-2xl border border-zinc-800/50 gap-1">
                        <button
                            onClick={() => setActiveMainTab("data")}
                            className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all duration-300 flex items-center gap-2 ${activeMainTab === "data" ? "bg-white text-black shadow-lg shadow-white/10" : "text-zinc-500 hover:text-white"}`}
                        >
                            <TrendingUp className="w-4 h-4" />
                            DATA MANAGER
                        </button>
                        <button
                            onClick={() => setActiveMainTab("ai")}
                            className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all duration-300 flex items-center gap-2 ${activeMainTab === "ai" ? "bg-white text-black shadow-lg shadow-white/10" : "text-zinc-500 hover:text-white"}`}
                        >
                            <Zap className="w-4 h-4" />
                            AI & AUTOMATION
                        </button>
                        <button
                            onClick={() => setActiveMainTab("test")}
                            className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all duration-300 flex items-center gap-2 ${activeMainTab === "test" ? "bg-white text-black shadow-lg shadow-white/10" : "text-zinc-500 hover:text-white"}`}
                        >
                            <Beaker className="w-4 h-4" />
                            TEST MODEL
                        </button>
                        <button
                            onClick={() => setActiveMainTab("scan")}
                            className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all duration-300 flex items-center gap-2 ${activeMainTab === "scan" ? "bg-white text-black shadow-lg shadow-white/10" : "text-zinc-500 hover:text-white"}`}
                        >
                            <Brain className="w-4 h-4" />
                            FAST SCAN
                        </button>
                        <button
                            onClick={() => setActiveMainTab("backtest")}
                            className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all duration-300 flex items-center gap-2 ${activeMainTab === "backtest" ? "bg-white text-black shadow-lg shadow-white/10" : "text-zinc-500 hover:text-white"}`}
                        >
                            <LineChart className="w-4 h-4" />
                            BACKTEST
                        </button>
                    </nav>
                </div>
                {/* Right side reserved for future admin metrics */}
                <div className="hidden lg:flex items-center gap-4 px-4 py-2 rounded-xl" />
            </div>
        </header>
    );
}
