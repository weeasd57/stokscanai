"use client";

import { BarChart3, Database, Zap, TrendingUp } from "lucide-react";

interface AdminHeaderProps {
    activeMainTab: "data" | "ai";
    setActiveMainTab: (tab: "data" | "ai") => void;
    usage: { used: number, limit: number, extraLeft: number } | null;
}

export default function AdminHeader({ activeMainTab, setActiveMainTab, usage }: AdminHeaderProps) {
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

                    <nav className="hidden md:flex items-center bg-zinc-900/50 p-1.5 rounded-2xl border border-zinc-800/50">
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
                    </nav>
                </div>
                {/* Usage Stats (Condensed) */}
                <div className="hidden lg:flex items-center gap-4 px-4 py-2 rounded-xl bg-zinc-900/30 border border-zinc-800/50">
                    <div className="flex flex-col gap-1 min-w-[120px]">
                        <div className="flex justify-between text-[9px] font-bold text-zinc-500 uppercase">
                            <span>API Usage</span>
                            <span className="text-indigo-400 font-mono">{usage?.used || 0} / {usage?.limit || 1000}</span>
                        </div>
                        <div className="h-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-indigo-500 transition-all"
                                style={{ width: `${Math.min(100, ((usage?.used || 0) / (usage?.limit || 1)) * 100)}%` }}
                            />
                        </div>
                    </div>
                    <BarChart3 className="w-4 h-4 text-zinc-600" />
                </div>
            </div>
        </header>
    );
}
