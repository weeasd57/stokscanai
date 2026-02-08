"use client";

interface AdminHeaderProps {
    activeMainTab: "data" | "ai" | "test" | "scan" | "backtest" | "bot";
    setActiveMainTab: (tab: "data" | "ai" | "test" | "scan" | "backtest" | "bot") => void;
}

export default function AdminHeader({ activeMainTab, setActiveMainTab }: AdminHeaderProps) {
    const tabs = [
        { id: "data", label: "DATA MANAGER" },
        { id: "ai", label: "AI & AUTOMATION" },
        { id: "test", label: "TEST MODEL" },
        { id: "scan", label: "FAST SCAN" },
        { id: "bot", label: "LIVE BOT" },
        { id: "backtest", label: "BACKTEST" },
    ] as const;

    return (
        <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-black/60 backdrop-blur-xl supports-[backdrop-filter]:bg-black/30">
            <div className="max-w-[1920px] mx-auto">
                <div className="flex flex-col md:flex-row md:items-center justify-start px-4 md:px-8 py-4 gap-4 md:gap-0">

                    {/* Navigation - Scrollable on mobile, Centered pills on desktop */}
                    <nav className="relative w-full md:flex-1 overflow-x-auto pb-2 md:pb-0 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
                        <div className="flex items-center gap-1.5 md:gap-2 w-full p-1.5 bg-zinc-900/50 border border-white/5 rounded-2xl backdrop-blur-sm">
                            {tabs.map((tab) => {
                                const isActive = activeMainTab === tab.id;

                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveMainTab(tab.id as any)}
                                        className={`
                                            relative px-4 md:px-6 py-2.5 rounded-xl text-[10px] md:text-xs font-black tracking-wider transition-all duration-300 flex items-center justify-center flex-1
                                            ${isActive
                                                ? "text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]"
                                                : "text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
                                            }
                                        `}
                                    >
                                        {isActive && (
                                            <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl -z-10" />
                                        )}
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </div>
                    </nav>

                </div>
            </div>

            <style jsx>{`
                .scrollbar-hide::-webkit-scrollbar {
                    display: none;
                }
                .scrollbar-hide {
                    -ms-overflow-style: none;
                    scrollbar-width: none;
                }
            `}</style>
        </header>
    );
}
