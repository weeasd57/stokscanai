"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { BookOpen, Calendar, User, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function BlogsPage() {
    const { t } = useLanguage();

    const posts = [
        {
            title: "How AI is Revolutionizing Stock Market Predictions",
            excerpt: "Explore the internal workings of RandomForest models and how they identify non-linear patterns in market data...",
            date: "May 15, 2026",
            author: "Dr. Analyst",
            category: "AI & Tech"
        },
        {
            title: "Understanding Technical Indicators in the Modern Era",
            excerpt: "RSI, MACD, and Bollinger Bands are classic, but are they still relevant when combined with neural networks?",
            date: "May 10, 2026",
            author: "Market Guru",
            category: "Analysis"
        },
        {
            title: "Top 5 AI Stocks to Watch for Q3 2026",
            excerpt: "Our models have flagged these five giants as potentially undervalued based on fundamental and sentiment analysis...",
            date: "May 05, 2026",
            author: "Investment Team",
            category: "Top Picks"
        }
    ];

    return (
        <div className="flex flex-col gap-12 pb-20 pt-10">
            <header className="space-y-4 max-w-2xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-black uppercase tracking-widest">
                    <BookOpen className="w-3 h-3" />
                    Market Blogs & Insights
                </div>
                <h1 className="text-5xl font-black tracking-tighter text-white uppercase italic">
                    Expert <span className="text-indigo-500">Analysis</span>
                </h1>
                <p className="text-zinc-500 text-lg leading-relaxed">
                    Deep dives into market trends, algorithmic strategies, and the future of AI-driven finance.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {posts.map((post, i) => (
                    <div key={i} className="group relative flex flex-col p-8 rounded-[2.5rem] border border-white/5 bg-zinc-950/40 hover:border-white/10 transition-all duration-500">
                        <div className="flex items-center justify-between mb-6">
                            <span className="text-[10px] font-black text-indigo-500 uppercase tracking-widest">{post.category}</span>
                            <div className="flex items-center gap-2 text-[10px] text-zinc-600 font-bold uppercase tracking-widest">
                                <Calendar className="w-3 h-3" />
                                {post.date}
                            </div>
                        </div>

                        <h2 className="text-xl font-bold text-white mb-4 group-hover:text-indigo-400 transition-colors leading-tight">
                            {post.title}
                        </h2>

                        <p className="text-sm text-zinc-500 mb-8 flex-1 leading-relaxed">
                            {post.excerpt}
                        </p>

                        <div className="flex items-center justify-between mt-auto pt-6 border-t border-white/5">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center text-[10px] font-bold text-zinc-400">
                                    {post.author[0]}
                                </div>
                                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{post.author}</span>
                            </div>
                            <Link href="#" className="p-2 rounded-xl bg-white/5 text-white hover:bg-indigo-600 transition-all flex items-center justify-center group/btn">
                                <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                            </Link>
                        </div>
                    </div>
                ))}
            </div>

            <section className="mt-12 p-12 rounded-[3rem] border border-white/5 bg-indigo-600/5 relative overflow-hidden text-center space-y-6">
                <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-600/10 blur-[100px] -z-10 rounded-full" />
                <h2 className="text-2xl font-black text-white italic uppercase">Stay Updated</h2>
                <p className="text-zinc-500 max-w-lg mx-auto text-sm leading-relaxed">
                    Subscribe to our newsletter to receive the latest market insights and algorithmic predictions directly in your inbox.
                </p>
                <div className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto">
                    <input
                        type="email"
                        placeholder="your@email.com"
                        className="flex-1 h-12 rounded-2xl bg-zinc-950/50 border border-white/5 px-4 text-sm text-white outline-none focus:border-indigo-500/50 transition-all"
                    />
                    <button className="h-12 px-8 rounded-2xl bg-indigo-600 text-white text-[11px] font-black uppercase tracking-widest hover:bg-indigo-500 transition-all shadow-xl shadow-indigo-600/20 active:scale-95">
                        Subscribe
                    </button>
                </div>
            </section>
        </div>
    );
}
