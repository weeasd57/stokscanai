"use client";

import { useState, useEffect, memo } from "react";
import { ExternalLink, Clock, Newspaper, Loader2, AlertCircle, LayoutDashboard } from "lucide-react";

import { fetchStockNews, type NewsArticle } from "@/lib/api";

interface NewsWidgetProps {
    symbol: string;
}

function NewsWidget({ symbol }: NewsWidgetProps) {
    const [articles, setArticles] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchNews() {
            setLoading(true);
            setError(null);
            try {
                const news = await fetchStockNews(symbol);
                setArticles(news);
            } catch (err) {
                console.error("News fetch error:", err);
                setError("Failed to load news articles");
            } finally {
                setLoading(false);
            }
        }

        void fetchNews();
    }, [symbol]);

    if (loading) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-orange-500/10 border border-orange-500/20">
                        <LayoutDashboard className="w-5 h-5 text-orange-400" />
                    </div>
                    <h3 className="text-xl font-black text-white uppercase tracking-tight italic">Market Intelligence</h3>
                </div>
                <div className="flex flex-col items-center justify-center p-20 border border-white/5 bg-zinc-950/40 rounded-[2.5rem] backdrop-blur-xl gap-4">
                    <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Fetching Market Intelligence...</p>
                </div>
            </div>
        );
    }

    if (error || articles.length === 0) {
        return null;
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-orange-500/10 border border-orange-500/20">
                    <LayoutDashboard className="w-5 h-5 text-orange-400" />
                </div>
                <h3 className="text-xl font-black text-white uppercase tracking-tight italic">Market Intelligence</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {articles.map((article, i) => (
                    <a
                        key={i}
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex flex-col p-6 rounded-[2rem] border border-white/5 bg-zinc-950/40 hover:border-indigo-500/30 hover:bg-indigo-600/5 transition-all duration-500 overflow-hidden relative"
                    >
                        <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                            <ExternalLink className="w-4 h-4 text-indigo-400" />
                        </div>

                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 rounded-xl bg-white/5 border border-white/5 group-hover:bg-indigo-600/20 group-hover:border-indigo-500/30 transition-all">
                                <Clock className="w-3.5 h-3.5 text-zinc-500 group-hover:text-indigo-400" />
                            </div>
                            <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest group-hover:text-indigo-400/60 transition-colors">
                                {new Date(article.publishedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                            </span>
                        </div>

                        <h4 className="text-lg font-bold text-white mb-3 group-hover:text-indigo-400 transition-colors line-clamp-2 leading-tight">
                            {article.title}
                        </h4>

                        <p className="text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors line-clamp-3 leading-relaxed mb-6">
                            {article.description}
                        </p>

                        <div className="mt-auto pt-6 border-t border-white/5 flex items-center justify-between">
                            <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                                {article.source.name}
                            </span>
                            <div className="text-[10px] font-black text-zinc-700 uppercase tracking-[0.2em] group-hover:text-white transition-colors">
                                Read Article
                            </div>
                        </div>
                    </a>
                ))}
            </div>
        </div>
    );
}

export default memo(NewsWidget);
