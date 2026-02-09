"use client";

import Link from "next/link";
import { useLanguage } from "@/contexts/LanguageContext";

export default function Footer() {
    const { t } = useLanguage();

    return (
        <footer className="w-full border-t border-white/5 bg-zinc-950/40 backdrop-blur-xl py-12 mt-20">
            <div className="mx-auto max-w-5xl px-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
                    <div className="col-span-1 md:col-span-2 space-y-4">
                        <h3 className="text-xl font-black text-white italic tracking-tighter uppercase">
                            {t("app.title")}
                        </h3>
                        <p className="text-sm text-zinc-500 max-w-xs leading-relaxed">
                            Advanced AI-driven stock analysis platform. Combining RandomForest models with multi-source fundamentals to give you the edge.
                        </p>
                    </div>

                    <div className="space-y-4">
                        <h4 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Platform</h4>
                        <ul className="space-y-2">
                            <li><Link href="/" className="text-sm text-zinc-500 hover:text-indigo-400 transition-colors">{t("nav.home")}</Link></li>
                            <li><Link href="/scanner/ai" className="text-sm text-zinc-500 hover:text-indigo-400 transition-colors">{t("nav.scanner.ai")}</Link></li>
                            <li><Link href="/scanner/technical" className="text-sm text-zinc-500 hover:text-indigo-400 transition-colors">{t("nav.scanner.tech")}</Link></li>
                            <li><Link href="/pro" className="text-sm text-zinc-500 hover:text-indigo-400 transition-colors font-bold text-indigo-400/80">Pro Dashboard</Link></li>
                        </ul>
                    </div>

                    <div className="space-y-4">
                        <h4 className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Resources</h4>
                        <ul className="space-y-2">
                            <li><Link href="/blogs" className="text-sm text-zinc-500 hover:text-indigo-400 transition-colors">Market Blogs</Link></li>
                        </ul>
                    </div>
                </div>

                <div className="border-t border-white/5 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-center md:text-left">
                    <p className="text-[10px] font-black text-zinc-700 uppercase tracking-widest">
                        © 2026 Artoro. Built for professional analysis.
                    </p>
                </div>

                <p className="mt-8 text-[9px] font-bold text-zinc-800 text-center uppercase tracking-widest leading-relaxed">
                    {t("home.footer.disclaimer")}
                </p>
            </div>
        </footer>
    );
}
