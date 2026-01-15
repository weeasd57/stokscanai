"use client";

import { Cloud, Server, Clock, Zap, ExternalLink, ShieldCheck, Database, BarChart3, Info } from "lucide-react";

export default function ControlPanel() {
    const deploymentSteps = [
        {
            title: "Web Service (FastAPI)",
            icon: <Server className="w-5 h-5 text-indigo-400" />,
            desc: "Render Dashboard → New → Web Service",
            details: [
                "Root Directory: api",
                "Build Command: pip install -r requirements.txt",
                "Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT"
            ],
            env: ["EODHD_API_KEY", "NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
        },
        {
            title: "Daily Price Update",
            icon: <Clock className="w-5 h-5 text-emerald-400" />,
            desc: "Render Dashboard → New → Cron Job",
            details: [
                "Command: python smart_update.py --exchange US --days 365 --prices --funds",
                "Schedule: 0 3 * * * (Daily at 3 AM)"
            ]
        },
        {
            title: "Daily AI Training",
            icon: <Zap className="w-5 h-5 text-amber-400" />,
            desc: "Render Dashboard → New → Cron Job",
            details: [
                "Command: python train_exchange_model.py --exchange US",
                "Schedule: 0 4 * * * (Daily at 4 AM)"
            ]
        }
    ];

    return (
        <div className="p-4 md:p-8 max-w-[1600px] mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Header info */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 bg-zinc-900/30 border border-zinc-800/50 p-8 rounded-3xl backdrop-blur-md">
                <div className="space-y-2">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 rounded-2xl bg-indigo-500/10 border border-indigo-500/20">
                            <BarChart3 className="w-6 h-6 text-indigo-400" />
                        </div>
                        <h2 className="text-3xl font-black text-white tracking-tight">System Control</h2>
                    </div>
                    <p className="text-zinc-500 font-medium max-w-xl">
                        Manage your deployment settings, automation pipelines, and infrastructure configuration for Render and external services.
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="px-5 py-3 rounded-2xl bg-zinc-950 border border-zinc-800 flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                        <span className="text-[10px] text-zinc-400 font-black uppercase tracking-widest">Render: Connected</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {deploymentSteps.map((step, idx) => (
                    <div key={idx} className="group bg-zinc-900/20 border border-zinc-800 hover:border-zinc-700 rounded-3xl p-6 transition-all duration-300 hover:shadow-2xl hover:shadow-indigo-500/5 hover:-translate-y-1">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 rounded-2xl bg-zinc-950 border border-zinc-800 group-hover:bg-zinc-900 transition-colors">
                                {step.icon}
                            </div>
                            <ExternalLink className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                        </div>
                        <h3 className="text-lg font-black text-white mb-2 tracking-tight">{step.title}</h3>
                        <p className="text-xs text-indigo-400 font-bold mb-6">{step.desc}</p>

                        <div className="space-y-3">
                            <label className="text-[10px] text-zinc-500 uppercase font-black tracking-widest px-1">Configuration</label>
                            <div className="bg-black/40 rounded-2xl p-4 border border-zinc-800/50 space-y-3">
                                {step.details.map((detail, dIdx) => (
                                    <div key={dIdx} className="flex gap-3">
                                        <div className="mt-1 w-1 h-1 rounded-full bg-zinc-700 shrink-0"></div>
                                        <p className="text-[11px] text-zinc-400 font-mono break-all">{detail}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {step.env && (
                            <div className="mt-6 space-y-3">
                                <label className="text-[10px] text-zinc-500 uppercase font-black tracking-widest px-1">Required Env Vars</label>
                                <div className="flex flex-wrap gap-2">
                                    {step.env.map((ev, eIdx) => (
                                        <span key={eIdx} className="px-2.5 py-1 rounded-lg bg-zinc-950 border border-zinc-800 text-[9px] text-zinc-500 font-mono">
                                            {ev}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Recommendations Section */}
            <div className="bg-indigo-600/5 border border-indigo-500/20 rounded-3xl p-8">
                <div className="flex items-start gap-4">
                    <div className="p-3 rounded-2xl bg-indigo-500/10 border border-indigo-500/20">
                        <ShieldCheck className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div className="space-y-4">
                        <h3 className="text-xl font-black text-white tracking-tight">Best Practices & Strategy</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-indigo-400 font-bold uppercase text-[10px] tracking-widest">
                                    <Clock className="w-3.5 h-3.5" />
                                    Optimized Schedule
                                </div>
                                <p className="text-sm text-zinc-400 leading-relaxed">
                                    Price updates are best run daily after market close. AI Model training can be scheduled weekly for better efficiency if data volume is high.
                                </p>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center gap-2 text-emerald-400 font-bold uppercase text-[10px] tracking-widest">
                                    <Database className="w-3.5 h-3.5" />
                                    Storage Strategy
                                </div>
                                <p className="text-sm text-zinc-400 leading-relaxed">
                                    The system automatically persists trained models to Supabase Storage. The API layer always fetches the latest stable model for inference.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Warning Message */}
            <div className="flex items-center gap-4 bg-amber-500/5 border border-amber-500/20 p-6 rounded-2xl">
                <Info className="w-5 h-5 text-amber-500 shrink-0" />
                <p className="text-xs text-amber-500/80 font-medium">
                    Important: Ensure all Environment Variables in your Render Cron Jobs match exactly with your Web Service configuration to avoid connectivity failures.
                </p>
            </div>
        </div>
    );
}
