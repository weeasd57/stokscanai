"use client";

import { useState, useEffect } from "react";
import { Check, Zap, Star, Shield, Crown, Loader2, CreditCard } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAuth } from "@/contexts/AuthContext";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";
import Link from "next/link";
import { useMemo } from "react";

interface Plan {
    id: string; // from pricing_plans
    name: string;
    price: number | string;
    period?: string;
    desc: string;
    features: string[] | Record<string, any>;
    featured?: boolean;
    button_text?: string;
    current?: boolean;
}

interface Subscription {
    plan_id: string;
    status: string;
    current_period_end: string | null;
}

export default function ProPage() {
    const { t } = useLanguage();
    const { user } = useAuth();
    const supabase = useMemo(() => createSupabaseBrowserClient(), []);

    const [plans, setPlans] = useState<Plan[]>([]);
    const [subscription, setSubscription] = useState<Subscription | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;

        async function loadData() {
            setLoading(true);
            try {
                // 1. Fetch plans (from API or direct from Supabase)
                // We'll keep the API call as it likely returns the nicely formatted data,
                // but we also need the plan IDs from Supabase table pricing_plans if we want to match subscriptions.
                const res = await fetch("/api/admin/plans");
                const data = await res.json();

                if (Array.isArray(data) && !cancelled) {
                    setPlans(data);
                }

                // 2. Fetch subscription if user is logged in
                if (user) {
                    const { data: subRow } = await supabase
                        .from("subscriptions")
                        .select("plan_id, status, current_period_end")
                        .eq("user_id", user.id)
                        .maybeSingle();

                    if (!cancelled) setSubscription((subRow ?? null) as any);
                }
            } catch (err) {
                console.error("Failed to fetch pro data:", err);
                if (!cancelled) setError("Failed to load pro dashboard");
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        void loadData();
        return () => { cancelled = true; };
    }, [user, supabase]);

    const getIcon = (name: string) => {
        switch (name?.toLowerCase()) {
            case 'free': return <Star className="w-6 h-6 text-zinc-400" />;
            case 'pro': return <Zap className="w-6 h-6 text-indigo-400" />;
            case 'enterprise': return <Crown className="w-6 h-6 text-amber-400" />;
            default: return <Star className="w-6 h-6 text-zinc-400" />;
        }
    };

    if (loading && plans.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
                <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
                <p className="text-zinc-500 font-bold uppercase tracking-widest text-[10px]">Synchronizing Plans...</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-16 pb-20 max-w-[1200px] mx-auto pt-10">
            <header className="text-center space-y-4">
                <h1 className="text-5xl font-black tracking-tighter text-white uppercase italic">
                    Upgrade to <span className="text-indigo-500">Pro</span>
                </h1>
                <p className="text-zinc-400 max-w-2xl mx-auto text-lg">
                    Unlock the full power of AI-driven market analysis and stay ahead of the curve.
                </p>

                {/* Subscription Status Banner */}
                <div className="pt-6">
                    {user ? (
                        <div className="inline-flex flex-col items-center gap-2 p-6 rounded-[2rem] border border-white/5 bg-zinc-900/30 backdrop-blur-xl animate-in fade-in slide-in-from-top-4 duration-500">
                            <div className="flex items-center gap-3 text-zinc-500 text-[10px] font-black uppercase tracking-[0.2em]">
                                <CreditCard className="w-4 h-4 text-indigo-500" />
                                Your Subscription
                            </div>
                            <div className="text-sm text-zinc-200 font-bold">
                                {subscription ? (
                                    <div className="flex items-center gap-3">
                                        <span className="px-3 py-1 rounded-full bg-indigo-500 text-white text-[10px] uppercase font-black">
                                            {subscription.plan_id}
                                        </span>
                                        <span className="text-zinc-500 font-medium tracking-widest uppercase text-[10px]">
                                            Status: {subscription.status}
                                        </span>
                                    </div>
                                ) : (
                                    <span className="text-zinc-600 italic">No active subscription found.</span>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="inline-flex items-center gap-4 p-4 px-6 rounded-2xl border border-white/5 bg-zinc-900/30 text-xs font-bold text-zinc-400">
                            Login into your account to manage subscriptions
                            <Link href="/login" className="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 transition-all">
                                Login
                            </Link>
                        </div>
                    )}
                </div>
            </header>

            {error && (
                <div className="p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-500 text-center max-w-md mx-auto text-sm font-bold animate-in fade-in zoom-in duration-300">
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 px-4">
                {plans.map((plan, i) => (
                    <div
                        key={i}
                        className={`
              relative flex flex-col p-8 rounded-[3rem] border transition-all duration-500
              ${plan.featured
                                ? "bg-gradient-to-br from-indigo-950/40 via-zinc-950/40 to-zinc-950 border-indigo-500/30 shadow-2xl shadow-indigo-600/10 scale-105 z-10"
                                : "bg-zinc-950/40 border-white/5 hover:border-white/10"
                            }
            `}
                    >
                        {plan.featured && (
                            <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-indigo-600 text-[10px] font-black uppercase tracking-widest rounded-full text-white shadow-lg shadow-indigo-600/40">
                                Most Popular
                            </div>
                        )}

                        <div className="flex items-center justify-between mb-8">
                            <div className={`p-4 rounded-2xl ${plan.featured ? "bg-indigo-600/20" : "bg-zinc-900/50"}`}>
                                {getIcon(plan.name)}
                            </div>
                            <div className="text-right">
                                <div className="text-3xl font-black text-white">${plan.price}</div>
                                <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                                    {plan.period === 'forever' ? 'forever' : `/${plan.period || 'month'}`}
                                </div>
                            </div>
                        </div>

                        <h2 className="text-2xl font-black text-white mb-2">{plan.name}</h2>
                        <p className="text-sm text-zinc-500 mb-8">{plan.desc}</p>

                        <ul className="space-y-4 flex-1 mb-10">
                            {(Array.isArray(plan.features) ? plan.features : Object.keys(plan.features)).map((feat, j) => (
                                <li key={j} className="flex items-start gap-3 text-sm text-zinc-300">
                                    <div className={`mt-0.5 p-0.5 rounded-full ${plan.featured ? "bg-indigo-500/20 text-indigo-400" : "bg-zinc-800 text-zinc-500"}`}>
                                        <Check className="w-3.5 h-3.5" />
                                    </div>
                                    <span>{typeof feat === 'string' ? feat : String(feat)}</span>
                                </li>
                            ))}
                        </ul>

                        <button
                            className={`
                w-full py-4 rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] transition-all
                ${plan.featured
                                    ? "bg-indigo-600 text-white hover:bg-indigo-500 shadow-xl shadow-indigo-600/20 active:scale-95"
                                    : "bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 active:scale-95"
                                }
                ${subscription?.plan_id === plan.name ? "opacity-30 cursor-default" : ""}
              `}
                        >
                            {subscription?.plan_id === plan.name ? "Current Plan" : (plan.button_text || "Choose Plan")}
                        </button>
                    </div>
                ))}
            </div>

            <section className="mt-10 rounded-[3rem] border border-white/5 bg-zinc-950/40 p-12 text-center space-y-8">
                <div className="flex justify-center gap-4">
                    <Shield className="w-12 h-12 text-emerald-500" />
                </div>
                <div className="space-y-2">
                    <h2 className="text-2xl font-black text-white uppercase italic">Secure & Trusted</h2>
                    <p className="text-zinc-500 max-w-xl mx-auto text-sm">
                        We use industry-standard encryption and security protocols to ensure your data and payments are always safe. No hidden fees, cancel anytime.
                    </p>
                    <p className="text-[10px] text-zinc-800 font-bold uppercase tracking-[0.3em] mt-4">
                        Payments are not wired yet. This dashboard reflects stored plans and subscriptions.
                    </p>
                </div>
            </section>
        </div>
    );
}
