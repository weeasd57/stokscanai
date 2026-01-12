"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";

type PricingPlan = {
  id: string;
  name: string;
  price_monthly_cents: number;
  features: Record<string, any>;
};

type Subscription = {
  plan_id: string;
  status: string;
  current_period_end: string | null;
};

export default function PricingPage() {
  const { user } = useAuth();
  const supabase = useMemo(() => createSupabaseBrowserClient(), []);

  const [plans, setPlans] = useState<PricingPlan[]>([]);
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      const { data: planRows } = await supabase
        .from("pricing_plans")
        .select("id, name, price_monthly_cents, features")
        .eq("is_active", true)
        .order("price_monthly_cents", { ascending: true });

      if (!cancelled) setPlans((planRows ?? []) as PricingPlan[]);

      if (user) {
        const { data: subRow } = await supabase
          .from("subscriptions")
          .select("plan_id, status, current_period_end")
          .eq("user_id", user.id)
          .maybeSingle();

        if (!cancelled) setSubscription((subRow ?? null) as any);
      } else {
        if (!cancelled) setSubscription(null);
      }

      if (!cancelled) setLoading(false);
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [supabase, user]);

  return (
    <div className="flex flex-col gap-8 pb-20">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight text-white">Pricing</h1>
        <p className="text-sm text-zinc-400">Choose a plan that fits your workflow.</p>
      </header>

      {user ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
          <div className="text-xs uppercase tracking-wide text-zinc-500">Current subscription</div>
          <div className="mt-1 text-sm text-zinc-200">
            {subscription ? (
              <span>
                Plan: <span className="font-semibold">{subscription.plan_id}</span> · Status:{" "}
                <span className="font-semibold">{subscription.status}</span>
              </span>
            ) : (
              <span className="text-zinc-400">No subscription record yet.</span>
            )}
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
          You&apos;re not logged in. <Link className="text-indigo-400 hover:text-indigo-300" href="/login">Login</Link> to see your subscription status.
        </div>
      )}

      {loading ? (
        <div className="text-sm text-zinc-500">Loading plans...</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {plans.map((p) => (
            <div key={p.id} className="rounded-2xl border border-zinc-800 bg-zinc-950 p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold text-zinc-100">{p.name}</div>
                  <div className="mt-1 text-xs text-zinc-500">Plan ID: {p.id}</div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-white">${(p.price_monthly_cents / 100).toFixed(2)}</div>
                  <div className="text-xs text-zinc-500">per month</div>
                </div>
              </div>

              <div className="mt-4 space-y-2 text-sm text-zinc-300">
                {Object.entries(p.features ?? {}).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between gap-4 rounded-lg border border-zinc-900 bg-zinc-950/50 px-3 py-2">
                    <span className="text-zinc-400">{k}</span>
                    <span className="font-mono text-zinc-100">{typeof v === "boolean" ? (v ? "true" : "false") : String(v)}</span>
                  </div>
                ))}
              </div>

              <div className="mt-5 text-xs text-zinc-500">
                Payments are not wired yet. This page reflects Supabase-stored plans/subscriptions.
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
