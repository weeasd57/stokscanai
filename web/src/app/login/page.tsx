"use client";

import { useMemo, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";

export default function LoginPage() {
  const router = useRouter();
  const { signIn, user, loading } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = useMemo(() => email.trim().length > 3 && password.length >= 6 && !submitting, [email, password, submitting]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await signIn(email.trim(), password);
      if (res.error) {
        setError(res.error);
        return;
      }
      router.push("/profile");
    } finally {
      setSubmitting(false);
    }
  }

  if (!loading && user) {
    router.replace("/profile");
    return null;
  }

  return (
    <div className="mx-auto max-w-md">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-6">
        <h1 className="text-2xl font-semibold text-zinc-100">Login</h1>
        <p className="mt-1 text-sm text-zinc-400">Sign in to sync your watchlist and settings.</p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-medium text-zinc-400">Email</label>
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              type="email"
              autoComplete="email"
              className="h-10 w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none focus:border-indigo-500"
              placeholder="you@example.com"
              required
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-zinc-400">Password</label>
            <input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              type="password"
              autoComplete="current-password"
              className="h-10 w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none focus:border-indigo-500"
              placeholder="••••••••"
              required
            />
          </div>

          {error && <div className="rounded-lg border border-red-900/40 bg-red-950/30 p-3 text-xs text-red-300">{error}</div>}

          <button
            type="submit"
            disabled={!canSubmit}
            className="h-10 w-full rounded-lg bg-indigo-600 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600"
          >
            {submitting ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div className="mt-4 text-xs text-zinc-500">
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="text-indigo-400 hover:text-indigo-300">
            Create one
          </Link>
        </div>
      </div>
    </div>
  );
}
