import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getAiScore(precision: number): { score: number; label: string; color: string; bg: string } {
  const p = precision;
  if (p >= 0.9) return { score: 10, label: "Strong Buy", color: "text-emerald-500", bg: "bg-emerald-500/10" };
  if (p >= 0.85) return { score: 9, label: "Buy", color: "text-emerald-400", bg: "bg-emerald-400/10" };
  if (p >= 0.75) return { score: 8, label: "Buy", color: "text-emerald-400/80", bg: "bg-emerald-400/5" };
  if (p >= 0.7) return { score: 7, label: "Buy", color: "text-emerald-400/60", bg: "bg-emerald-400/5" };
  if (p >= 0.6) return { score: 6, label: "Neutral", color: "text-zinc-400", bg: "bg-zinc-400/10" };
  if (p >= 0.5) return { score: 5, label: "Neutral", color: "text-zinc-500", bg: "bg-zinc-500/10" };
  if (p >= 0.4) return { score: 4, label: "Weak", color: "text-orange-400", bg: "bg-orange-400/10" };
  if (p >= 0.3) return { score: 3, label: "Avoid", color: "text-red-400", bg: "bg-red-400/10" };
  return { score: 1, label: "Avoid", color: "text-red-600", bg: "bg-red-600/10" };
}