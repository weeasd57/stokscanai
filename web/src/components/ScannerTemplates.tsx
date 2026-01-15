"use client";

import React from "react";

export type ScannerTemplateId =
  | "ai_growth"
  | "macd_cross"
  | "rsi_oversold"
  | "volume_breakout"
  | "sma_200_breakout";

export interface ScannerTemplate {
  id: ScannerTemplateId;
  title: string; // Localize outside files if needed
  description: string; // Localize outside files if needed
  risk: "Low" | "Medium" | "High" | "Very High";
  gradient: string; // tailwind classes
  icon?: React.ReactNode;
}

export interface ScannerTemplatesProps {
  templates?: ScannerTemplate[];
  onSelect?: (id: ScannerTemplateId) => void;
}

const DEFAULT_TEMPLATES: ScannerTemplate[] = [
  {
    id: "ai_growth",
    title: "AI Smart Pick",
    description:
      "Stocks favored by the AI model (Random Forest trained on ~2 years of data).",
    risk: "Medium",
    gradient:
      "from-indigo-600 via-violet-600 to-purple-600 text-white",
  },
  {
    id: "macd_cross",
    title: "MACD Golden Cross",
    description:
      "Bullish when MACD line crosses above the signal line.",
    risk: "Medium",
    gradient: "from-emerald-600 to-green-600 text-white",
  },
  {
    id: "rsi_oversold",
    title: "RSI Oversold",
    description:
      "RSI < 30: potential rebound candidates.",
    risk: "High",
    gradient: "from-blue-600 to-cyan-500 text-white",
  },
  {
    id: "volume_breakout",
    title: "Volume Breakout",
    description:
      "Unusual high volume (> 2x average) suggests strong interest.",
    risk: "Very High",
    gradient: "from-orange-500 to-red-600 text-white",
  },
  {
    id: "sma_200_breakout",
    title: "Trend Breakout (SMA 200)",
    description:
      "Price closing above 200-day SMA implies a new uptrend.",
    risk: "Low",
    gradient: "from-slate-600 to-teal-600 text-white",
  },
];

export default function ScannerTemplates({
  templates = DEFAULT_TEMPLATES,
  onSelect,
}: ScannerTemplatesProps) {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-black tracking-tight">One-Click Stock Strategies</h2>
          <p className="text-xs text-zinc-500 font-medium">Pick a strategy to apply immediately.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {templates.map((tpl) => (
          <button
            key={tpl.id}
            onClick={() => onSelect?.(tpl.id)}
            className={`group relative overflow-hidden rounded-2xl p-5 text-left bg-gradient-to-br ${tpl.gradient} shadow-[0_0_24px_rgba(0,0,0,0.25)] hover:shadow-[0_0_36px_rgba(0,0,0,0.35)] transition-all`}
          >
            <div className="absolute inset-0 opacity-10 pointer-events-none bg-[radial-gradient(circle_at_20%_0%,white,transparent_40%)]" />
            <div className="relative flex flex-col gap-3">
              <div className="text-sm font-black drop-shadow">{tpl.title}</div>
              <div className="text-xs opacity-90 font-medium leading-relaxed">
                {tpl.description}
              </div>
              <div className="mt-2 inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-widest">
                <span className="px-2 py-0.5 rounded-md bg-black/20">Risk</span>
                <span className="px-2 py-0.5 rounded-md bg-black/20">{tpl.risk}</span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
