"use client";

import { ArrowUpRight, Shield, Sparkles } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

export type ScannerTemplateId =
  | "ai_growth"
  | "macd_cross"
  | "rsi_oversold"
  | "volume_breakout"
  | "sma_200_breakout";

const templates: Array<{
  id: ScannerTemplateId;
  titleKey: string;
  descKey: string;
  riskKey: string;
  gradient: string;
  accent: string;
}> = [
  {
    id: "ai_growth",
    titleKey: "scanner.templates.ai_growth.title",
    descKey: "scanner.templates.ai_growth.desc",
    riskKey: "scanner.templates.risk.medium",
    gradient: "from-indigo-500/20 via-purple-500/10 to-transparent",
    accent: "text-indigo-300",
  },
  {
    id: "macd_cross",
    titleKey: "scanner.templates.macd_cross.title",
    descKey: "scanner.templates.macd_cross.desc",
    riskKey: "scanner.templates.risk.medium",
    gradient: "from-emerald-500/20 via-teal-500/10 to-transparent",
    accent: "text-emerald-300",
  },
  {
    id: "rsi_oversold",
    titleKey: "scanner.templates.rsi_oversold.title",
    descKey: "scanner.templates.rsi_oversold.desc",
    riskKey: "scanner.templates.risk.high",
    gradient: "from-sky-500/20 via-cyan-500/10 to-transparent",
    accent: "text-sky-300",
  },
  {
    id: "volume_breakout",
    titleKey: "scanner.templates.volume_breakout.title",
    descKey: "scanner.templates.volume_breakout.desc",
    riskKey: "scanner.templates.risk.very_high",
    gradient: "from-orange-500/20 via-rose-500/10 to-transparent",
    accent: "text-orange-300",
  },
  {
    id: "sma_200_breakout",
    titleKey: "scanner.templates.sma_200_breakout.title",
    descKey: "scanner.templates.sma_200_breakout.desc",
    riskKey: "scanner.templates.risk.low",
    gradient: "from-slate-500/20 via-teal-500/10 to-transparent",
    accent: "text-teal-300",
  },
];

type ScannerTemplatesProps = {
  onSelect?: (id: ScannerTemplateId) => void;
};

export default function ScannerTemplates({ onSelect }: ScannerTemplatesProps) {
  const { t } = useLanguage();

  return (
    <section className="rounded-[2.5rem] border border-white/5 bg-zinc-950/40 backdrop-blur-xl p-8 shadow-2xl relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.12),_transparent_55%)]" />
      <div className="relative z-10 flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.35em] text-zinc-500 font-black">
            <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
            {t("scanner.templates.kicker")}
          </div>
          <h2 className="text-2xl md:text-3xl font-black tracking-tight text-white">
            {t("scanner.templates.title")}
          </h2>
          <p className="text-sm text-zinc-500 max-w-2xl">
            {t("scanner.templates.subtitle")}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {templates.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect?.(item.id)}
              className="group relative rounded-3xl border border-white/5 bg-zinc-950/60 p-6 overflow-hidden transition-all hover:border-white/10 text-left"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${item.gradient}`} />
              <div className="relative z-10 flex flex-col gap-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className={`text-lg font-black ${item.accent}`}>
                      {t(item.titleKey)}
                    </h3>
                    <p className="mt-2 text-sm text-zinc-400 leading-relaxed">
                      {t(item.descKey)}
                    </p>
                  </div>
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-zinc-900/60 text-zinc-400 group-hover:text-white transition-colors">
                    <ArrowUpRight className="h-4 w-4" />
                  </div>
                </div>

                <div className="mt-auto flex items-center justify-between text-[10px] uppercase tracking-[0.3em] text-zinc-500 font-black">
                  <div className="flex items-center gap-2">
                    <Shield className="h-3.5 w-3.5 text-zinc-400" />
                    {t(item.riskKey)}
                  </div>
                  <span className="text-zinc-600">#{item.id}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
