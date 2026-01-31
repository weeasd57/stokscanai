import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Sparkles } from "lucide-react";

interface StrategySettingsProps {
    targetPct: number;
    setTargetPct: (v: number) => void;
    stopLossPct: number;
    setStopLossPct: (v: number) => void;
    lookForwardDays: number;
    setLookForwardDays: (v: number) => void;
    buyThreshold: number;
    setBuyThreshold: (v: number) => void;
    asOfDate: string;
    setAsOfDate: (v: string) => void;
    isAutoDetected: boolean;
    setIsAutoDetected: (v: boolean) => void;
    useMultipleModels: boolean;
}

export default function StrategySettings({
    targetPct,
    setTargetPct,
    stopLossPct,
    setStopLossPct,
    lookForwardDays,
    setLookForwardDays,
    buyThreshold,
    setBuyThreshold,
    asOfDate,
    setAsOfDate,
    isAutoDetected,
    setIsAutoDetected,
    useMultipleModels,
}: StrategySettingsProps) {
    return (
        <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
            <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Sparkles className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" />
                    Strategy Window
                </div>
                {isAutoDetected && !useMultipleModels && (
                    <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                        Auto-Detected
                    </span>
                )}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="space-y-1">
                    <Label className="text-[9px] uppercase text-zinc-500">Target %</Label>
                    <Select value={String(targetPct ?? 0.15)} onValueChange={(v: string) => { setTargetPct(Number(v)); setIsAutoDetected(false); }}>
                        <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                            {[0.01, 0.05, 0.10, 0.15, 0.20, 0.30].map((v: number) => (
                                <SelectItem key={v} value={v.toString()}>{(v * 100).toFixed(0)}%</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-1">
                    <Label className="text-[9px] uppercase text-zinc-500">Stop %</Label>
                    <Select value={String(stopLossPct ?? 0.05)} onValueChange={(v: string) => { setStopLossPct(Number(v)); setIsAutoDetected(false); }}>
                        <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                            {[0.01, 0.03, 0.05, 0.07, 0.10].map((v: number) => (
                                <SelectItem key={v} value={v.toString()}>{(v * 100).toFixed(0)}%</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-1">
                    <Label className="text-[9px] uppercase text-zinc-500">Days</Label>
                    <Select value={String(lookForwardDays ?? 20)} onValueChange={(v: string) => { setLookForwardDays(Number(v)); setIsAutoDetected(false); }}>
                        <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                            {[10, 15, 20, 30].map((v: number) => (
                                <SelectItem key={v} value={v.toString()}>{v} Days</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-1">
                    <Label className="text-[9px] uppercase text-zinc-500">Sensitivity</Label>
                    <Select value={String(buyThreshold ?? 0.40)} onValueChange={(v: string) => { setBuyThreshold(Number(v)); setIsAutoDetected(false); }}>
                        <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                            {[0.10, 0.20, 0.30, 0.40, 0.45, 0.50].map((v: number) => (
                                <SelectItem key={v} value={v.toString()}>{(100 - v * 100).toFixed(0)}% (prob {v})</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                <div className="space-y-1">
                    <Label className="text-[9px] uppercase text-zinc-500">As Of Date</Label>
                    <Input
                        type="date"
                        value={asOfDate}
                        onChange={(e) => setAsOfDate(e.target.value)}
                        className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5"
                    />
                </div>
            </div>
            <p className="text-[9px] text-zinc-600 mt-3 leading-tight italic">
                {isAutoDetected && !useMultipleModels
                    ? "Values detected from model file. You can override them above."
                    : "Specify the strategy target/stop for calculating Precision and Earn metrics."}
            </p>
        </div>
    );
}
