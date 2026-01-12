"use client";

import { useState, useEffect } from "react";
import { X, Target, ShieldAlert } from "lucide-react";

interface TargetStopModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (target: number, stop: number) => void;
    defaultTarget?: number;
    defaultStop?: number;
    symbolName?: string;
}

export default function TargetStopModal({
    isOpen,
    onClose,
    onConfirm,
    defaultTarget = 5,
    defaultStop = 2,
    symbolName
}: TargetStopModalProps) {
    const [target, setTarget] = useState(defaultTarget.toString());
    const [stop, setStop] = useState(defaultStop.toString());

    useEffect(() => {
        if (isOpen) {
            setTarget(defaultTarget.toString());
            setStop(defaultStop.toString());
        }
    }, [isOpen, defaultTarget, defaultStop]);

    const handleConfirm = () => {
        const t = parseFloat(target);
        const s = parseFloat(stop);
        if (isNaN(t) || t <= 0 || t > 100) return;
        if (isNaN(s) || s <= 0 || s > 100) return;
        onConfirm(t, s);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[300] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

            <div className="relative w-full max-w-md bg-zinc-950/90 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Glow effect */}
                <div className="absolute -top-20 -right-20 w-40 h-40 bg-indigo-600/20 blur-[80px] rounded-full" />
                <div className="absolute -bottom-20 -left-20 w-40 h-40 bg-purple-600/20 blur-[80px] rounded-full" />

                {/* Header */}
                <div className="relative p-6 border-b border-white/5">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-xl font-black text-white uppercase tracking-tight">Set Targets</h3>
                            {symbolName && (
                                <p className="text-xs text-indigo-400 font-mono mt-1">{symbolName}</p>
                            )}
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 rounded-xl text-zinc-500 hover:text-white hover:bg-white/5 transition-all"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Body */}
                <div className="relative p-6 space-y-6">
                    {/* Target Input */}
                    <div className="space-y-2">
                        <label className="flex items-center gap-2 text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em]">
                            <Target className="w-3.5 h-3.5 text-emerald-500" />
                            Target Gain %
                        </label>
                        <div className="relative group">
                            <input
                                type="number"
                                value={target}
                                onChange={(e) => setTarget(e.target.value)}
                                className="w-full h-14 px-5 rounded-2xl border border-white/10 bg-zinc-900/50 text-xl font-black text-emerald-400 outline-none focus:ring-2 focus:ring-emerald-500/30 transition-all font-mono"
                                placeholder="5"
                                min="0.1"
                                max="100"
                                step="0.5"
                            />
                            <span className="absolute right-5 top-1/2 -translate-y-1/2 font-black text-zinc-600">%</span>
                        </div>
                    </div>

                    {/* Stop Loss Input */}
                    <div className="space-y-2">
                        <label className="flex items-center gap-2 text-[10px] font-black text-zinc-500 uppercase tracking-[0.2em]">
                            <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                            Stop Loss %
                        </label>
                        <div className="relative group">
                            <input
                                type="number"
                                value={stop}
                                onChange={(e) => setStop(e.target.value)}
                                className="w-full h-14 px-5 rounded-2xl border border-white/10 bg-zinc-900/50 text-xl font-black text-red-400 outline-none focus:ring-2 focus:ring-red-500/30 transition-all font-mono"
                                placeholder="2"
                                min="0.1"
                                max="100"
                                step="0.5"
                            />
                            <span className="absolute right-5 top-1/2 -translate-y-1/2 font-black text-zinc-600">%</span>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="relative p-6 pt-0 flex gap-3">
                    <button
                        onClick={onClose}
                        className="flex-1 h-12 rounded-2xl border border-white/10 text-zinc-400 font-black text-[10px] uppercase tracking-[0.2em] hover:bg-white/5 transition-all"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleConfirm}
                        className="flex-1 h-12 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-black text-[10px] uppercase tracking-[0.2em] hover:opacity-90 transition-all shadow-xl shadow-indigo-600/20"
                    >
                        Confirm
                    </button>
                </div>
            </div>
        </div>
    );
}
