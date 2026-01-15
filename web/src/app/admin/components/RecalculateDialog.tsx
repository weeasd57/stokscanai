"use client";

import { useState, useEffect } from "react";
import { Zap, X, Loader2 } from "lucide-react";

interface RecalculateDialogProps {
    open: boolean;
    onClose: () => void;
    exchanges: { exchange: string, country: string, count: number }[];
    onRun: (exchange: string) => void;
    recalculating: boolean;
}

export default function RecalculateDialog({ open, onClose, exchanges, onRun, recalculating }: RecalculateDialogProps) {
    const [selectedExchange, setSelectedExchange] = useState<string | null>(null);

    useEffect(() => {
        if (!open) {
            setSelectedExchange(null);
        }
    }, [open]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-in fade-in duration-200">
            <div className="w-full max-w-2xl bg-zinc-950 border border-zinc-800 rounded-3xl shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
                <div className="p-6 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                            <Zap className="w-5 h-5" />
                        </div>
                        <div>
                            <h3 className="text-xl font-black text-white">Recalculate Technicals</h3>
                            <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">Select an exchange to recalculate all indicators</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 rounded-xl hover:bg-zinc-800 transition-colors">
                        <X className="w-5 h-5 text-zinc-500" />
                    </button>
                </div>

                <div className="flex-1 overflow-auto p-6 space-y-4 min-h-0">
                    {exchanges.length === 0 ? (
                        <div className="py-20 text-center opacity-30 italic text-sm">No exchanges with data found</div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                            {exchanges.map(ex => (
                                <button
                                    key={ex.exchange}
                                    onClick={() => setSelectedExchange(ex.exchange)}
                                    className={`p-4 rounded-xl border text-left transition-all ${selectedExchange === ex.exchange
                                        ? 'bg-indigo-600/20 border-indigo-500/50 ring-2 ring-indigo-500/30'
                                        : 'bg-zinc-900/40 border-zinc-800/50 hover:border-zinc-700'
                                        }`}
                                >
                                    <div className="font-bold text-sm text-zinc-100 flex justify-between items-center">
                                        {ex.exchange}
                                        {selectedExchange === ex.exchange && <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />}
                                    </div>
                                    <div className="text-[9px] text-zinc-500 font-semibold uppercase">{ex.country}</div>
                                    <div className="mt-2 text-xs text-indigo-400 font-mono">{ex.count} symbols</div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="p-6 border-t border-zinc-800 bg-zinc-900/50 flex items-center justify-between">
                    <div className="text-[10px] font-bold text-zinc-500 uppercase">
                        {selectedExchange ? `Selected: ${selectedExchange}` : 'No exchange selected'}
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-6 py-2.5 rounded-xl border border-zinc-800 text-xs font-bold text-zinc-500 hover:bg-zinc-900 transition-all"
                        >
                            CANCEL
                        </button>
                        <button
                            onClick={() => selectedExchange && onRun(selectedExchange)}
                            disabled={!selectedExchange || recalculating}
                            className="px-8 py-2.5 rounded-xl bg-indigo-600 text-xs font-black text-white hover:bg-indigo-500 disabled:opacity-50 transition-all flex items-center gap-2"
                        >
                            {recalculating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                            RECALCULATE ALL
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
