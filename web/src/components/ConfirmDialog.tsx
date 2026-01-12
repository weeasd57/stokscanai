"use client";

import { X, AlertTriangle, Loader2 } from "lucide-react";
import { useEffect } from "react";

interface ConfirmDialogProps {
    isOpen: boolean;
    title: string;
    message: string;
    onClose: () => void;
    onConfirm: () => void;
    isLoading?: boolean;
    confirmLabel?: string;
    cancelLabel?: string;
    variant?: "danger" | "info" | "success";
}

export default function ConfirmDialog({
    isOpen,
    title,
    message,
    onClose,
    onConfirm,
    isLoading = false,
    confirmLabel = "Confirm",
    cancelLabel = "Cancel",
    variant = "danger"
}: ConfirmDialogProps) {

    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        if (isOpen) window.addEventListener("keydown", handleEsc);
        return () => window.removeEventListener("keydown", handleEsc);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    const variantStyles = {
        danger: "bg-red-500 hover:bg-red-600 shadow-red-500/20 text-white",
        info: "bg-indigo-500 hover:bg-indigo-600 shadow-indigo-500/20 text-white",
        success: "bg-emerald-500 hover:bg-emerald-600 shadow-emerald-500/20 text-white"
    };

    const iconColors = {
        danger: "text-red-400",
        info: "text-indigo-400",
        success: "text-emerald-400"
    };

    return (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300"
                onClick={onClose}
            />

            <div className="relative w-full max-w-md rounded-[2rem] border border-white/10 bg-zinc-950 p-8 shadow-2xl animate-in fade-in zoom-in-95 duration-300">
                <div className="flex flex-col items-center text-center gap-6">
                    <div className={`p-4 rounded-2xl bg-white/5 ${iconColors[variant]}`}>
                        <AlertTriangle className="h-8 w-8" />
                    </div>

                    <div className="space-y-2">
                        <h3 className="text-xl font-black text-white uppercase tracking-tight">{title}</h3>
                        <p className="text-sm text-zinc-500 font-medium leading-relaxed">{message}</p>
                    </div>

                    <div className="flex flex-col w-full gap-3 pt-2">
                        <button
                            onClick={onConfirm}
                            disabled={isLoading}
                            className={`h-14 w-full rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] transition-all active:scale-[0.98] flex items-center justify-center gap-3 shadow-xl ${variantStyles[variant]}`}
                        >
                            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : confirmLabel}
                        </button>

                        <button
                            onClick={onClose}
                            disabled={isLoading}
                            className="h-14 w-full rounded-2xl bg-zinc-900 text-zinc-400 text-[11px] font-black uppercase tracking-[0.2em] hover:bg-zinc-800 transition-all active:scale-[0.98]"
                        >
                            {cancelLabel}
                        </button>
                    </div>
                </div>

                <button
                    onClick={onClose}
                    className="absolute top-6 right-6 p-2 rounded-xl text-zinc-600 hover:text-white transition-colors"
                >
                    <X className="h-5 w-5" />
                </button>
            </div>
        </div>
    );
}
