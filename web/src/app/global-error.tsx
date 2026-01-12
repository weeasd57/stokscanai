"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function GlobalError({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    useEffect(() => {
        console.error("Global application error:", error);
    }, [error]);

    return (
        <html>
            <body className="bg-zinc-950 text-white">
                <div className="min-h-screen flex items-center justify-center p-8">
                    <div className="max-w-md w-full text-center space-y-6">
                        <div className="mx-auto w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
                            <AlertTriangle className="w-8 h-8 text-red-500" />
                        </div>

                        <div className="space-y-2">
                            <h1 className="text-2xl font-black uppercase tracking-tight">
                                Critical Error
                            </h1>
                            <p className="text-sm text-zinc-500">
                                A critical error occurred. Please refresh the page.
                            </p>
                            {error.message && (
                                <p className="text-xs text-red-400/60 font-mono mt-4 p-3 rounded-xl bg-red-500/5 border border-red-500/10">
                                    {error.message}
                                </p>
                            )}
                        </div>

                        <button
                            onClick={reset}
                            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 font-bold text-sm uppercase tracking-widest transition-all"
                        >
                            <RefreshCw className="w-4 h-4" />
                            Refresh Page
                        </button>
                    </div>
                </div>
            </body>
        </html>
    );
}
