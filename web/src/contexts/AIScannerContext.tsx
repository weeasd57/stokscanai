"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { scanAiFastWithParams, type ScanAiParams, type ScanResult } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";

type AiScannerState = {
    country: string;
    scanAllMarket: boolean;
    limit: number;
    modelName: string;
    results: ScanResult[];
    progress: { current: number; total: number };
    hasScanned: boolean;
    scanHistory: Array<{
        id?: string;
        createdAt: number;
        params: ScanAiParams;
        results: ScanResult[];
        scannedCount: number;
        durationMs?: number;
    }>;
    showPrecisionInfo: boolean;
    selected: ScanResult | null;
    detailData: PredictResponse | null;
    rfPreset: "fast" | "default" | "accurate";
    rfParamsJson: string;
    chartType: "candle" | "area";
    lastScanStartedAt: number | null;
    lastScanEndedAt: number | null;
    lastDurationMs: number | null;
    showEma50: boolean;
    showEma200: boolean;
    showBB: boolean;
    showRsi: boolean;
    showVolume: boolean;
};

interface AIScannerContextType {
    state: AiScannerState;
    setAiScanner: React.Dispatch<React.SetStateAction<AiScannerState>>;
    loading: boolean;
    error: string | null;
    runAiScan: (opts: { rfParams: Record<string, unknown> | null; minPrecision?: number; force?: boolean }) => Promise<void>;
    stopAiScan: () => void;
    clearAiScannerView: () => void;
    restoreLastAiScan: () => boolean;
    resetAiScanner: () => void;
}

const DEFAULT_STATE: AiScannerState = {
    country: "Egypt",
    scanAllMarket: true,
    limit: 100,
    modelName: "",
    results: [],
    progress: { current: 0, total: 0 },
    hasScanned: false,
    scanHistory: [],
    showPrecisionInfo: false,
    selected: null,
    detailData: null,
    rfPreset: "fast",
    rfParamsJson: "{}",
    chartType: "candle",
    lastScanStartedAt: null,
    lastScanEndedAt: null,
    lastDurationMs: null,
    showEma50: false,
    showEma200: false,
    showBB: false,
    showRsi: false,
    showVolume: false,
};

const AIScannerContext = createContext<AIScannerContextType | undefined>(undefined);

export const AIScannerProvider = ({ children }: { children: ReactNode }) => {

    const [state, setAiScanner] = useState<AiScannerState>(DEFAULT_STATE);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    // Load History from LocalStorage
    useEffect(() => {
        try {
            const saved = localStorage.getItem("ai_scan_history");
            if (saved) {
                const history = JSON.parse(saved);
                if (Array.isArray(history)) {
                    setAiScanner(prev => ({ ...prev, scanHistory: history }));
                }
            }
        } catch (err) {
            console.error("Failed to load history from localStorage", err);
        }
    }, []);

    // Save History to LocalStorage whenever it changes
    useEffect(() => {
        if (state.scanHistory.length > 0) {
            localStorage.setItem("ai_scan_history", JSON.stringify(state.scanHistory));
        }
    }, [state.scanHistory]);

    const stopAiScan = useCallback(() => {
        if (abortRef.current) {
            abortRef.current.abort();
            abortRef.current = null;
            setLoading(false);
        }
    }, []);

    const runAiScan = useCallback(async (opts: { rfParams: Record<string, unknown> | null; minPrecision?: number; force?: boolean }) => {
        if (loading) return;

        const controller = new AbortController();
        abortRef.current = controller;

        const { country, limit, modelName } = state;
        const params: ScanAiParams = {
            country,
            limit: state.scanAllMarket ? 400 : limit,
            modelName: modelName,
            minPrecision: opts.minPrecision ?? 0.5,
            rfPreset: state.rfPreset,
            rfParamsJson: state.rfParamsJson,
            rfParams: opts.rfParams,
            scanAll: state.scanAllMarket
        };

        setLoading(true);
        setError(null);
        const startedAt = Date.now();

        setAiScanner(prev => ({
            ...prev,
            results: [],
            hasScanned: false,
            progress: { current: 0, total: params.limit || 100 },
            lastScanStartedAt: startedAt,
            lastScanEndedAt: null,
            lastDurationMs: null,
        }));

        try {
            const res = await scanAiFastWithParams(params, controller.signal);
            const sortedResults = (res.results || []).slice().sort((a, b) => (b.precision ?? 0) - (a.precision ?? 0));
            const endedAt = Date.now();
            const duration = endedAt - startedAt;

            setAiScanner(prev => {
                const newEntry = {
                    createdAt: endedAt,
                    params,
                    results: sortedResults,
                    scannedCount: res.scanned_count,
                    durationMs: duration
                };
                return {
                    ...prev,
                    results: sortedResults,
                    progress: { current: params.limit || 100, total: params.limit || 100 },
                    hasScanned: true,
                    lastScanEndedAt: endedAt,
                    lastDurationMs: duration,
                    scanHistory: [newEntry, ...prev.scanHistory].slice(0, 10),
                };
            });
        } catch (err: any) {
            if (err.name === 'AbortError') {
                console.log('AI Scan aborted');
            } else {
                setError(err instanceof Error ? err.message : 'Scan failed');
            }
        } finally {
            setLoading(false);
            abortRef.current = null;
        }
    }, [state, loading]);

    const clearAiScannerView = useCallback(() => {
        setAiScanner(prev => ({
            ...prev,
            results: [],
            hasScanned: false,
            selected: null,
            progress: { current: 0, total: 0 }
        }));
    }, []);

    const restoreLastAiScan = useCallback(() => {
        if (state.scanHistory.length === 0) return false;
        const last = state.scanHistory[0];
        setAiScanner(prev => ({
            ...prev,
            results: last.results,
            hasScanned: true,
            progress: { current: last.scannedCount, total: last.scannedCount },
            lastDurationMs: last.durationMs || null,
            lastScanEndedAt: last.createdAt,
        }));
        return true;
    }, [state.scanHistory]);

    const resetAiScanner = useCallback(() => {
        setAiScanner(DEFAULT_STATE);
    }, []);

    const value = useMemo(() => ({
        state,
        setAiScanner,
        loading,
        error,
        runAiScan,
        stopAiScan,
        clearAiScannerView,
        restoreLastAiScan,
        resetAiScanner,
    }), [state, loading, error, runAiScan, stopAiScan, clearAiScannerView, restoreLastAiScan, resetAiScanner]);

    return <AIScannerContext.Provider value={value}>{children}</AIScannerContext.Provider>;
};

export const useAIScanner = () => {
    const context = useContext(AIScannerContext);
    if (!context) throw new Error("useAIScanner must be used within an AIScannerProvider");
    return context;
};
