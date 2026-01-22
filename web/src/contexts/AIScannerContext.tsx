"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { scanAiFastWithParams, evaluateScan, type ScanAiParams, type ScanResult } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";
import { useAuth } from "./AuthContext";

type AiScannerState = {
    country: string;
    scanAllMarket: boolean;
    limit: number;
    modelName: string;
    startDate: string;
    endDate: string;
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
    runAiScan: (opts: { rfParams: Record<string, unknown> | null; minPrecision?: number; force?: boolean; shouldSave?: boolean }) => Promise<void>;
    stopAiScan: () => void;
    clearAiScannerView: () => void;
    restoreLastAiScan: () => boolean;
    resetAiScanner: () => void;
    fetchScanHistory: () => Promise<any[]>;
    fetchScanResults: (scanId: string) => Promise<any[]>;
    fetchLatestScanForModel: (modelName: string) => Promise<{ history: any; results: any[] } | null>;
    refreshScanPerformance: (scanId: string) => Promise<{ count: number; message: string }>;
    saveCurrentScan: () => Promise<void>;
}

const DEFAULT_STATE: AiScannerState = {
    country: "Egypt",
    scanAllMarket: true,
    limit: 100,
    modelName: "",
    startDate: "",
    endDate: "",
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

    const { user } = useAuth();
    const supabase = useMemo(() => createSupabaseBrowserClient(), []);
    const [state, setAiScanner] = useState<AiScannerState>(DEFAULT_STATE);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const saveScanToSupabase = useCallback(async (
        scanParams: ScanAiParams,
        results: ScanResult[],
        scannedCount: number,
        durationMs: number
    ) => {
        if (!user) return;

        try {
            // 1. Insert header into scan_history
            const { data: history, error: historyErr } = await supabase
                .from("scan_history")
                .insert({
                    user_id: user.id,
                    model_name: scanParams.modelName || "unknown",
                    country: scanParams.country,
                    from_date: scanParams.from_date || null,
                    to_date: scanParams.to_date || null,
                    scanned_count: scannedCount,
                    duration_ms: durationMs
                })
                .select()
                .single();

            if (historyErr) throw historyErr;
            if (!history) return;

            // 2. Insert rows into scan_results
            const rows = results.map(r => ({
                scan_id: history.id,
                symbol: r.symbol,
                exchange: r.exchange || "US",
                last_close: r.last_close,
                precision: r.precision,
                signal: r.signal,
                confidence: r.confidence,
                status: 'open',
                entry_price: r.last_close
            }));

            if (rows.length > 0) {
                const { error: resultsErr } = await supabase
                    .from("scan_results")
                    .insert(rows);
                if (resultsErr) throw resultsErr;
            }

            console.log(`Saved scan ${history.id} to Supabase with ${rows.length} items`);
        } catch (err) {
            console.error("Failed to save scan to Supabase:", err);
        }
    }, [user, supabase]);

    const saveCurrentScan = useCallback(async () => {
        if (!state.results.length || !user) return;

        // Calculate dynamic stats
        const scannedCount = state.progress.total || state.results.length;
        const durationMs = state.lastDurationMs || 0;

        const params: ScanAiParams = {
            country: state.country,
            limit: state.limit,
            modelName: state.modelName,
            from_date: state.startDate,
            to_date: state.endDate,
            scanAll: state.scanAllMarket, // NOTE: scanAllMarket maps to scanAll in ScanAiParams not represented in the snippet but assumed correct based on context or need correction if scanAllMarket is the key in params which is scanAll
            minPrecision: 0.6, // Default or placeholder
            rfPreset: state.rfPreset,
            rfParamsJson: state.rfParamsJson,
            rfParams: null
        };

        await saveScanToSupabase(params, state.results, scannedCount, durationMs);
    }, [state.results, state.progress.total, state.lastDurationMs, state.country, state.limit, state.modelName, state.startDate, state.endDate, state.scanAllMarket, saveScanToSupabase, user]);
    const fetchScanHistory = useCallback(async () => {
        if (!user) return [];
        const { data, error } = await supabase
            .from("scan_history")
            .select("*")
            .eq("user_id", user.id)
            .order("created_at", { ascending: false });
        if (error) {
            console.error("Error fetching scan history:", error);
            return [];
        }
        return data;
    }, [user, supabase]);

    const fetchScanResults = useCallback(async (scanId: string) => {
        if (!supabase) return [];
        const { data, error } = await supabase
            .from("scan_results")
            .select("*")
            .eq("scan_id", scanId)
            .order("precision", { ascending: false });
        if (error) {
            console.error("Error fetching scan results:", error);
            return [];
        }
        return data || [];
    }, [supabase]);

    const fetchLatestScanForModel = useCallback(async (modelName: string) => {
        if (!supabase) return null;

        // 1. Get the latest history record for this model
        const { data: history, error: hErr } = await supabase
            .from("scan_history")
            .select("*")
            .eq("model_name", modelName)
            .order("created_at", { ascending: false })
            .limit(1)
            .maybeSingle();

        if (hErr || !history) return null;

        // 2. Get the results for this history record
        const results = await fetchScanResults(history.id);

        return { history, results };
    }, [supabase, fetchScanResults]);

    const refreshScanPerformance = useCallback(async (scanId: string) => {
        try {
            const res = await evaluateScan(scanId);
            return res;
        } catch (err) {
            console.error("Error refreshing scan performance:", err);
            throw err;
        }
    }, []);

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

    const runAiScan = useCallback(async (opts: { rfParams: Record<string, unknown> | null; minPrecision?: number; force?: boolean; shouldSave?: boolean }) => {
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
            scanAll: state.scanAllMarket,
            from_date: state.startDate || undefined,
            to_date: state.endDate || undefined
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

            // Save to Supabase in background if requested
            if (opts.shouldSave !== false) {
                saveScanToSupabase(params, sortedResults, res.scanned_count, duration);
            }

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
    }, [state, loading, saveScanToSupabase]);

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
        fetchScanHistory,
        fetchScanResults,
        fetchLatestScanForModel,
        refreshScanPerformance,
        saveCurrentScan,
    }), [state, loading, error, runAiScan, stopAiScan, clearAiScannerView, restoreLastAiScan, resetAiScanner, fetchScanHistory, fetchScanResults, fetchLatestScanForModel, refreshScanPerformance, saveCurrentScan]);

    return <AIScannerContext.Provider value={value}>{children}</AIScannerContext.Provider>;
};

export const useAIScanner = () => {
    const context = useContext(AIScannerContext);
    if (!context) throw new Error("useAIScanner must be used within an AIScannerProvider");
    return context;
};
