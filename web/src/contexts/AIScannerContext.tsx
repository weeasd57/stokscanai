"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { scanAiFastWithParams, evaluateScan, getAdminConfig, type ScanAiParams, type ScanResult } from "@/lib/api";
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
    showMacd: boolean;
    scanDays: number;
    scanAllHistory: boolean;
    targetPct: number;
    stopLossPct: number;
    lookForwardDays: number;
    buyThreshold: number;
};

interface AIScannerContextType {
    state: AiScannerState;
    setAiScanner: React.Dispatch<React.SetStateAction<AiScannerState>>;
    loading: boolean;
    error: string | null;
    runAiScan: (opts: {
        rfParams: Record<string, unknown> | null;
        minPrecision?: number;
        force?: boolean;
        shouldSave?: boolean;
        buy_threshold?: number;
        target_pct?: number;
        stop_loss_pct?: number;
        look_forward_days?: number;
    }) => Promise<void>;
    stopAiScan: () => void;
    clearAiScannerView: () => void;
    restoreLastAiScan: () => boolean;
    resetAiScanner: () => void;
    fetchScanHistory: (filters?: { country?: string; model?: string; isPublic?: boolean }) => Promise<any[]>;
    fetchScanResults: (scanId: string) => Promise<any[]>;
    fetchLatestScanForModel: (modelName: string, onlyPublic?: boolean) => Promise<{ history: any; results: any[] } | null>;
    refreshScanPerformance: (scanId: string) => Promise<{ count: number; message: string }>;
    saveCurrentScan: (isPublic?: boolean) => Promise<boolean>;
    saveSelectedResults: (results: ScanResult[], isPublic: boolean) => Promise<boolean>;
    fetchPublishedResults: (filters?: { country?: string; model?: string; startDate?: string; endDate?: string }) => Promise<ScanResult[]>;
    toggleResultPublicStatus: (id: string, isPublic: boolean) => Promise<boolean>;
    bulkUpdatePublicStatus: (ids: string[], isPublic: boolean) => Promise<boolean>;
    fetchPublicScanDates: (modelName: string) => Promise<string[]>;
    fetchScanResultsByDate: (modelName: string, date: string) => Promise<any[]>;
    updateResultStatus: (id: string, status: "win" | "loss") => Promise<boolean>;
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
    showMacd: true,
    scanDays: 450,
    scanAllHistory: false,
    targetPct: 0.10,
    stopLossPct: 0.05,
    lookForwardDays: 20,
    buyThreshold: 0.40,
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
        durationMs: number,
        isPublic: boolean = false
    ) => {
        if (!user) return;

        try {
            const batchId = crypto.randomUUID();

            // Insert rows directly into scan_results with all metadata
            // Use the scan's end date (to_date) as created_at so records are saved with the selected date
            const scanDate = scanParams.to_date
                ? new Date(scanParams.to_date + 'T12:00:00Z').toISOString()
                : new Date().toISOString();

            const rows = results.map(r => ({
                batch_id: batchId,
                user_id: user.id,
                symbol: r.symbol,
                exchange: r.exchange || "US",
                name: r.name,
                model_name: scanParams.modelName || "unknown",
                country: scanParams.country,
                last_close: r.last_close,
                precision: r.precision,
                signal: r.signal,
                is_public: isPublic,
                from_date: scanParams.from_date || null,
                to_date: scanParams.to_date || null,
                scanned_count: scannedCount,
                duration_ms: durationMs,
                status: 'open',
                entry_price: r.last_close,
                target_price: r.target_price,
                stop_loss: r.stop_loss,
                logo_url: r.logo_url,
                top_reasons: r.top_reasons,
                features: r.features ? JSON.stringify({
                    ...((Array.isArray(r.features) ? {} : r.features) as any), // Handle array vs object legacy
                    raw_features: Array.isArray(r.features) ? r.features : [],
                    technical_score: r.technical_score || 0,
                    fundamental_score: r.fundamental_score || 0
                }) : JSON.stringify({
                    technical_score: r.technical_score || 0,
                    fundamental_score: r.fundamental_score || 0
                }),
                created_at: scanDate
            }));

            if (rows.length > 0) {
                const { error: resultsErr } = await supabase
                    .from("scan_results")
                    .insert(rows);
                if (resultsErr) throw resultsErr;
            }

            console.log(`Saved batch ${batchId} to scan_results with ${rows.length} items`);
        } catch (err) {
            console.error("Failed to save scan results:", err);
        }
    }, [user, supabase]);

    const saveCurrentScan = useCallback(async (isPublic: boolean = false): Promise<boolean> => {
        if (!state.results.length || !user) return false;
        const scannedCount = state.progress.total || state.results.length;
        const durationMs = state.lastDurationMs || 0;
        const targetEndDate = state.endDate || new Date().toISOString().split('T')[0];
        const lookback = state.scanDays || 450;
        const calculatedFromDate = state.scanAllHistory
            ? "2010-01-01"
            : new Date(new Date(targetEndDate).getTime() - lookback * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        const params: ScanAiParams = {
            country: state.country,
            limit: state.limit,
            modelName: state.modelName,
            from_date: calculatedFromDate,
            to_date: targetEndDate,
            scanAll: state.scanAllMarket,
            minPrecision: state.buyThreshold,
            rfPreset: state.rfPreset,
            rfParamsJson: state.rfParamsJson,
            rfParams: null,
            target_pct: state.targetPct,
            stop_loss_pct: state.stopLossPct,
            look_forward_days: state.lookForwardDays,
            buy_threshold: state.buyThreshold
        };

        try {
            await saveScanToSupabase(params, state.results, scannedCount, durationMs, isPublic);
            return true;
        } catch (err) {
            console.error("Save Current Scan failed:", err);
            return false;
        }
    }, [state.results, state.progress.total, state.lastDurationMs, state.country, state.limit, state.modelName, state.endDate, state.scanAllMarket, state.rfPreset, state.rfParamsJson, saveScanToSupabase, user]);

    const saveSelectedResults = useCallback(async (selectedResults: ScanResult[], isPublic: boolean = true): Promise<boolean> => {
        if (!selectedResults.length || !user) return false;
        const scannedCount = state.progress.total || state.results.length;
        const durationMs = state.lastDurationMs || 0;
        const targetEndDate = state.endDate || new Date().toISOString().split('T')[0];
        const lookback = state.scanDays || 450;
        const calculatedFromDate = state.scanAllHistory
            ? "2010-01-01"
            : new Date(new Date(targetEndDate).getTime() - lookback * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        const params: ScanAiParams = {
            country: state.country,
            limit: state.limit,
            modelName: state.modelName,
            from_date: calculatedFromDate,
            to_date: targetEndDate,
            scanAll: state.scanAllMarket,
            minPrecision: state.buyThreshold,
            rfPreset: state.rfPreset,
            rfParamsJson: state.rfParamsJson,
            rfParams: null,
            target_pct: state.targetPct,
            stop_loss_pct: state.stopLossPct,
            look_forward_days: state.lookForwardDays,
            buy_threshold: state.buyThreshold
        };

        try {
            // Save only the selected results
            await saveScanToSupabase(params, selectedResults, scannedCount, durationMs, isPublic);
            return true;
        } catch (err) {
            console.error("Save Selected Results failed:", err);
            return false;
        }
    }, [state.results, state.progress.total, state.lastDurationMs, state.country, state.limit, state.modelName, state.endDate, state.scanAllMarket, state.rfPreset, state.rfParamsJson, saveScanToSupabase, user]);
    const fetchScanHistory = useCallback(async (filters?: { country?: string; model?: string; isPublic?: boolean }) => {
        if (!user) return [];

        // Query unique batch_ids to simulate history
        let query = supabase
            .from("scan_results")
            .select("batch_id, created_at, model_name, country, from_date, to_date, scanned_count, duration_ms, is_public")
            .order("created_at", { ascending: false });

        if (filters?.country) query = query.eq("country", filters.country);
        if (filters?.model) query = query.eq("model_name", filters.model);
        if (filters?.isPublic !== undefined) query = query.eq("is_public", filters.isPublic);
        else query = query.eq("user_id", user.id);

        const { data, error } = await query;
        if (error) {
            console.error("Error fetching scan history:", error);
            return [];
        }

        // Deduplicate by batch_id on client side (or use distinct in SQL if available)
        const uniqueBatches: any[] = [];
        const seen = new Set();
        for (const item of data) {
            if (!seen.has(item.batch_id)) {
                uniqueBatches.push(item);
                seen.add(item.batch_id);
            }
        }

        return uniqueBatches;
    }, [user, supabase]);

    const fetchScanResults = useCallback(async (batchId: string) => {
        if (!supabase) return [];
        const { data, error } = await supabase
            .from("scan_results")
            .select("*")
            .eq("batch_id", batchId)
            .order("precision", { ascending: false });
        if (error) {
            console.error("Error fetching scan results:", error);
            return [];
        }
        return (data || []).map((row: any) => {
            let tech = row.technical_score;
            let fund = row.fundamental_score;
            if ((tech === undefined || tech === null) && row.features) {
                try {
                    const f = typeof row.features === 'string' ? JSON.parse(row.features) : row.features;
                    if (f && typeof f.technical_score === 'number') tech = f.technical_score;
                    if (f && typeof f.fundamental_score === 'number') fund = f.fundamental_score;
                } catch (e) { }
            }
            return { ...row, technical_score: tech || 0, fundamental_score: fund || 0 };
        });
    }, [supabase]);

    const fetchLatestScanForModel = useCallback(async (modelName: string, onlyPublic: boolean = true) => {
        if (!supabase) return null;

        // 1. Get the latest batch_id or date for this model
        let query = supabase
            .from("scan_results")
            .select("batch_id, created_at")
            .eq("model_name", modelName)
            .order("created_at", { ascending: false });

        if (onlyPublic) {
            query = query.eq("is_public", true);
        }

        const { data, error } = await query.limit(1);

        if (error || !data || data.length === 0) return null;

        const latestCreatedAt = data[0].created_at;
        const dateStr = new Date(latestCreatedAt).toISOString().split('T')[0];

        // 2. Fetch results for that specific date (or batch if preferred, but user wants date pagination)
        const { data: results, error: resultsErr } = await supabase
            .from("scan_results")
            .select("*")
            .eq("model_name", modelName)
            .gte("created_at", `${dateStr}T00:00:00`)
            .lte("created_at", `${dateStr}T23:59:59`)
            .eq("is_public", true)
            .order("precision", { ascending: false });

        if (resultsErr || !results) return null;

        const mappedResults = (results || []).map((row: any) => {
            let tech = row.technical_score;
            let fund = row.fundamental_score;
            if ((tech === undefined || tech === null) && row.features) {
                try {
                    const f = typeof row.features === 'string' ? JSON.parse(row.features) : row.features;
                    if (f && typeof f.technical_score === 'number') tech = f.technical_score;
                    if (f && typeof f.fundamental_score === 'number') fund = f.fundamental_score;
                } catch (e) { }
            }
            return { ...row, technical_score: tech || 0, fundamental_score: fund || 0 };
        });

        return { history: mappedResults[0], results: mappedResults };
    }, [supabase]);

    const toggleResultPublicStatus = useCallback(async (id: string, isPublic: boolean) => {
        if (!supabase) return false;
        const { error } = await supabase
            .from("scan_results")
            .update({ is_public: isPublic })
            .eq("id", id);
        if (error) {
            console.error("Error toggling public status:", error);
            return false;
        }
        return true;
    }, [supabase]);

    const bulkUpdatePublicStatus = useCallback(async (ids: string[], isPublic: boolean) => {
        if (!supabase || ids.length === 0) return false;
        const { error } = await supabase
            .from("scan_results")
            .update({ is_public: isPublic })
            .in("id", ids);
        if (error) {
            console.error("Error bulk updating public status:", error);
            return false;
        }
        return true;
    }, [supabase]);

    const fetchPublicScanDates = useCallback(async (modelName: string) => {
        if (!supabase) return [];
        // Get unique dates where results are public
        const { data, error } = await supabase
            .from("scan_results")
            .select("created_at")
            .eq("model_name", modelName)
            .eq("is_public", true)
            .order("created_at", { ascending: false });

        if (error) {
            console.error("Error fetching public scan dates:", error);
            return [];
        }

        const dates = Array.from(new Set(data.map(d => new Date(d.created_at).toISOString().split('T')[0])));
        return dates;
    }, [supabase]);

    const fetchScanResultsByDate = useCallback(async (modelName: string, date: string) => {
        if (!supabase) return [];
        const { data, error } = await supabase
            .from("scan_results")
            .select("*")
            .eq("model_name", modelName)
            .eq("is_public", true)
            .gte("created_at", `${date}T00:00:00`)
            .lte("created_at", `${date}T23:59:59`)
            .order("precision", { ascending: false });

        if (error) {
            console.error("Error fetching scan results by date:", error);
            return [];
        }
        return (data || []).map((row: any) => {
            let tech = row.technical_score;
            let fund = row.fundamental_score;
            if ((tech === undefined || tech === null) && row.features) {
                try {
                    const f = typeof row.features === 'string' ? JSON.parse(row.features) : row.features;
                    if (f && typeof f.technical_score === 'number') tech = f.technical_score;
                    if (f && typeof f.fundamental_score === 'number') fund = f.fundamental_score;
                } catch (e) { }
            }
            return { ...row, technical_score: tech || 0, fundamental_score: fund || 0 };
        });
    }, [supabase]);

    const fetchPublishedResults = useCallback(async (filters?: { country?: string; model?: string; startDate?: string; endDate?: string }) => {
        if (!supabase) return [];
        let query = supabase
            .from("scan_results")
            .select("*")
            .eq("is_public", true)
            .order("created_at", { ascending: false });

        if (filters?.country) query = query.eq("country", filters.country);
        if (filters?.model) query = query.eq("model_name", filters.model);
        if (filters?.startDate) query = query.gte("created_at", `${filters.startDate}T00:00:00`);
        if (filters?.endDate) query = query.lte("created_at", `${filters.endDate}T23:59:59`);

        const { data, error } = await query;

        if (error) {
            console.error("Error fetching published results:", error);
            return [];
        }
        return (data || []).map((row: any) => {
            let tech = row.technical_score;
            let fund = row.fundamental_score;

            // Unpack from features JSON if missing from columns
            if ((tech === undefined || tech === null) && row.features) {
                try {
                    const f = typeof row.features === 'string' ? JSON.parse(row.features) : row.features;
                    if (f && typeof f.technical_score === 'number') tech = f.technical_score;
                    if (f && typeof f.fundamental_score === 'number') fund = f.fundamental_score;
                } catch (e) { /* ignore parse error */ }
            }
            return { ...row, technical_score: tech || 0, fundamental_score: fund || 0 };
        });
    }, [supabase]);

    const updateResultStatus = useCallback(async (id: string, status: "win" | "loss") => {
        if (!supabase) return false;

        // Calculate P/L if status is set
        // In a real app we might fetch the actual price, but for now we just mark status
        const { error } = await supabase
            .from("scan_results")
            .update({
                status: status,
                // If it's closed, naturally it's no longer 'open'
                // We'll use a virtual 'closed' status logic or just check the status field
            })
            .eq("id", id);

        if (error) {
            console.error("Error updating result status:", error);
            return false;
        }
        return true;
    }, [supabase]);

    const refreshScanPerformance = useCallback(async (scanId: string) => {
        try {
            const res = await evaluateScan(scanId);
            return res;
        } catch (err) {
            console.error("Error refreshing scan performance:", err);
            throw err;
        }
    }, []);

    // Sync scanDays from AdminConfig on mount
    useEffect(() => {
        getAdminConfig().then(cfg => {
            if (cfg.scanDays) {
                setAiScanner(prev => ({ ...prev, scanDays: cfg.scanDays! }));
            }
        }).catch(err => console.error("Failed to sync initial scanDays:", err));
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

    const runAiScan = useCallback(async (opts: {
        rfParams: Record<string, unknown> | null;
        minPrecision?: number;
        force?: boolean;
        shouldSave?: boolean;
        buy_threshold?: number;
        target_pct?: number;
        stop_loss_pct?: number;
        look_forward_days?: number;
    }) => {
        if (loading) return;

        const controller = new AbortController();
        abortRef.current = controller;

        const { country, limit, modelName } = state;
        setLoading(true);
        setError(null);

        // Show loading state initially (0 total means "fetching count...")
        setAiScanner(prev => ({
            ...prev,
            results: [],
            hasScanned: false,
            progress: { current: 0, total: 0 },
            lastScanStartedAt: Date.now(),
            lastScanEndedAt: null,
            lastDurationMs: null,
        }));

        let totalToScan = limit; // Default to user-specified limit
        try {
            // Dynamically fetch actual symbol count for this country
            const { getSyncedSymbols } = await import("@/lib/api");
            const allSymbols = await getSyncedSymbols(country, "local");
            if (allSymbols && allSymbols.length > 0) {
                totalToScan = state.scanAllMarket ? allSymbols.length : Math.min(limit, allSymbols.length);
            }
        } catch (err) {
            console.error("Failed to fetch symbol count for progress:", err);
            // If we can't get the count, use a reasonable estimate based on scanAllMarket
            totalToScan = state.scanAllMarket ? 500 : limit;
        }

        // Update progress with the actual total
        setAiScanner(prev => ({
            ...prev,
            progress: { current: 0, total: totalToScan },
        }));

        const targetEndDate = state.endDate || new Date().toISOString().split('T')[0];
        const lookback = state.scanDays || 450;
        const calculatedFromDate = state.scanAllHistory
            ? "2010-01-01"
            : new Date(new Date(targetEndDate).getTime() - lookback * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

        const params: ScanAiParams = {
            country,
            limit: totalToScan,
            modelName: modelName,
            minPrecision: opts.minPrecision ?? state.buyThreshold,
            rfPreset: state.rfPreset,
            rfParamsJson: state.rfParamsJson,
            rfParams: opts.rfParams,
            scanAll: state.scanAllMarket,
            from_date: calculatedFromDate,
            to_date: targetEndDate,
            target_pct: opts.target_pct ?? state.targetPct,
            stop_loss_pct: opts.stop_loss_pct ?? state.stopLossPct,
            look_forward_days: opts.look_forward_days ?? state.lookForwardDays,
            buy_threshold: opts.buy_threshold ?? state.buyThreshold
        };

        const startedAt = Date.now();

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
                    progress: { current: totalToScan, total: totalToScan },
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
        saveSelectedResults,
        toggleResultPublicStatus,
        bulkUpdatePublicStatus,
        fetchPublicScanDates,
        fetchScanResultsByDate: fetchScanResultsByDate, // explicit mapping if needed but usually it's just shorthand
        updateResultStatus,
        fetchPublishedResults,
    }), [state, loading, error, runAiScan, stopAiScan, clearAiScannerView, restoreLastAiScan, resetAiScanner, fetchScanHistory, fetchScanResults, fetchLatestScanForModel, refreshScanPerformance, saveCurrentScan, saveSelectedResults, toggleResultPublicStatus, bulkUpdatePublicStatus, fetchPublicScanDates, fetchScanResultsByDate, updateResultStatus, fetchPublishedResults]);

    return <AIScannerContext.Provider value={value}>{children}</AIScannerContext.Provider>;
};

export const useAIScanner = () => {
    const context = useContext(AIScannerContext);
    if (!context) throw new Error("useAIScanner must be used within an AIScannerProvider");
    return context;
};
