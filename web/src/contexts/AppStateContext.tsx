"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { ScanResult, TechResult, TechFilter } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import { getCountries, predictStock, scanAiWithParams, scanAiFastWithParams, scanTech, type ScanAiParams } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";


type Updater<T> = Partial<T> | ((prev: T) => T);

type HomeState = {
  ticker: string;
  data: PredictResponse | null;
  predictHistory: Array<{ key: string; createdAt: number; ticker: string; data: PredictResponse }>;
  chartType: "candle" | "area";
  showEma50: boolean;
  showEma200: boolean;
  showBB: boolean;
  showRsi: boolean;
  showVolume: boolean;
  watchlist: string[];
};

type TechScannerState = {
  country: string;
  results: TechResult[];
  hasScanned: boolean;
  scannedCount: number;
  scanHistory: Array<{ key: string; createdAt: number; filter: TechFilter; results: TechResult[]; scannedCount: number }>;
  selectedStock: TechResult | null;
  searchTerm: string;

  rsiMin: string;
  rsiMax: string;
  aboveEma50: boolean;
  aboveEma200: boolean;

  adxMin: string;
  adxMax: string;
  atrMin: string;
  atrMax: string;
  stochKMin: string;
  stochKMax: string;
  rocMin: string;
  rocMax: string;
  aboveVwap20: boolean;
  volumeAboveSma20: boolean;
  goldenCross: boolean;

  currentTab: 'overview' | 'performance' | 'valuation' | 'dividends' | 'financials';
  marketCapMin: string;
  marketCapMax: string;
  sector: string;
  industry: string;
};

type ComparisonScannerState = {
  symbols: string[];
  results: Record<string, PredictResponse>;
  loadingSymbols: string[];
  errors: Record<string, string>;
};


type AppState = {
  home: HomeState;
  techScanner: TechScannerState;
  comparisonScanner: ComparisonScannerState;
  aiScanner: AiScannerState;
};

type AiScannerState = {
  country: string;
  results: ScanResult[];
  hasScanned: boolean;
  scannedCount: number;
  scanHistory: Array<{ key: string; createdAt: number; results: ScanResult[]; scannedCount: number }>;

  minPrecision: string;
  modelName: string;
  rfPreset: "fast" | "default" | "accurate";
  rfParamsJson: string; // Stored as string to easy edit
};

type AppStateContextType = {
  state: AppState;
  countries: string[];
  countriesLoading: boolean;
  refreshCountries: (opts?: { force?: boolean }) => Promise<void>;
  syncedSymbols: any[];
  syncedSymbolsLoading: boolean;
  refreshSyncedSymbols: (country?: string) => Promise<void>;
  setHome: (u: Updater<HomeState>) => void;
  setTechScanner: (u: Updater<TechScannerState>) => void;
  setComparisonScanner: (u: Updater<ComparisonScannerState>) => void;
  setAiScanner: (u: Updater<AiScannerState>) => void;
  runHomePredict: (ticker: string, opts?: { signal?: AbortSignal; force?: boolean }) => Promise<void>;
  clearHomeView: () => void;
  restoreLastHomePredict: () => boolean;
  // Tech Scan with persistent loading state
  techScanLoading: boolean;
  techScanError: string | null;
  runTechScan: (opts?: { force?: boolean }) => Promise<void>;
  stopTechScan: () => void;
  clearTechScannerView: () => void;
  inventory: any[];
  inventoryLoading: boolean;
  refreshInventory: () => Promise<void>;
  isCountryActive: (country: string) => boolean;
  isSymbolActive: (symbol: string, exchange: string) => boolean;
  isAdmin: boolean;
  restoreLastTechScan: () => boolean;
  // Comparison
  addSymbolToCompare: (symbol: string) => Promise<void>;
  addSymbolsToCompare: (symbols: string[]) => Promise<void>;
  removeSymbolFromCompare: (symbol: string) => void;
  clearComparison: () => void;
  resetTechScanner: () => void;
  // Watchlist
  addSymbolToWatchlist: (symbol: string) => void;
  removeSymbolFromWatchlist: (symbol: string) => void;
};

const STORAGE_KEY = "ai_stocks_app_state_v1";
const COUNTRIES_KEY = "ai_stocks_countries_v1";
const HOME_PRED_KEY = "ai_stocks_home_last_predict_v1";

const DEFAULT_STATE: AppState = {
  home: {
    ticker: "AAPL",
    data: null,
    predictHistory: [],
    chartType: "candle",
    showEma50: false,
    showEma200: false,
    showBB: false,
    showRsi: false,
    showVolume: false,
    watchlist: [],
  },
  techScanner: {
    country: "Egypt",
    results: [],
    hasScanned: false,
    scannedCount: 0,
    scanHistory: [],
    selectedStock: null,
    searchTerm: "",

    rsiMin: "",
    rsiMax: "",
    aboveEma50: false,
    aboveEma200: false,

    adxMin: "",
    adxMax: "",
    atrMin: "",
    atrMax: "",
    stochKMin: "",
    stochKMax: "",
    rocMin: "",
    rocMax: "",
    aboveVwap20: false,
    volumeAboveSma20: false,
    goldenCross: false,
    currentTab: 'overview',
    marketCapMin: "",
    marketCapMax: "",
    sector: "",
    industry: "",
  },
  comparisonScanner: {
    symbols: [],
    results: {},
    loadingSymbols: [],
    errors: {},
  },
  aiScanner: {
    country: "Egypt",
    results: [],
    hasScanned: false,
    scannedCount: 0,
    scanHistory: [],
    minPrecision: "0.6",
    modelName: "",
    rfPreset: "default",
    rfParamsJson: "",
  }
};

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

type PersistedAppState = {
  home: Pick<HomeState, "ticker" | "chartType" | "showEma50" | "showEma200" | "showBB" | "showRsi" | "showVolume" | "watchlist">;
  techScanner: Pick<
    TechScannerState,
    | "country"
    | "results"
    | "hasScanned"
    | "scannedCount"
    | "scanHistory"
    // Note: 'selectedStock' intentionally NOT persisted - should start null on load
    | "searchTerm"
    | "rsiMin"
    | "rsiMax"
    | "aboveEma50"
    | "aboveEma200"
    | "adxMin"
    | "adxMax"
    | "atrMin"
    | "atrMax"
    | "stochKMin"
    | "stochKMax"
    | "rocMin"
    | "rocMax"
    | "aboveVwap20"
    | "volumeAboveSma20"
    | "goldenCross"
    | "currentTab"
    | "marketCapMin"
    | "marketCapMax"
    | "sector"
    | "industry"
  >;
  comparisonScanner: Pick<ComparisonScannerState, "symbols" | "results">;
  aiScanner: Pick<
    AiScannerState,
    | "country"
    | "results"
    | "hasScanned"
    | "scannedCount"
    | "scanHistory"
    | "minPrecision"
    | "modelName"
    | "rfPreset"
    | "rfParamsJson"
  >;
};

function safeParseState(raw: string | null): PersistedAppState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PersistedAppState;
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

function safeParseHomePredict(raw: string | null): PredictResponse | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as { data?: unknown };
    const data = (parsed as any)?.data;
    if (!data || typeof data !== "object") return null;
    if (typeof (data as any).ticker !== "string") return null;
    if (!Array.isArray((data as any).testPredictions)) return null;
    return data as PredictResponse;
  } catch {
    return null;
  }
}

function maybeRestoreHomeData(prev: AppState): AppState {
  const cached = safeParseHomePredict(localStorage.getItem(HOME_PRED_KEY));
  if (!cached) return prev;
  if (prev.home.data) return prev;
  return {
    ...prev,
    home: {
      ...prev.home,
      ticker: cached.ticker,
      data: cached,
    },
  };
}

type CachedCountries = { countries: string[]; fetchedAt: number; source?: "supabase" | "local" };

function safeParseCountries(raw: string | null): CachedCountries | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as { countries?: unknown; fetchedAt?: unknown; source?: unknown };
    if (!parsed || typeof parsed !== "object") return null;
    if (!Array.isArray(parsed.countries)) return null;
    const countries = (parsed.countries as unknown[]).filter((c) => typeof c === "string") as string[];
    const fetchedAt = typeof parsed.fetchedAt === "number" ? parsed.fetchedAt : 0;
    const source =
      parsed.source === "supabase" || parsed.source === "local"
        ? (parsed.source as "supabase" | "local")
        : undefined;
    return { countries, fetchedAt, source };
  } catch {
    return null;
  }
}

function mergeDefaults(saved: PersistedAppState): AppState {
  return {
    home: { ...DEFAULT_STATE.home, ...saved.home, data: null },
    techScanner: {
      ...DEFAULT_STATE.techScanner,
      ...saved.techScanner,
      // Reset transient UI state only
      selectedStock: null,
      // But preserve: results, hasScanned, scannedCount, scanHistory from saved
    },
    comparisonScanner: {
      ...DEFAULT_STATE.comparisonScanner,
      ...saved.comparisonScanner,
      // Keep results from saved state - don't clear them
      loadingSymbols: [], // Only clear loading state
      errors: {},         // Only clear errors
    },
    aiScanner: {
      ...DEFAULT_STATE.aiScanner,
      ...saved.aiScanner,
    },
  };
}

function toPersistedState(full: AppState): PersistedAppState {
  return {
    home: {
      ticker: full.home.ticker,
      chartType: full.home.chartType,
      showEma50: full.home.showEma50,
      showEma200: full.home.showEma200,
      showBB: full.home.showBB,
      showRsi: full.home.showRsi,
      showVolume: full.home.showVolume,
      watchlist: full.home.watchlist,
    },
    techScanner: {
      country: full.techScanner.country,
      results: full.techScanner.results,
      hasScanned: full.techScanner.hasScanned,
      scannedCount: full.techScanner.scannedCount,
      scanHistory: full.techScanner.scanHistory,
      // Note: 'selectedStock' intentionally NOT persisted
      searchTerm: full.techScanner.searchTerm,
      rsiMin: full.techScanner.rsiMin,
      rsiMax: full.techScanner.rsiMax,
      aboveEma50: full.techScanner.aboveEma50,
      aboveEma200: full.techScanner.aboveEma200,
      adxMin: full.techScanner.adxMin,
      adxMax: full.techScanner.adxMax,
      atrMin: full.techScanner.atrMin,
      atrMax: full.techScanner.atrMax,
      stochKMin: full.techScanner.stochKMin,
      stochKMax: full.techScanner.stochKMax,
      rocMin: full.techScanner.rocMin,
      rocMax: full.techScanner.rocMax,
      aboveVwap20: full.techScanner.aboveVwap20,
      volumeAboveSma20: full.techScanner.volumeAboveSma20,
      goldenCross: full.techScanner.goldenCross,
      currentTab: full.techScanner.currentTab,
      marketCapMin: full.techScanner.marketCapMin,
      marketCapMax: full.techScanner.marketCapMax,
      sector: full.techScanner.sector,
      industry: full.techScanner.industry,
    },
    comparisonScanner: {
      symbols: full.comparisonScanner.symbols,
      results: full.comparisonScanner.results,
    },
    aiScanner: {
      country: full.aiScanner.country,
      results: full.aiScanner.results,
      hasScanned: full.aiScanner.hasScanned,
      scannedCount: full.aiScanner.scannedCount,
      scanHistory: full.aiScanner.scanHistory,
      minPrecision: full.aiScanner.minPrecision,
      modelName: full.aiScanner.modelName,
      rfPreset: full.aiScanner.rfPreset,
      rfParamsJson: full.aiScanner.rfParamsJson,
    }
  };
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const supabase = useMemo(() => createSupabaseBrowserClient(), []);

  const [state, setState] = useState<AppState>(DEFAULT_STATE);

  const [countries, setCountries] = useState<string[]>(["Egypt", "USA"]);
  const [countriesLoading, setCountriesLoading] = useState(false);

  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Scan loading states (persist across navigation)
  const [techScanLoading, setTechScanLoading] = useState(false);
  const [techScanError, setTechScanError] = useState<string | null>(null);
  const techScanAbortRef = useRef<AbortController | null>(null);

  const [inventory, setInventory] = useState<any[]>([]);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const isInventoryFetchingRef = useRef(false);

  useEffect(() => {
    if (user) {
      const email = user.email?.toLowerCase();
      const isSystemAdmin = user.app_metadata?.role === "admin" || email === "weeeessd57@gmail.com";
      setIsAdmin(isSystemAdmin);
    } else {
      setIsAdmin(false);
    }
  }, [user]);

  const refreshInventory = useCallback(async () => {
    if (isInventoryFetchingRef.current) return;
    isInventoryFetchingRef.current = true;
    setInventoryLoading(true);
    try {
      const { getInventory } = await import("@/lib/api");
      const next = await getInventory();
      setInventory(next);
    } catch (e) {
      console.error("Failed to fetch inventory:", e);
    } finally {
      setInventoryLoading(false);
      isInventoryFetchingRef.current = false;
    }
  }, []);

  const isCountryActive = useCallback((country: string) => {
    if (isAdmin) return true;
    // Country is active if any exchange for that country has price_count > 0
    return inventory.some(item => item.country === country && item.price_count > 0);
  }, [inventory, isAdmin]);

  const isSymbolActive = useCallback((symbol: string, exchange: string) => {
    if (isAdmin) return true;
    if (symbol.toLowerCase().startsWith("fund_")) return false;
    // Symbol is "active" if its exchange has price data
    // (A more precise check would involve a lookup, but this is consistent with inventory logic)
    return inventory.some(item => item.exchange === exchange && item.price_count > 0);
  }, [inventory, isAdmin]);

  useEffect(() => {
    refreshInventory();
  }, [refreshInventory]);

  const [syncedSymbols, setSyncedSymbols] = useState<any[]>([]);
  const [syncedSymbolsLoading, setSyncedSymbolsLoading] = useState(false);

  const symbolCacheRef = useRef<Map<string, any[]>>(new Map());

  const refreshSyncedSymbols = useCallback(async (country?: string) => {
    const source = isAdmin ? "local" : "supabase";
    const cacheKey = `${source}:${country || "all"}`;

    // 1. Check if we have it in cache
    if (symbolCacheRef.current.has(cacheKey)) {
      setSyncedSymbols(symbolCacheRef.current.get(cacheKey)!);
      return;
    }

    setSyncedSymbolsLoading(true);
    try {
      const { getSyncedSymbols } = await import("@/lib/api");
      const next = await getSyncedSymbols(country, source);

      // 2. Save to cache
      symbolCacheRef.current.set(cacheKey, next);
      setSyncedSymbols(next);
    } catch (e) {
      console.error("Failed to fetch synced symbols:", e);
    } finally {
      setSyncedSymbolsLoading(false);
    }
  }, [isAdmin]);

  const refreshCountries = useCallback(async (opts?: { force?: boolean }) => {
    const cacheTtlMs = 7 * 24 * 60 * 60 * 1000;
    const source = isAdmin ? "local" : "supabase";
    if (!opts?.force) {
      const cached = safeParseCountries(localStorage.getItem(COUNTRIES_KEY));
      if (
        cached?.countries?.length &&
        cached.source === source &&
        Date.now() - cached.fetchedAt < cacheTtlMs
      ) {
        setCountries(cached.countries);
        return;
      }
    }

    setCountriesLoading(true);
    try {
      // For Admins, always show local source. For users, show Supabase (which is filtered)
      const next = await getCountries(source);
      if (Array.isArray(next) && next.length > 0) {
        setCountries(next);
        localStorage.setItem(
          COUNTRIES_KEY,
          JSON.stringify({ countries: next, fetchedAt: Date.now(), source })
        );
      }
    } finally {
      setCountriesLoading(false);
    }
  }, [isAdmin]);

  // Refetch when admin status changes to ensure we have the right list
  useEffect(() => {
    if (isAdmin) {
      void refreshCountries({ force: true });
    }
  }, [isAdmin, refreshCountries]);

  useEffect(() => {
    const cached = safeParseCountries(localStorage.getItem(COUNTRIES_KEY));
    if (cached?.countries?.length) {
      setCountries(cached.countries);
    }
    void refreshCountries();
  }, [refreshCountries]);

  // useAuth moved to top

  // Refs to prevent redundant API calls
  const hasSettingsInitializedRef = useRef(false);
  const lastSettingsFetchedAtRef = useRef<number>(0);
  const lastSettingsUserIdRef = useRef<string | null>(null);
  const SETTINGS_STALE_TIME_MS = 60 * 1000; // 1 minute

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Always try to load from local storage first for immediate UI state
      const local = safeParseState(localStorage.getItem(STORAGE_KEY));

      if (!user) {
        // If not logged in, rely solely on local storage
        if (local) {
          setState((prev) => maybeRestoreHomeData(mergeDefaults(local)));
        } else {
          setState((prev) => maybeRestoreHomeData(DEFAULT_STATE));
        }
        hasSettingsInitializedRef.current = false;
        lastSettingsUserIdRef.current = null;
        return;
      }

      // Skip if already loaded for this user and data is fresh
      const now = Date.now();
      const isSameUser = lastSettingsUserIdRef.current === user.id;
      const isFresh = (now - lastSettingsFetchedAtRef.current) < SETTINGS_STALE_TIME_MS;
      if (hasSettingsInitializedRef.current && isSameUser && isFresh) {
        return;
      }

      const { data: row } = await supabase
        .from("user_settings")
        .select("app_state")
        .eq("user_id", user.id)
        .maybeSingle();

      const remote = row?.app_state as PersistedAppState | undefined;


      const hasRemote = remote && typeof remote === "object" && Object.keys(remote as any).length > 0;

      if (local) {
        await supabase
          .from("user_settings")
          .upsert({ user_id: user.id, app_state: local }, { onConflict: "user_id" });
        // We Keep local storage as a sync mirror
      }

      // Merge strategy: Remote > Local > Default
      // If we have remote, use it. If not, use local.
      const nextState = hasRemote ? mergeDefaults(remote as PersistedAppState) : local ? mergeDefaults(local) : DEFAULT_STATE;
      if (cancelled) return;
      setState(maybeRestoreHomeData(nextState));

      // Mark as initialized
      hasSettingsInitializedRef.current = true;
      lastSettingsFetchedAtRef.current = Date.now();
      lastSettingsUserIdRef.current = user.id;
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [supabase, user]);

  useEffect(() => {
    const persisted = toPersistedState(state);

    // Always save to local storage immediately (or debounced)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));

    if (user) {
      const timeoutId = window.setTimeout(() => {
        void supabase.from("user_settings").upsert({ user_id: user.id, app_state: persisted }, { onConflict: "user_id" });
      }, 1000); // 1s debounce for invalidation

      return () => {
        window.clearTimeout(timeoutId);
      };
    }
  }, [state, supabase, user]);

  const setHome = useCallback((u: Updater<HomeState>) => {
    setState((prev) => ({
      ...prev,
      home: typeof u === "function" ? (u as (p: HomeState) => HomeState)(prev.home) : { ...prev.home, ...u },
    }));
  }, []);

  const setTechScanner = useCallback((u: Updater<TechScannerState>) => {
    setState((prev) => ({
      ...prev,
      techScanner: typeof u === "function" ? (u as (p: TechScannerState) => TechScannerState)(prev.techScanner) : { ...prev.techScanner, ...u },
    }));
  }, []);

  const setComparisonScanner = useCallback((u: Updater<ComparisonScannerState>) => {
    setState((prev) => ({
      ...prev,
      comparisonScanner: typeof u === "function" ? (u as (p: ComparisonScannerState) => ComparisonScannerState)(prev.comparisonScanner) : { ...prev.comparisonScanner, ...u },
    }));
  }, []);

  const setAiScanner = useCallback((u: Updater<AiScannerState>) => {
    setState((prev) => ({
      ...prev,
      aiScanner: typeof u === "function" ? (u as (p: AiScannerState) => AiScannerState)(prev.aiScanner) : { ...prev.aiScanner, ...u },
    }));
  }, []);


  const clearHomeView = useCallback(() => {
    setState((prev) => ({
      ...prev,
      home: { ...prev.home, data: null },
    }));
  }, []);

  const restoreLastHomePredict = useCallback(() => {
    let restored = false;
    setState((prev) => {
      const last = prev.home.predictHistory?.[prev.home.predictHistory.length - 1];
      if (!last) return prev;
      restored = true;
      return {
        ...prev,
        home: {
          ...prev.home,
          ticker: last.ticker,
          data: last.data,
        },
      };
    });
    return restored;
  }, []);

  const runHomePredict = useCallback(
    async (ticker: string, opts?: { signal?: AbortSignal; force?: boolean }) => {
      const normalized = ticker.trim().toUpperCase();
      const now = Date.now();
      const cacheTtlMs = 15 * 60 * 1000;
      const key = JSON.stringify({ ticker: normalized });

      const cached = stateRef.current.home.predictHistory?.find((h) => h.key === key && now - h.createdAt < cacheTtlMs);
      if (cached && !opts?.force) {
        setState((prev) => ({
          ...prev,
          home: {
            ...prev.home,
            ticker: normalized,
            data: cached.data,
          },
        }));
        return;
      }

      setState((prev) => ({
        ...prev,
        home: {
          ...prev.home,
          ticker: normalized,
          data: null,
        },
      }));

      let res;
      try {
        res = await predictStock({ ticker: normalized }, opts?.signal);
      } catch (err: any) {
        if (err.name === 'AbortError') return;
        // On error, clear any old data for this symbol and rethrow
        setState((prev) => {
          const history = Array.isArray(prev.home.predictHistory) ? prev.home.predictHistory : [];
          const filtered = history.filter((h) => h.key !== key);
          return {
            ...prev,
            home: {
              ...prev.home,
              predictHistory: filtered,
              data: null,
            },
          };
        });
        // Also clear localStorage cache for this symbol
        localStorage.removeItem(HOME_PRED_KEY);
        throw err;
      }
      setState((prev) => {
        const history = Array.isArray(prev.home.predictHistory) ? prev.home.predictHistory : [];
        const snapshot = { key, createdAt: Date.now(), ticker: normalized, data: res };
        const deduped = history.filter((h) => h.key !== key);
        const capped = [...deduped, snapshot].slice(-10);
        return {
          ...prev,
          home: {
            ...prev.home,
            ticker: normalized,
            data: res,
            predictHistory: capped,
          },
        };
      });

      localStorage.setItem(HOME_PRED_KEY, JSON.stringify({ data: res, savedAt: Date.now() }));
    },
    []
  );


  const clearTechScannerView = useCallback(() => {
    setState((prev) => ({
      ...prev,
      techScanner: {
        ...prev.techScanner,
        results: [],
        hasScanned: false,
        scannedCount: 0,
        selectedStock: null,
      },
    }));
  }, []);

  const restoreLastTechScan = useCallback(() => {
    let restored = false;
    setState((prev) => {
      const last = prev.techScanner.scanHistory?.[prev.techScanner.scanHistory.length - 1];
      if (!last) return prev;
      restored = true;
      return {
        ...prev,
        techScanner: {
          ...prev.techScanner,
          results: last.results,
          hasScanned: true,
          scannedCount: last.scannedCount,
          selectedStock: null,
        },
      };
    });
    return restored;
  }, []);

  const runTechScan = useCallback(
    async (opts?: { force?: boolean }) => {
      // Create internal abort controller
      if (techScanAbortRef.current) {
        techScanAbortRef.current.abort();
      }
      const controller = new AbortController();
      techScanAbortRef.current = controller;

      const s = stateRef.current.techScanner;
      const filter: TechFilter = {
        country: s.country,
        limit: 100,
        rsi_min: s.rsiMin ? parseFloat(s.rsiMin) : undefined,
        rsi_max: s.rsiMax ? parseFloat(s.rsiMax) : undefined,
        above_ema50: s.aboveEma50,
        above_ema200: s.aboveEma200,
        adx_min: s.adxMin ? parseFloat(s.adxMin) : undefined,
        adx_max: s.adxMax ? parseFloat(s.adxMax) : undefined,
        atr_min: s.atrMin ? parseFloat(s.atrMin) : undefined,
        atr_max: s.atrMax ? parseFloat(s.atrMax) : undefined,
        stoch_k_min: s.stochKMin ? parseFloat(s.stochKMin) : undefined,
        stoch_k_max: s.stochKMax ? parseFloat(s.stochKMax) : undefined,
        roc_min: s.rocMin ? parseFloat(s.rocMin) : undefined,
        roc_max: s.rocMax ? parseFloat(s.rocMax) : undefined,
        above_vwap20: s.aboveVwap20,
        volume_above_sma20: s.volumeAboveSma20,
        market_cap_min: s.marketCapMin ? parseFloat(s.marketCapMin) : undefined,
        market_cap_max: s.marketCapMax ? parseFloat(s.marketCapMax) : undefined,
        sector: s.sector || undefined,
        industry: s.industry || undefined,
        golden_cross: s.goldenCross,
      };

      const now = Date.now();
      const cacheTtlMs = 15 * 60 * 1000;
      const key = JSON.stringify({
        country: filter.country,
        limit: filter.limit,
        rsi_min: filter.rsi_min,
        rsi_max: filter.rsi_max,
        market_cap_min: filter.market_cap_min,
        market_cap_max: filter.market_cap_max,
        sector: filter.sector,
        industry: filter.industry,
        golden_cross: filter.golden_cross,
        above_ema50: filter.above_ema50,
        above_ema200: filter.above_ema200,
        adx_min: filter.adx_min,
        adx_max: filter.adx_max,
        atr_min: filter.atr_min,
        atr_max: filter.atr_max,
        stoch_k_min: filter.stoch_k_min,
        stoch_k_max: filter.stoch_k_max,
        roc_min: filter.roc_min,
        roc_max: filter.roc_max,
        above_vwap20: filter.above_vwap20,
        volume_above_sma20: filter.volume_above_sma20,
      });

      const cached = stateRef.current.techScanner.scanHistory?.find((h) => h.key === key && now - h.createdAt < cacheTtlMs);
      if (cached && !opts?.force) {
        setState((prev) => ({
          ...prev,
          techScanner: {
            ...prev.techScanner,
            results: cached.results,
            hasScanned: true,
            scannedCount: cached.scannedCount,
            selectedStock: null,
          },
        }));
        return;
      }

      setTechScanLoading(true);
      setTechScanError(null);
      setState((prev) => ({
        ...prev,
        techScanner: {
          ...prev.techScanner,
          results: [],
          hasScanned: false,
          scannedCount: 0,
          selectedStock: null,
        },
      }));

      try {
        const res = await scanTech(filter, controller.signal);

        let next = res.results || [];
        if (s.goldenCross) {
          next = next.filter(r => r.ema50 > r.ema200);
        }

        setState((prev) => {
          const history = Array.isArray(prev.techScanner.scanHistory) ? prev.techScanner.scanHistory : [];
          const snapshot = { key, createdAt: Date.now(), filter, results: next, scannedCount: res.scanned_count };
          const deduped = history.filter((h) => h.key !== key);
          const capped = [...deduped, snapshot].slice(-5);
          return {
            ...prev,
            techScanner: {
              ...prev.techScanner,
              results: next,
              hasScanned: true,
              scannedCount: res.scanned_count,
              scanHistory: capped,
            },
          };
        });
      } catch (err: any) {
        if (err.name === 'AbortError') {
          console.log('Tech Scan aborted');
        } else {
          setTechScanError(err instanceof Error ? err.message : 'Scan failed');
        }
      } finally {
        setTechScanLoading(false);
        techScanAbortRef.current = null;
      }
    },
    []
  );

  const stopTechScan = useCallback(() => {
    if (techScanAbortRef.current) {
      techScanAbortRef.current.abort();
      techScanAbortRef.current = null;
      setTechScanLoading(false);
    }
  }, []);

  const resetTechScanner = useCallback(() => {
    setState((prev) => ({ ...prev, techScanner: DEFAULT_STATE.techScanner }));
  }, []);


  const addSymbolToCompare = useCallback(async (symbol: string) => {
    const s = symbol.toUpperCase().trim();
    if (!s) return;

    // Check if we already have this symbol's data cached
    const existingResult = stateRef.current.comparisonScanner.results[s];
    if (existingResult) {
      // Just add to symbols list if not already there
      setComparisonScanner(prev => {
        if (prev.symbols.includes(s)) return prev;
        return { ...prev, symbols: [...prev.symbols, s] };
      });
      return;
    }

    setComparisonScanner(prev => {
      if (prev.symbols.includes(s)) return prev;
      return { ...prev, symbols: [...prev.symbols, s] };
    });

    setComparisonScanner(prev => ({
      ...prev,
      loadingSymbols: Array.from(new Set([...prev.loadingSymbols, s])),
      errors: { ...prev.errors, [s]: "" }
    }));

    try {
      const res = await predictStock({ ticker: s });
      setComparisonScanner(prev => ({
        ...prev,
        results: { ...prev.results, [s]: res },
        loadingSymbols: prev.loadingSymbols.filter(sym => sym !== s)
      }));
    } catch (err) {
      setComparisonScanner(prev => ({
        ...prev,
        errors: { ...prev.errors, [s]: err instanceof Error ? err.message : "Failed to load" },
        loadingSymbols: prev.loadingSymbols.filter(sym => sym !== s)
      }));
    }
  }, [setComparisonScanner]);

  const addSymbolsToCompare = useCallback(async (newSymbols: string[]) => {
    const normalized = newSymbols.map(s => s.toUpperCase().trim()).filter(Boolean);
    if (normalized.length === 0) return;

    // Filter out already present ones
    const toAdd = normalized.filter(s => !stateRef.current.comparisonScanner.symbols.includes(s));
    if (toAdd.length === 0) return;

    // Separate symbols that need fetching from those already cached
    const currentResults = stateRef.current.comparisonScanner.results;
    const toFetch = toAdd.filter(s => !currentResults[s]);
    const alreadyCached = toAdd.filter(s => currentResults[s]);

    // Phase 1: Add all symbols to list (both cached and to-fetch)
    setComparisonScanner(prev => ({
      ...prev,
      symbols: Array.from(new Set([...prev.symbols, ...toAdd])),
      loadingSymbols: Array.from(new Set([...prev.loadingSymbols, ...toFetch])),
      errors: toFetch.reduce((acc, s) => ({ ...acc, [s]: "" }), prev.errors)
    }));

    // Phase 2: Only fetch symbols that aren't cached
    if (toFetch.length > 0) {
      await Promise.all(toFetch.map(async (s) => {
        try {
          const res = await predictStock({ ticker: s });
          setComparisonScanner(prev => ({
            ...prev,
            results: { ...prev.results, [s]: res },
            loadingSymbols: prev.loadingSymbols.filter(sym => sym !== s)
          }));
        } catch (err) {
          setComparisonScanner(prev => ({
            ...prev,
            errors: { ...prev.errors, [s]: err instanceof Error ? err.message : "Failed to load" },
            loadingSymbols: prev.loadingSymbols.filter(sym => sym !== s)
          }));
        }
      }));
    }
  }, [setComparisonScanner]);

  const removeSymbolFromCompare = useCallback((symbol: string) => {
    setComparisonScanner(prev => ({
      ...prev,
      symbols: prev.symbols.filter(s => s !== symbol),
      results: Object.fromEntries(Object.entries(prev.results).filter(([k]) => k !== symbol)),
      loadingSymbols: prev.loadingSymbols.filter(s => s !== symbol),
      errors: Object.fromEntries(Object.entries(prev.errors).filter(([k]) => k !== symbol)),
    }));
  }, [setComparisonScanner]);

  const addSymbolToWatchlist = useCallback((symbol: string) => {
    const s = symbol.toUpperCase().trim();
    if (!s) return;
    setHome(prev => {
      const current = prev.watchlist || [];
      if (current.includes(s)) return prev;
      return { ...prev, watchlist: [...current, s] };
    });
  }, [setHome]);

  const removeSymbolFromWatchlist = useCallback((symbol: string) => {
    setHome(prev => ({
      ...prev,
      watchlist: (prev.watchlist || []).filter(t => t !== symbol)
    }));
  }, [setHome]);

  const clearComparison = useCallback(() => {
    setComparisonScanner(DEFAULT_STATE.comparisonScanner);
  }, [setComparisonScanner]);

  const value = useMemo<AppStateContextType>(
    () => ({
      state,
      countries,
      countriesLoading,
      refreshCountries,
      setHome,
      setTechScanner,
      setComparisonScanner,
      setAiScanner,
      syncedSymbols,
      syncedSymbolsLoading,
      refreshSyncedSymbols,
      runHomePredict,
      clearHomeView,
      restoreLastHomePredict,
      // Tech Scan
      techScanLoading,
      techScanError,
      runTechScan,
      stopTechScan,
      clearTechScannerView,
      restoreLastTechScan,
      // Comparison
      addSymbolToCompare,
      addSymbolsToCompare,
      removeSymbolFromCompare,
      clearComparison,
      resetTechScanner,
      addSymbolToWatchlist,
      removeSymbolFromWatchlist,
      inventory,
      inventoryLoading,
      refreshInventory,
      isCountryActive,
      isSymbolActive,
      isAdmin,
    }),
    [
      state,
      countries,
      countriesLoading,
      refreshCountries,
      setHome,
      setTechScanner,
      setComparisonScanner,
      setAiScanner,
      syncedSymbols,
      syncedSymbolsLoading,
      refreshSyncedSymbols,
      runHomePredict,
      clearHomeView,
      restoreLastHomePredict,
      techScanLoading,
      techScanError,
      runTechScan,
      stopTechScan,
      clearTechScannerView,
      restoreLastTechScan,
      addSymbolToCompare,
      addSymbolsToCompare,
      removeSymbolFromCompare,
      clearComparison,
      resetTechScanner,
      addSymbolToWatchlist,
      removeSymbolFromWatchlist,
      inventory,
      inventoryLoading,
      refreshInventory,
      isCountryActive,
      isSymbolActive,
      isAdmin,
    ]
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within an AppStateProvider");
  return ctx;
}
