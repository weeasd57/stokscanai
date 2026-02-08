import type { PredictResponse } from "@/lib/types";

export type PredictParams = {
  ticker: string;
  exchange?: string;
  fromDate?: string;
  toDate?: string;
  includeFundamentals?: boolean;
  rfPreset?: string;
  rfParams?: Record<string, unknown>;
  modelName?: string;
  forceLocal?: boolean;
  targetPct?: number;
  stopLossPct?: number;
  lookForwardDays?: number;
  buyThreshold?: number;
};

export async function predictStock(params: PredictParams, signal?: AbortSignal): Promise<PredictResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  let ticker = params.ticker.trim().toUpperCase();
  let exchange = params.exchange?.toUpperCase();

  if ((!exchange || exchange.trim() === "") && ticker.includes(".")) {
    const parts = ticker.split(".").filter(Boolean);
    if (parts.length >= 2) {
      exchange = parts[parts.length - 1];
      ticker = parts.slice(0, -1).join(".");
    }
  }

  if (!exchange || exchange === "") {
    if (!ticker.includes(".")) {
      exchange = "EGX";
    }
  }

  const payload = {
    ticker,
    exchange: exchange ?? null,
    from_date: params.fromDate ?? "2020-01-01",
    to_date: params.toDate ?? null,
    include_fundamentals: params.includeFundamentals ?? true,
    rf_preset: params.rfPreset ?? null,
    rf_params: params.rfParams ?? null,
    model_name: params.modelName ?? null,
    force_local: params.forceLocal ?? true,
    target_pct: params.targetPct ?? 0.15,
    stop_loss_pct: params.stopLossPct ?? 0.05,
    look_forward_days: params.lookForwardDays ?? 20,
    buy_threshold: params.buyThreshold ?? 0.45,
  };

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal,
    });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/predict`) : await doFetch("/api/predict");
  } catch (e) {
    if (baseUrl) {
      res = await doFetch("/api/predict");
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data?.detail) msg = data.detail;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }

  return (await res.json()) as PredictResponse;
}

export type SymbolResult = {
  symbol: string;
  exchange: string;
  name: string;
  country: string;
  hasLocal?: boolean;
};

export type CountriesResponse = {
  countries: string[];
};

export type SymbolSearchResponse = {
  results: SymbolResult[];
};

export type LocalModelMeta = {
  name: string;
  size_bytes?: number;
  size_mb?: number;
  created_at?: string;
  modified_at?: string;
  type?: string;
  num_features?: number;
  num_parameters?: number;
  trainingSamples?: number;
  target_pct?: number;
  stop_loss_pct?: number;
  look_forward_days?: number;
  buyThreshold?: number;
};

export type LocalModelsResponse = {
  models: (string | LocalModelMeta)[];
};

export async function getLocalModels(): Promise<(string | LocalModelMeta)[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const endpoint = "/models/local";

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store" });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api${endpoint}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch models (${res.status})`);
  }

  const data = (await res.json()) as LocalModelsResponse;
  return data.models ?? [];
}

export type DateSymbolResult = {
  symbol: string;
  exchange: string;
  name: string;
  rowCount?: number;
};

export async function getSymbolsByDate(params: {
  start: string;
  end: string;
  exchange?: string;
  limit?: number;
  searchTerm?: string;
}): Promise<DateSymbolResult[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const query = new URLSearchParams({ start: params.start, end: params.end });
  if (params.exchange) query.set("exchange", params.exchange);
  if (params.limit) query.set("limit", String(params.limit));
  if (params.searchTerm) query.set("search_term", params.searchTerm);

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store" });
  }

  let res: Response;
  try {
    const endpoint = `/symbols/by-date?${query.toString()}`;
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/symbols/by-date?${query.toString()}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch symbols (${res.status})`);
  }

  const data = (await res.json()) as { results: DateSymbolResult[] };
  return data.results;
}

export async function getSymbolsForExchange(exchange: string): Promise<DateSymbolResult[]> {
  try {
    const res = await fetch(`/api/admin/db-symbols/${exchange}?mode=prices`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Failed to fetch symbols (${res.status})`);
    }
    const data = (await res.json()) as any[];
    return data.map((item) => ({
      symbol: item.symbol,
      exchange: exchange,
      name: item.name || "",
      rowCount: item.rowCount || item.row_count || 0,
    }));
  } catch (error) {
    console.error("Failed to fetch symbols from db-symbols:", error);
    return [];
  }
}

export async function getInventory(): Promise<any[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const endpoint = "/symbols/inventory";

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store" });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api${endpoint}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch inventory (${res.status})`);
  }

  const data = (await res.json()) as { inventory: any[] };
  return data.inventory;
}

export async function getCountries(source?: "supabase" | "local"): Promise<string[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const params = new URLSearchParams();
  if (source) params.set("source", source);

  const endpoint = `/symbols/countries?${params.toString()}`;

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store" });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api${endpoint}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch countries (${res.status})`);
  }

  const data = (await res.json()) as CountriesResponse;
  return data.countries;
}

export async function searchSymbols(
  query: string,
  country?: string,
  limit: number = 50,
  signal?: AbortSignal,
  source?: "supabase" | "local"
): Promise<SymbolResult[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  const params = new URLSearchParams({ q: query, limit: String(limit) });
  if (country) params.set("country", country);
  if (source) params.set("source", source);

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store", signal });
  }

  let res: Response;
  try {
    const endpoint = `/symbols/search?${params.toString()}`;
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/symbols/search?${params.toString()}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to search symbols (${res.status})`);
  }

  const data = (await res.json()) as SymbolSearchResponse;
  return data.results;
}

export async function getSyncedSymbols(country?: string, source?: "supabase" | "local"): Promise<SymbolResult[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const params = new URLSearchParams();
  if (country) params.set("country", country);
  if (source) params.set("source", source);

  async function doFetch(url: string) {
    return await fetch(url, { cache: "no-store" });
  }

  let res: Response;
  try {
    const endpoint = `/symbols/synced?${params.toString()}`;
    res = baseUrl ? await doFetch(`${baseUrl}${endpoint}`) : await doFetch(`/api${endpoint}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/symbols/synced?${params.toString()}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch synced symbols (${res.status})`);
  }
  const data = (await res.json()) as SymbolSearchResponse;
  return data.results;
}

// Scan AI
export type ScanResult = {
  symbol: string;
  exchange?: string | null;
  name: string;
  last_close: number;
  precision: number;
  signal: string;
  confidence: string;
  logo_url?: string | null;
  status?: "win" | "loss" | "pending" | "open" | "hit_stop" | "hit_target" | null;
  profit_loss_pct?: number | null;
  top_reasons?: string[];
  target_price?: number;
  stop_loss?: number;
  id?: string;
  created_at?: string;
  updated_at?: string;
  exit_price?: number;
  features?: number[] | null;
  technical_score?: number;
  fundamental_score?: number;
  council_score?: number;
  consensus_ratio?: string;
};

export type ScanResponse = {
  results: ScanResult[];
  scanned_count: number;
};

export type ScanAiParams = {
  country: string;
  scanAll: boolean;
  limit: number;
  minPrecision: number;
  rfPreset: "fast" | "default" | "accurate";
  rfParamsJson: string;
  rfParams: Record<string, unknown> | null;
  modelName?: string;
  from_date?: string;
  to_date?: string;
  target_pct?: number;
  stop_loss_pct?: number;
  look_forward_days?: number;
  buy_threshold?: number;
  councilModel?: string;
  validatorModel?: string;
};

export async function scanAiFastWithParams(params: ScanAiParams, signal?: AbortSignal): Promise<ScanResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  const query = new URLSearchParams({
    country: params.country,
    limit: String(params.limit),
    min_precision: String(params.minPrecision ?? 0.1),
    model_name: params.modelName ?? "",
    from_date: params.from_date || new Date(Date.now() - 300 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    target_pct: String(params.target_pct ?? 0.10),
    stop_loss_pct: String(params.stop_loss_pct ?? 0.05),
    look_forward_days: String(params.look_forward_days ?? 20),
    buy_threshold: String(params.buy_threshold ?? 0.45),
    council_model: params.councilModel ?? "",
    validator_model: params.validatorModel ?? "",
  });
  if (params.to_date) {
    query.set("to_date", params.to_date);
  }

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      signal,
    });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/scan/fast?${query}`) : await doFetch(`/api/scan/fast?${query}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/scan/fast?${query}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Scan failed(${res.status})`;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data?.detail) msg = data.detail;
    } catch {
      try {
        const text = await res.text();
        if (text) msg = text;
      } catch {
      }
    }
    throw new Error(msg);
  }

  return (await res.json()) as ScanResponse;
}

export async function scanAiWithParams(params: ScanAiParams, signal?: AbortSignal): Promise<ScanResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rf_preset: params.rfPreset,
        rf_params: params.rfParams ?? null,
        model_name: params.modelName ?? null,
      }),
      cache: "no-store",
      signal: signal,
    });
  }

  const query = new URLSearchParams({
    country: params.country,
    limit: String(params.limit),
    min_precision: String(params.minPrecision),
  });

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/scan/ai?${query}`) : await doFetch(`/api/scan/ai?${query}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/scan/ai?${query}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Scan failed(${res.status})`;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data?.detail) msg = data.detail;
    } catch {
      try {
        const text = await res.text();
        if (text) msg = text;
      } catch {
      }
    }
    throw new Error(msg);
  }

  return (await res.json()) as ScanResponse;
}

export async function scanAi(country: string = "Egypt", signal?: AbortSignal): Promise<ScanResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      signal: signal
    });
  }

  const query = new URLSearchParams({ country, limit: "50", min_precision: "0.6" });
  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/scan/ai?${query}`) : await doFetch(`/api/scan/ai?${query}`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/scan/ai?${query}`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Scan failed(${res.status})`;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data?.detail) msg = data.detail;
    } catch {
      try {
        const text = await res.text();
        if (text) msg = text;
      } catch {
        // ignore
      }
    }
    throw new Error(msg);
  }

  return (await res.json()) as ScanResponse;
}

// Technical Scan Types
export type TechFilter = {
  country?: string;
  limit?: number;
  rsi_min?: number;
  rsi_max?: number;
  min_price?: number;
  above_ema50?: boolean;
  above_ema200?: boolean;
  below_ema50?: boolean;
  adx_min?: number;
  adx_max?: number;
  atr_min?: number;
  atr_max?: number;
  stoch_k_min?: number;
  stoch_k_max?: number;
  roc_min?: number;
  roc_max?: number;
  above_vwap20?: boolean;
  volume_above_sma20?: boolean;
  market_cap_min?: number;
  market_cap_max?: number;
  sector?: string;
  industry?: string;
  golden_cross?: boolean;
  use_ai_filter?: boolean;
  min_ai_precision?: number;
};

export type TechResult = {
  symbol: string;
  name: string;
  last_close: number;
  rsi: number;
  volume: number;
  ema50: number;
  ema200: number;
  momentum: number;
  atr14?: number;
  adx14?: number;
  stoch_k?: number;
  stoch_d?: number;
  cci20?: number;
  vwap20?: number;
  roc12?: number;
  vol_sma20?: number;
  change_p: number;
  market_cap?: number;
  pe_ratio?: number;
  eps?: number;
  dividend_yield?: number;
  sector?: string;
  industry?: string;
  beta?: number;
  logo_url?: string | null;
};

export type TechResponse = {
  results: TechResult[];
  scanned_count: number;
};

export async function scanTech(filter: TechFilter, signal?: AbortSignal): Promise<TechResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        country: filter.country ?? "Egypt",
        limit: filter.limit ?? 50,
        rsi_min: filter.rsi_min,
        rsi_max: filter.rsi_max,
        min_price: filter.min_price,
        above_ema50: filter.above_ema50 ?? false,
        above_ema200: filter.above_ema200 ?? false,
        below_ema50: filter.below_ema50 ?? false,
        adx_min: filter.adx_min,
        adx_max: filter.adx_max,
        atr_min: filter.atr_min,
        atr_max: filter.atr_max,
        stoch_k_min: filter.stoch_k_min,
        stoch_k_max: filter.stoch_k_max,
        roc_min: filter.roc_min,
        roc_max: filter.roc_max,
        above_vwap20: filter.above_vwap20 ?? false,
        volume_above_sma20: filter.volume_above_sma20 ?? false,
        market_cap_min: filter.market_cap_min,
        market_cap_max: filter.market_cap_max,
        sector: filter.sector,
        industry: filter.industry,
        golden_cross: filter.golden_cross ?? false,
        use_ai_filter: filter.use_ai_filter ?? false,
        min_ai_precision: filter.min_ai_precision ?? 0.6
      }),
      cache: "no-store",
      signal: signal
    });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/scan/technical`) : await doFetch("/api/scan/technical");
  } catch (e) {
    if (baseUrl) {
      res = await doFetch("/api/scan/technical");
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Scan failed(${res.status})`;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data?.detail) msg = data.detail;
    } catch {
      try {
        const text = await res.text();
        if (text) msg = text;
      } catch {
        // ignore
      }
    }
    throw new Error(msg);
  }

  return (await res.json()) as TechResponse;
}

export async function scanAiSingle(symbol: string, exchange?: string, min_precision: number = 0.6, signal?: AbortSignal): Promise<ScanResult | null> {
  const res = await fetch("/api/scan/ai/single", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, exchange, min_precision }),
    signal: signal
  });
  if (!res.ok) return null;
  return res.json();
}

export interface AdminConfig {
  priceSource: string;
  fundSource: string;
  maxWorkers: number;
  enabledModels: string[];
  modelAliases?: Record<string, string>;
  scanDays?: number;
}

export async function getAdminConfig(): Promise<AdminConfig> {
  const res = await fetch("/api/admin/config");
  if (!res.ok) throw new Error("Failed to fetch admin config");
  return res.json();
}

export async function updateAdminConfig(config: Partial<AdminConfig>): Promise<AdminConfig> {
  const res = await fetch("/api/admin/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config)
  });
  if (!res.ok) throw new Error("Failed to update admin config");
  return res.json();
}

export type NewsArticle = {
  title: string;
  description: string;
  content: string;
  url: string;
  image: string;
  publishedAt: string;
  source: {
    name: string;
    url: string;
  };
};

export type NewsResponse = {
  totalArticles: number;
  articles: NewsArticle[];
};

export async function fetchStockNews(query: string, limit: number = 3): Promise<NewsArticle[]> {
  const apiKey = process.env.NEXT_PUBLIC_GNEWS_API_KEY;
  if (!apiKey) {
    return [];
  }

  try {
    const q = encodeURIComponent(query);
    const url = `https://gnews.io/api/v4/search?q=${q}&lang=en&max=${limit}&token=${apiKey}`;
    const res = await fetch(url);
    if (!res.ok) return [];

    const data = (await res.json()) as NewsResponse;
    return data.articles || [];
  } catch (e) {
    console.error("Failed to fetch news:", e);
    return [];
  }
}

export async function evaluateScan(scanId: string): Promise<{ count: number; message: string }> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const url = baseUrl ? `${baseUrl}/scan/fast/evaluate/${scanId}` : `/api/scan/fast/evaluate/${scanId}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to evaluate scan performance");
  return await response.json();
}

export async function getBacktests(model?: string): Promise<any[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const url = model ? `${baseUrl}/backtests?model=${encodeURIComponent(model)}` : `${baseUrl}/backtests`;
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to fetch backtests");
  return await response.json();
}

export async function getBacktestTrades(backtestId: string): Promise<any[]> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const response = await fetch(`${baseUrl}/backtests/${backtestId}/trades`);
  if (!response.ok) throw new Error("Failed to fetch backtest trades");
  return await response.json();
}

export async function deleteBacktest(id: string): Promise<void> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const response = await fetch(`${baseUrl}/backtests/${id}`, {
    method: "DELETE"
  });
  if (!response.ok) throw new Error("Failed to delete backtest");
}

export async function updateBacktestVisibility(id: string, isPublic: boolean): Promise<void> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const response = await fetch(`${baseUrl}/backtests/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_public: isPublic })
  });
  if (!response.ok) throw new Error("Failed to update backtest visibility");
}

export type AlpacaAsset = {
  symbol: string;
  name: string;
  exchange: string;
  class_name: string;
  status: string;
  tradable: boolean;
  marginable: boolean;
  shortable: boolean;
  easy_to_borrow: boolean;
  fractionable: boolean;
};

export type AlpacaAccountInfo = {
  account_number: string;
  status: string;
  crypto_status: string;
  currency: string;
  buying_power: string;
  cash: string;
  portfolio_value: string;
  pattern_day_trader: boolean;
  trading_blocked: boolean;
  transfers_blocked: boolean;
  account_blocked: boolean;
  created_at: string;
};

export async function getAlpacaAccount(): Promise<AlpacaAccountInfo> {
  const res = await fetch("/api/alpaca/account", { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca account" }));
    throw new Error(error.detail || "Failed to fetch Alpaca account");
  }
  return res.json();
}

export type AlpacaPositionInfo = {
  symbol: string;
  qty: string;
  side?: string | null;
  market_value?: string | null;
  unrealized_intraday_pl?: string | null;
  unrealized_pl?: string | null;
};

export async function getAlpacaPositions(): Promise<AlpacaPositionInfo[]> {
  const res = await fetch("/api/alpaca/positions", { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca positions" }));
    throw new Error(error.detail || "Failed to fetch Alpaca positions");
  }
  return res.json();
}

export async function getAlpacaAssets(
  exchange?: string,
  assetClass: "us_equity" | "crypto" = "us_equity"
): Promise<AlpacaAsset[]> {
  const params = new URLSearchParams();
  if (exchange) params.set("exchange", exchange);
  params.set("asset_class", assetClass);
  params.set("source", "local");
  const url = `/api/alpaca/assets?${params.toString()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca assets" }));
    throw new Error(error.detail || "Failed to fetch Alpaca assets");
  }
  return res.json();
}

export async function syncAlpacaSymbols(
  symbols: string[],
  opts?: { assetClass?: "us_equity" | "crypto"; exchange?: string | null; source?: "local" | "live" }
): Promise<{ success: boolean; synced_count: number; saved_count?: number }> {
  const res = await fetch("/api/alpaca/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbols,
      asset_class: opts?.assetClass ?? "us_equity",
      exchange: opts?.exchange ?? undefined,
      source: opts?.source ?? "local",
    }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to sync Alpaca symbols" }));
    throw new Error(error.detail || "Failed to sync Alpaca symbols");
  }
  return res.json();
}

export async function updateAlpacaLocalCache(
  markets?: Array<"us_equity" | "crypto">
): Promise<{ success: boolean; count: number; filename: string }> {
  const res = await fetch("/api/alpaca/update-local-cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(markets ? { markets } : {}),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to update Alpaca local cache" }));
    throw new Error(error.detail || "Failed to update Alpaca local cache");
  }
  return res.json();
}

export type AlpacaExchange = {
  name: string;
  asset_count: number;
};

export async function getAlpacaExchanges(assetClass: "us_equity" | "crypto" = "us_equity"): Promise<AlpacaExchange[]> {
  const params = new URLSearchParams();
  params.set("asset_class", assetClass);
  params.set("source", "local");
  const res = await fetch(`/api/alpaca/exchanges?${params.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca exchanges" }));
    throw new Error(error.detail || "Failed to fetch Alpaca exchanges");
  }
  const data = await res.json();
  return data;
}

export type AlpacaCacheMeta = {
  exists: boolean;
  asset_class: "us_equity" | "crypto";
  updated_at?: string;
  total_assets?: number;
};

export async function getAlpacaCacheMeta(assetClass: "us_equity" | "crypto" = "us_equity"): Promise<AlpacaCacheMeta> {
  const params = new URLSearchParams();
  params.set("asset_class", assetClass);
  const res = await fetch(`/api/alpaca/cache-meta?${params.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca cache meta" }));
    throw new Error(error.detail || "Failed to fetch Alpaca cache meta");
  }
  return res.json();
}

export type AlpacaSupabaseStats = {
  asset_class: "us_equity" | "crypto";
  exchange_filter?: string | null;
  alpaca_exchanges?: string[];
  alpaca_assets_cache?: { rows: number; last_updated_at?: string | null };
  stock_prices?: { rows: number; last_date?: string | null };
  stock_bars_intraday?: {
    rows: number;
    last_ts?: string | null;
    by_timeframe?: Record<"1m" | "1h" | "1d", number>;
  };
};

export type CryptoSymbolStat = {
  symbol: string;
  rows_count: number;
  first_ts: string | null;
  last_ts: string | null;
};

export async function getAlpacaSupabaseStats(
  assetClass: "us_equity" | "crypto" = "us_equity",
  exchange?: string
): Promise<AlpacaSupabaseStats> {
  const params = new URLSearchParams();
  params.set("asset_class", assetClass);
  if (exchange) params.set("exchange", exchange);
  const res = await fetch(`/api/alpaca/supabase-stats?${params.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch Alpaca Supabase stats" }));
    throw new Error(error.detail || "Failed to fetch Alpaca Supabase stats");
  }
  return res.json();
}

export async function syncAlpacaPrices(
  symbols: string[],
  opts: {
    assetClass: "us_equity" | "crypto";
    exchange?: string | null;
    days: number;
    source?: "local" | "live" | "tradingview" | "binance";
    timeframe?: "1m" | "1h" | "1d";
  }
): Promise<{
  success: boolean;
  symbols: number;
  rows_upserted: number;
  days: number;
  timeframe?: "1m" | "1h" | "1d";
  volume_total?: number;
  volume_missing?: number;
}> {
  const res = await fetch("/api/alpaca/sync-prices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbols,
      asset_class: opts.assetClass,
      exchange: opts.exchange ?? undefined,
      days: opts.days,
      source: opts.source ?? "local",
      timeframe: opts.timeframe ?? "1d",
    }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to sync Alpaca prices" }));
    throw new Error(error.detail || "Failed to sync Alpaca prices");
  }
  return res.json();
}

export async function getCryptoSymbolStats(
  timeframe: "1m" | "1h" | "1d" = "1h"
): Promise<CryptoSymbolStat[]> {
  const params = new URLSearchParams();
  params.set("timeframe", timeframe);
  const res = await fetch(`/api/alpaca/crypto-symbols-stats?${params.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to fetch crypto symbol stats" }));
    throw new Error(error.detail || "Failed to fetch crypto symbol stats");
  }
  return res.json();
}

export async function deleteCryptoBars(
  symbols: string[],
  timeframe: "1m" | "1h" | "1d" = "1h"
): Promise<{ success: boolean; deleted: number; symbols: number; timeframe: string }> {
  const res = await fetch("/api/alpaca/crypto-delete-bars", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols, timeframe }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to delete crypto bars" }));
    throw new Error(error.detail || "Failed to delete crypto bars");
  }
  return res.json();
}
