import type { PredictResponse } from "@/lib/types";

export type PredictParams = {
  ticker: string;
  exchange?: string;
  fromDate?: string;
  includeFundamentals?: boolean;
  rfPreset?: string;
  rfParams?: Record<string, unknown>;
  modelName?: string;
};

export async function predictStock(params: PredictParams): Promise<PredictResponse> {
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

  // Auto-detect EGX if exchange is missing and ticker is known OR looks like an EGX ticker
  // User reported RTVC failing; it should be RTVC.EGX
  if (!exchange || exchange === "") {
    // If it's 4-5 chars and all alpha, it's likely an EGX ticker in this app context
    // Or we can just check if it doesn't already have one.
    // For safety, we only append if it doesn't have a dot and is common.
    // However, the user specifically mentioned RTVC.
    if (!ticker.includes(".")) {
      exchange = "EGX";
    }
  }

  const payload = {
    ticker,
    exchange: exchange ?? null,
    from_date: params.fromDate ?? "2020-01-01",
    include_fundamentals: params.includeFundamentals ?? true,
    rf_preset: params.rfPreset ?? null,
    rf_params: params.rfParams ?? null,
    model_name: params.modelName ?? null,
  };

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
  }

  let res: Response;
  try {
    res = baseUrl ? await doFetch(`${baseUrl}/predict`) : await doFetch("/api/predict");
  } catch (e) {
    // If external API is configured but unreachable, fallback to internal route.
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

// Types for symbol search
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

// Get symbols for an exchange (fallback from db-inventory)
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

// Get list of available countries
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

// Search symbols with optional country filter
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

// Fetch all synced symbols for a country
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
};

export async function scanAiWithParams(params: ScanAiParams, signal?: AbortSignal): Promise<ScanResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  async function doFetch(url: string) {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rf_preset: params.rfPreset, rf_params: params.rfParams ?? null }),
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
    let msg = `Scan failed (${res.status})`;
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
    let msg = `Scan failed (${res.status})`;
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

// Scan Technical
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
    res = baseUrl ? await doFetch(`${baseUrl}/scan/technical`) : await doFetch(`/api/scan/technical`);
  } catch (e) {
    if (baseUrl) {
      res = await doFetch(`/api/scan/technical`);
    } else {
      throw e;
    }
  }

  if (!res.ok) {
    let msg = `Scan failed (${res.status})`;
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
  return await res.json();
}


// News API
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
    // Return mock news or empty if no key
    // For now, let's just return empty to fail gracefully
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
