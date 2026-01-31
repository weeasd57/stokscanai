import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Body, Request
from pydantic import BaseModel, field_validator
from eodhd import APIClient

from api.stock_ai import get_stock_data_eodhd, add_technical_indicators, check_local_cache, is_ticker_synced, get_company_fundamentals
from api.symbols_local import load_symbols_for_country

router = APIRouter(prefix="/scan", tags=["scan"])

class TechFilter(BaseModel):
    country: str = "Egypt"
    limit: int = 50
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    min_price: Optional[float] = None
    above_ema50: bool = False
    above_ema200: bool = False
    below_ema50: bool = False
    adx_min: Optional[float] = None
    adx_max: Optional[float] = None
    atr_min: Optional[float] = None
    atr_max: Optional[float] = None
    stoch_k_min: Optional[float] = None
    stoch_k_max: Optional[float] = None
    roc_min: Optional[float] = None
    roc_max: Optional[float] = None
    above_vwap20: bool = False
    volume_above_sma20: bool = False
    # New Fundamental Filters
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    golden_cross: bool = False
    use_ai_filter: bool = False
    min_ai_precision: float = 0.6

class TechResult(BaseModel):
    symbol: str
    name: str
    last_close: float
    rsi: float
    volume: float
    ema50: float
    ema200: float
    momentum: float
    atr14: float
    adx14: float
    stoch_k: float
    stoch_d: float
    cci20: float
    vwap20: float
    roc12: float
    vol_sma20: float
    # New Fundamental/Price Change Fields
    change_p: float = 0.0
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    beta: Optional[float] = None
    ai_precision: Optional[float] = None
    ai_signal: Optional[str] = None
    logo_url: Optional[str] = None

    @field_validator('*', mode='before')
    def check_nan(cls, v):
        if isinstance(v, float) and (v != v):  # isnan
            return 0.0
        return v

class TechResponse(BaseModel):
    results: List[TechResult]
    scanned_count: int

class IndicatorDashboard(BaseModel):
    buy_signals: int
    sell_signals: int
    win_rate: float

class DashboardResponse(BaseModel):
    rsi: IndicatorDashboard
    macd: IndicatorDashboard
    ema: IndicatorDashboard
    bb: IndicatorDashboard
    scanned_count: int

@router.post("/technical", response_model=TechResponse)
async def scan_technical(
    request: Request,
    f: TechFilter = Body(...)
):
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")

    try:
        symbols_data = load_symbols_for_country(f.country)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No symbols found for country: {f.country}")

    from api.stock_ai import is_ticker_synced
    
    # Pre-calculate sync status for ALL symbols in the country to avoid O(N) single queries
    # Sort candidates to prioritize those already in cache
    cached_candidates = []
    others = []
    
    for row in symbols_data:
        sym = str(row.get("Code", row.get("Symbol", "")))
        ex = str(row.get("Exchange", ""))
        if is_ticker_synced(sym, ex):
            cached_candidates.append(row)
        else:
            others.append(row)
            
    # Combine: Local cached first, then others up to the limit
    sorted_candidates = cached_candidates + others
    candidates = sorted_candidates[:f.limit]
    api = APIClient(api_key)
    
    results = []
    
    for row in candidates:
        # Check if user disconnected to stop processing immediately
        if await request.is_disconnected():
            print("Client disconnected, stopping scan_technical.")
            break

        symbol = str(row.get("Code", row.get("Symbol", "")))
        name = str(row.get("Name", ""))
        exchange = str(row.get("Exchange", ""))
        
        # Skip if symbol is empty or NOT in sync (Cloud-First optimization)
        if not symbol or not is_ticker_synced(symbol, exchange):
            continue

        try:
            # We already verified it's synced, so we can fetch it directly
            df = get_stock_data_eodhd(api, symbol, from_date="2023-01-01", tolerance_days=5, exchange=exchange, force_local=True)
            
            if df.empty: continue

            df = add_technical_indicators(df)
            if df.empty: continue

            last = df.iloc[-1]
            
            # Extract values safely
            close = float(last.get("Close", 0))
            rsi = float(last.get("RSI", 0))
            ema50 = float(last.get("EMA_50", 0))
            ema200 = float(last.get("EMA_200", 0))
            volume = float(last.get("Volume", 0))
            momentum = float(last.get("Momentum", 0))
            atr14 = float(last.get("ATR_14", 0))
            adx14 = float(last.get("ADX_14", 0))
            stoch_k = float(last.get("STOCH_K", 0))
            stoch_d = float(last.get("STOCH_D", 0))
            cci20 = float(last.get("CCI_20", 0))
            vwap20 = float(last.get("VWAP_20", 0))
            roc12 = float(last.get("ROC_12", 0))
            vol_sma20 = float(last.get("VOL_SMA20", 0))

            # Daily change calculation
            prev_close = float(df.iloc[-2].get("Close", close)) if len(df) > 1 else close
            change_p = ((close - prev_close) / prev_close * 100) if prev_close != 0 else 0

            # Get Fundamentals
            funds = get_company_fundamentals(symbol) or {}
            m_cap = funds.get("marketCap")
            pe = funds.get("peRatio")
            eps_val = funds.get("eps")
            div_y = funds.get("dividendYield")
            sec = funds.get("sector")
            ind = funds.get("industry")
            beta_val = funds.get("beta")

            # Apply Fundamental Filters
            if f.market_cap_min and (m_cap or 0) < f.market_cap_min: continue
            if f.market_cap_max and (m_cap or 0) > f.market_cap_max: continue
            if f.sector and f.sector.lower() not in (sec or "").lower(): continue
            if f.industry and f.industry.lower() not in (ind or "").lower(): continue

            # Apply filters
            if f.min_price and close < f.min_price: continue
            if f.rsi_min and rsi < f.rsi_min: continue
            if f.rsi_max and rsi > f.rsi_max: continue
            if f.above_ema50 and close <= ema50: continue
            if f.below_ema50 and close >= ema50: continue
            if f.above_ema200 and close <= ema200: continue
            if f.adx_min and adx14 < f.adx_min: continue
            if f.adx_max and adx14 > f.adx_max: continue
            if f.atr_min and atr14 < f.atr_min: continue
            if f.atr_max and atr14 > f.atr_max: continue
            if f.stoch_k_min and stoch_k < f.stoch_k_min: continue
            if f.stoch_k_max and stoch_k > f.stoch_k_max: continue
            if f.roc_min and roc12 < f.roc_min: continue
            if f.roc_max and roc12 > f.roc_max: continue
            if f.above_vwap20 and close <= vwap20: continue
            if f.volume_above_sma20 and volume <= vol_sma20: continue
            if f.golden_cross and ema50 <= ema200: continue

            # AI Filter
            ai_prec = None
            ai_sig = None
            if f.use_ai_filter:
                from api.stock_ai import run_pipeline
                prediction = run_pipeline(
                    api_key=api_key,
                    ticker=symbol,
                    from_date="2020-01-01",
                    include_fundamentals=False,
                    tolerance_days=5,
                    exchange=exchange,
                    force_local=True
                )
                if prediction["tomorrowPrediction"] != 1: continue
                if prediction["precision"] < f.min_ai_precision: continue
                ai_prec = prediction["precision"]
                ai_sig = "BUY"

            results.append(TechResult(
                symbol=symbol,
                name=name,
                last_close=close,
                rsi=rsi,
                volume=volume,
                ema50=ema50,
                ema200=ema200,
                momentum=momentum,
                atr14=atr14,
                adx14=adx14,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                cci20=cci20,
                vwap20=vwap20,
                roc12=roc12,
                vol_sma20=vol_sma20,
                change_p=change_p,
                market_cap=m_cap,
                pe_ratio=pe,
                eps=eps_val,
                dividend_yield=div_y,
                sector=sec,
                industry=ind,
                beta=beta_val,
                ai_precision=ai_prec,
                ai_signal=ai_sig,
                logo_url=funds.get("logoUrl")
            ))
                    
        except Exception as e:
            continue

    return TechResponse(results=results, scanned_count=len(candidates))

@router.get("/dashboard", response_model=DashboardResponse)
async def get_scan_dashboard(request: Request, country: str = "Egypt", limit: int = 20, days: int = 60):
    """
    Returns aggregate indicator performance (win rates) across multiple symbols in a market.
    """
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")

    try:
        symbols_data = load_symbols_for_country(country)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No symbols found for country: {country}")

    # Prioritize cached symbols for speed
    candidates = []
    for row in symbols_data:
        sym = str(row.get("Code", row.get("Symbol", "")))
        ex = str(row.get("Exchange", ""))
        if check_local_cache(sym, ex):
            candidates.append(row)
        if len(candidates) >= limit:
            break
            
    if not candidates:
        # Fallback to first few if none in cache
        candidates = symbols_data[:5]

    api = APIClient(api_key)
    from api.lib.indicators import calculate_indicator_stats_v2
    from api.stock_ai import run_pipeline
    
    aggr = {
        "rsi": {"wins": 0, "total": 0, "buys": 0, "sells": 0},
        "macd": {"wins": 0, "total": 0, "buys": 0, "sells": 0},
        "ema": {"wins": 0, "total": 0, "buys": 0, "sells": 0},
        "bb": {"wins": 0, "total": 0, "buys": 0, "sells": 0},
    }

    scanned = 0
    for row in candidates:
        if await request.is_disconnected():
            break

        symbol = str(row.get("Code", row.get("Symbol", "")))
        exchange = str(row.get("Exchange", ""))
        
        try:
            # We need historical predictions to calculate WR
            # Using run_pipeline which handles fetching + technicals
            data = run_pipeline(api_key, symbol, exchange=exchange, include_fundamentals=False)
            if not data or "testPredictions" not in data: continue
            
            # Filter by days if specified
            predictions = data["testPredictions"]
            if days > 0:
                predictions = predictions[-days:]
            
            stats = calculate_indicator_stats_v2(predictions)
            
            for key in aggr:
                s = stats.get(key)
                if s:
                    aggr[key]["buys"] += s.get("buySignals", 0)
                    aggr[key]["sells"] += s.get("sellSignals", 0)
                    total_signals = s.get("buySignals", 0) # Only tracking buys for winrate in aggr currently
                    if total_signals > 0:
                        wr = float(s.get("buyWinRate", 0)) / 100.0
                        aggr[key]["wins"] += (wr * total_signals)
                        aggr[key]["total"] += total_signals

            scanned += 1
        except Exception:
            continue

    def build_dashboard(key):
        total = aggr[key]["total"]
        win_rate = (aggr[key]["wins"] / total * 100) if total > 0 else 0
        return IndicatorDashboard(
            buy_signals=aggr[key]["buys"],
            sell_signals=aggr[key]["sells"],
            win_rate=round(win_rate, 1)
        )

    return DashboardResponse(
        rsi=build_dashboard("rsi"),
        macd=build_dashboard("macd"),
        ema=build_dashboard("ema"),
        bb=build_dashboard("bb"),
        scanned_count=scanned
    )
