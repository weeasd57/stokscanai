import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from stock_ai import run_pipeline, check_local_cache
from symbols_local import load_symbols_for_country

router = APIRouter(prefix="/scan", tags=["scan"])

class ScanResult(BaseModel):
    symbol: str
    exchange: Optional[str] = None
    name: str
    last_close: float
    precision: float
    signal: str  # "BUY" or "SELL/HOLD"
    confidence: str # High/Medium/Low based on precision

class SingleScanRequest(BaseModel):
    symbol: str
    exchange: Optional[str] = None
    min_precision: float = 0.6
    rf_preset: Optional[str] = "fast"
    rf_params: Optional[Dict[str, Any]] = None


class ScanAiOptions(BaseModel):
    rf_preset: Optional[str] = "fast"
    rf_params: Optional[Dict[str, Any]] = None

class ScanResponse(BaseModel):
    results: List[ScanResult]
    scanned_count: int

@router.post("/ai", response_model=ScanResponse)
async def scan_ai(
    request: Request,
    country: str = Query(default="Egypt", description="Country to scan"),
    limit: int = Query(default=50, ge=1, le=200, description="Max symbols to scan"),
    min_precision: float = Query(default=0.6, ge=0.0, le=1.0, description="Min precision to include"),
    opts: Optional[ScanAiOptions] = None,
):
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")

    try:
        symbols_data = load_symbols_for_country(country)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No symbols found for country: {country}")
    except Exception as e:
        print(f"scan_ai: failed loading symbols for {country}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load symbols")

    # Sort candidates to prioritize those already in cache
    # This makes the scan "Local-First" and much faster
    cached_candidates = []
    others = []
    
    for row in symbols_data:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("Code", row.get("Symbol", "")))
        ex = str(row.get("Exchange", ""))
        try:
            if check_local_cache(sym, ex):
                cached_candidates.append(row)
            else:
                others.append(row)
        except Exception:
            continue
            
    # Combine: Local cached first, then others
    sorted_candidates = cached_candidates + others
    candidates = sorted_candidates[:limit]
    
    results = []

    rf_preset = (opts.rf_preset if opts else None) or "fast"
    rf_params = (opts.rf_params if opts else None) or None
    
    try:
        for row in candidates:
            # Check if user disconnected to stop processing immediately
            if await request.is_disconnected():
                print("Client disconnected, stopping scan_ai.")
                break

            if not isinstance(row, dict):
                continue

            symbol = str(row.get("Code", row.get("Symbol", "")))
            name = str(row.get("Name", ""))
            exchange = str(row.get("Exchange", ""))
        
            # Skip if symbol is empty or NOT in local cache (Local-First enforcement)
            if not symbol or not check_local_cache(symbol, exchange):
                continue

            try:
                # We skip fundamentals for speed during scan
                prediction = run_pipeline(
                    api_key=api_key,
                    ticker=symbol,
                    from_date="2020-01-01",
                    include_fundamentals=False,
                    tolerance_days=5, # Allow cached data up to 5 days old for scanning speed
                    exchange=exchange,
                    force_local=True,
                    rf_preset=rf_preset,
                    rf_params=rf_params,
                )
            
                # Check for BUY signal
                if prediction["tomorrowPrediction"] == 1:
                    prec = prediction["precision"]

                    if prec >= min_precision:
                        results.append(ScanResult(
                            symbol=symbol,
                            exchange=exchange or None,
                            name=name,
                            last_close=prediction["lastClose"],
                            precision=prec,
                            signal="BUY",
                            confidence="High" if prec > 0.7 else "Medium"
                        ))

            except Exception:
                continue
    except Exception as e:
        print(f"scan_ai: unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Scan failed")

    # Sort by precision descending
    results.sort(key=lambda x: x.precision, reverse=True)

    return ScanResponse(results=results, scanned_count=len(candidates))


@router.post("/ai/single", response_model=Optional[ScanResult])
async def scan_ai_single(req: SingleScanRequest):
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")

    if not req.symbol:
        return None

    try:
        # We enforce local-first for scanning speed
        if not check_local_cache(req.symbol, req.exchange):
            return None

        prediction = run_pipeline(
            api_key=api_key,
            ticker=req.symbol,
            from_date="2020-01-01",
            include_fundamentals=False,
            tolerance_days=5,
            exchange=req.exchange,
            force_local=True,
            rf_preset=req.rf_preset or "fast",
            rf_params=req.rf_params,
        )
        
        if prediction["tomorrowPrediction"] == 1:
            prec = prediction["precision"]
            if prec >= req.min_precision:
                return ScanResult(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    name=req.symbol, 
                    last_close=prediction["lastClose"],
                    precision=prec,
                    signal="BUY",
                    confidence="High" if prec > 0.7 else "Medium"
                )
    except Exception as e:
        print(f"Error scanning {req.symbol}: {e}")
        return None
    
    return None
