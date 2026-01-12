# AI Stocks - Roadmap / TODO

This file tracks planned features, architecture improvements, optimizations, and quality work.

## Product Features
- [x] Supabase backend migration
  - [x] Add Supabase project + env config (URL/Anon Key)
  - [x] Run SQL schema in Supabase (see supabase/schema.sql)
  - [x] Replace all client localStorage persistence with Supabase tables (user_settings, profiles, positions)
  - [x] One-time migration: upload existing local watchlist + persisted UI settings to Supabase after first login

- [x] Auth (Signup / Login)
  - [x] Add /signup and /login pages
  - [x] Add /profile page (protected)
  - [x] Add logout button in header

- [x] Pricing
  - [x] Add /pricing page
  - [x] Store plan definitions in Supabase (pricing_plans)
  - [x] Add subscriptions table usage for plan status (UI only unless payments are added)

- [x] Profile dashboard
  - [x] Dashboard stats for saved symbols (open/win/loss/win rate)
  - [x] Leaderboard section in profile (Supabase RPC get_leaderboard)
  - [x] User defaults editor (default target % / default stop %)

- [x] Targets / Stop-loss tracking for saved symbols
  - [x] When saving a symbol, require selecting target % and stop % (prefill from profile defaults)
  - [x] Store each saved symbol as an open position (positions table)
  - [x] Add "Evaluate positions" button in profile
  - [x] Mark a position as win (hit_target) OR loss (hit_stop), never both
  - [ ] Show per-position progress (current price vs target/stop)

- [ ] Alerts (indicator-based and AI-based)
  - [ ] Create alert rules (RSI/MACD/EMA crosses, AI BUY with min precision)
  - [ ] Add alert delivery channels (in-app notifications, optional email/telegram)
  - [ ] Add alert history and enable/disable controls

- [ ] Saved Screeners / Presets
  - [ ] Save/rename/delete presets for AI Scanner and Technical Scanner
  - [ ] Store presets in localStorage (lightweight only) and optionally in backend per user

- [ ] Multi-timeframe Support
  - [ ] Add timeframe parameter to price history + indicator calculation
  - [ ] UI controls for 1D / 4H / 1H and persistence via AppStateContext

- [ ] Advanced Chart Tools
  - [ ] More overlays (VWAP, ATR bands, support/resistance)
  - [ ] Drawing tools (future)

- [ ] Backtesting (lightweight)
  - [ ] Simple strategy backtest based on signals
  - [ ] Equity curve + drawdown + trade stats

## Supabase Data Cache Migration (SaaS)

> **Goal**: Move `data_cache/` (fundamentals + prices) to Supabase so all users share centralized, daily-updated market data.

### Database Schema
- [ ] Create `stock_fundamentals` table
  ```sql
  CREATE TABLE stock_fundamentals (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,       -- EGX, PA, US, france
    name TEXT,
    country TEXT,
    data JSONB NOT NULL,          -- Full fund data (P/E, EPS, BVPS, news, etc.)
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange)
  );
  ```
- [ ] Create `stock_prices` table
  ```sql
  CREATE TABLE stock_prices (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(18,6),
    high NUMERIC(18,6),
    low NUMERIC(18,6),
    close NUMERIC(18,6),
    adjusted_close NUMERIC(18,6),
    volume BIGINT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange, date)
  );
  CREATE INDEX idx_stock_prices_symbol ON stock_prices(symbol, exchange);
  CREATE INDEX idx_stock_prices_date ON stock_prices(date DESC);
  ```
- [ ] Add RLS policies (read: all authenticated users, write: service role only)

### Data Migration Script
### Data Migration Script
- [x] Create Python script to upload existing `data_cache/` files to Supabase
  - [x] Read all `fund/*.json` → insert into `stock_fundamentals`
    - [x] **IMPORTANT:** Skip files where `status` is "error" or data is empty
  - [x] Read all `prices/*.csv` → insert into `stock_prices`
  - [x] Support all exchanges: EGX, PA, US, france

### Admin Update Functionality
### Admin Update Functionality
- [x] Add admin endpoint: `POST /admin/sync-data`
  - [x] Accepts optional `exchange` filter (or sync all)
  - [x] Fetches fresh data from EODHD API
  - [x] Upserts into Supabase tables (same structure as funds)
  - [x] Logs sync status and duration
- [x] Add admin UI button in web dashboard to trigger manual sync
- [x] Store sync history in new table `data_sync_logs`:
  ```sql
  CREATE TABLE data_sync_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange TEXT,
    symbols_updated INT,
    prices_updated INT,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running', -- running, success, failed
    error_message TEXT,
    triggered_by TEXT -- 'admin', 'cron', 'manual'
  );
  ```

### Daily Automated Sync (Cron Job)
- [ ] Create Supabase Edge Function or external cron for daily updates
  - [ ] Run after market close (e.g., 18:00 EET for EGX)
  - [ ] Fetch latest prices + fundamentals from EODHD
  - [ ] Upsert into Supabase tables
  - [ ] Send notification on failure (optional)

### API Changes
- [ ] Modify `stock_ai.py` to read from Supabase instead of local files
  - [ ] `get_fundamentals(symbol, exchange)` → query `stock_fundamentals`
  - [ ] `get_price_history(symbol, exchange, from_date)` → query `stock_prices`
  - [ ] Add in-memory caching with TTL for frequently accessed data
- [ ] Keep local file fallback for development mode (optional)

### Frontend Changes
- [ ] Remove any direct file reads from frontend (already done via API)
- [ ] Add "Last updated" timestamp display from `stock_fundamentals.updated_at`

---

## Backend Architecture
- [ ] Indicator registry
  - [ ] Move each indicator to a dedicated function/module
  - [ ] Add a registry that declares: name, params, input columns, output fields
  - [ ] Allow scanner endpoints to request indicators by name

- [ ] Scans as background jobs
  - [ ] POST /scan/*/start -> returns jobId
  - [ ] GET /scan/*/status/{jobId} -> progress + partial results
  - [ ] POST /scan/*/cancel/{jobId}

- [ ] Data source strategy
  - [ ] Separate configuration for price history source vs fundamentals source
  - [ ] Add fallback logic per exchange (EGX vs US)

## Frontend Architecture
- [ ] Supabase integration layer
  - [ ] Add Supabase client/server helpers
  - [ ] Cookie-based session handling (avoid localStorage session)
  - [ ] Protected routes (middleware or server checks)

- [ ] Replace WatchlistContext storage
  - [ ] Store watchlist/positions in Supabase (positions table)
  - [ ] Add migration from localStorage watchlist key (ai_stocks_watchlist)

- [ ] Config-driven filters UI
  - [ ] Define filter schema (range/toggle/select)
  - [ ] Render filter panel from config for AI/Technical scanners

- [ ] Global state management
  - [ ] Extend AppStateContext for additional pages/features
  - [ ] Persist lightweight UI config to Supabase user_settings (no localStorage)

- [ ] Tables performance
  - [ ] Pagination and/or virtualization for large result sets

## Performance & Caching
- [ ] Backend caching
  - [ ] Cache results by (symbol, from_date, timeframe, indicators)
  - [ ] Cache scan results by filter hash (short TTL)

- [ ] Fundamentals caching
  - [ ] Add TTL + stale-cache fallback to reduce rate limits
  - [ ] Prefer cached fundamentals during batch updates

## Security & Reliability
- [ ] Input validation
  - [ ] Validate symbol formats and filter ranges

- [ ] Rate limiting
  - [ ] Protect expensive endpoints (/predict, /scan/*)

- [ ] Observability
  - [ ] Log durations for scans/predict calls
  - [ ] Add structured errors to API responses

## Testing
- [ ] Unit tests for indicator calculations
- [ ] Integration tests for scanner endpoints
- [ ] UI smoke tests for key pages (Home, AI Scanner, Technical Scanner)
