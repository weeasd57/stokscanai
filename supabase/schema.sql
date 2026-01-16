-- AI Stocks / Supabase Schema
-- Run this in Supabase SQL Editor.

-- Extensions
create extension if not exists pgcrypto;
create extension if not exists citext;

-- Base grants (Supabase projects usually have these, but keep explicit for portability)
grant usage on schema public to anon, authenticated;

-- Types
do $$ begin
  create type public.symbol_source as enum ('home','ai_scanner','tech_scanner');
exception when duplicate_object then null; end $$;

do $$ begin
  create type public.position_status as enum ('open','hit_target','hit_stop','closed_manual');
exception when duplicate_object then null; end $$;

-- Utility: updated_at trigger
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Profiles
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  username citext unique,
  display_name text,
  avatar_url text,
  language text default 'en' check (language in ('en','ar')),

  -- Defaults used when saving a symbol
  default_target_pct numeric(6,2) not null default 5.00 check (default_target_pct > 0 and default_target_pct <= 100),
  default_stop_pct numeric(6,2) not null default 2.00 check (default_stop_pct > 0 and default_stop_pct <= 100),

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists trg_profiles_updated_at on public.profiles;
create trigger trg_profiles_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

alter table public.profiles enable row level security;

create policy "profiles_select_own" on public.profiles
for select using (auth.uid() = id);

create policy "profiles_insert_own" on public.profiles
for insert with check (auth.uid() = id);

create policy "profiles_update_own" on public.profiles
for update using (auth.uid() = id) with check (auth.uid() = id);

grant select, insert, update on public.profiles to authenticated;

-- User settings (replace localStorage persistence)
create table if not exists public.user_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  app_state jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists trg_user_settings_updated_at on public.user_settings;
create trigger trg_user_settings_updated_at
before update on public.user_settings
for each row execute function public.set_updated_at();

alter table public.user_settings enable row level security;

create policy "user_settings_select_own" on public.user_settings
for select using (auth.uid() = user_id);

create policy "user_settings_insert_own" on public.user_settings
for insert with check (auth.uid() = user_id);

create policy "user_settings_update_own" on public.user_settings
for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update on public.user_settings to authenticated;

-- Pricing / Plans (UI + basic subscription storage)
create table if not exists public.pricing_plans (
  id text primary key,
  name text not null,
  price_monthly_cents int not null default 0 check (price_monthly_cents >= 0),
  features jsonb not null default '{}'::jsonb,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists trg_pricing_plans_updated_at on public.pricing_plans;
create trigger trg_pricing_plans_updated_at
before update on public.pricing_plans
for each row execute function public.set_updated_at();

alter table public.pricing_plans enable row level security;

-- Public read of plans
create policy "pricing_plans_select_public" on public.pricing_plans
for select using (is_active = true);

grant select on public.pricing_plans to anon, authenticated;

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  plan_id text not null references public.pricing_plans(id),
  status text not null default 'trialing' check (status in ('trialing','active','past_due','canceled')),
  current_period_start timestamptz,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

drop trigger if exists trg_subscriptions_updated_at on public.subscriptions;
create trigger trg_subscriptions_updated_at
before update on public.subscriptions
for each row execute function public.set_updated_at();

create index if not exists idx_subscriptions_user_id on public.subscriptions(user_id);

alter table public.subscriptions enable row level security;

create policy "subscriptions_select_own" on public.subscriptions
for select using (auth.uid() = user_id);

create policy "subscriptions_insert_own" on public.subscriptions
for insert with check (auth.uid() = user_id);

create policy "subscriptions_update_own" on public.subscriptions
for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update on public.subscriptions to authenticated;

-- Saved Symbols as Positions (targets/stop-loss)
create table if not exists public.positions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,

  symbol text not null,
  name text,
  source public.symbol_source not null,

  metadata jsonb not null default '{}'::jsonb,

  -- Entry
  entry_price numeric(18,6),
  entry_at timestamptz,

  -- Risk config (percent)
  target_pct numeric(6,2) not null check (target_pct > 0 and target_pct <= 100),
  stop_pct numeric(6,2) not null check (stop_pct > 0 and stop_pct <= 100),

  -- Derived values (stored for audit)
  target_price numeric(18,6),
  stop_price numeric(18,6),

  status public.position_status not null default 'open',
  status_at timestamptz,
  status_price numeric(18,6),

  added_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists trg_positions_updated_at on public.positions;
create trigger trg_positions_updated_at
before update on public.positions
for each row execute function public.set_updated_at();

create index if not exists idx_positions_user_id on public.positions(user_id);
create index if not exists idx_positions_user_status on public.positions(user_id, status);
create index if not exists idx_positions_user_symbol on public.positions(user_id, symbol);

-- Only one open position per user per symbol
create unique index if not exists ux_positions_user_symbol_open
on public.positions(user_id, symbol)
where status = 'open';

alter table public.positions enable row level security;

create policy "positions_select_own" on public.positions
for select using (auth.uid() = user_id);

create policy "positions_insert_own" on public.positions
for insert with check (auth.uid() = user_id);

create policy "positions_update_own" on public.positions
for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "positions_delete_own" on public.positions
for delete using (auth.uid() = user_id);

grant select, insert, update, delete on public.positions to authenticated;

-- Position events (audit / history)
create table if not exists public.position_events (
  id uuid primary key default gen_random_uuid(),
  position_id uuid not null references public.positions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,

  event_type text not null,
  event_at timestamptz not null default now(),
  payload jsonb not null default '{}'::jsonb
);

create index if not exists idx_position_events_position_id on public.position_events(position_id);
create index if not exists idx_position_events_user_id on public.position_events(user_id);

alter table public.position_events enable row level security;

create policy "position_events_select_own" on public.position_events
for select using (auth.uid() = user_id);

create policy "position_events_insert_own" on public.position_events
for insert with check (auth.uid() = user_id);

grant select, insert on public.position_events to authenticated;

-- Helper: compute target/stop prices when entry is set
create or replace function public.compute_target_stop_prices()
returns trigger
language plpgsql
as $$
begin
  if new.entry_price is not null then
    new.target_price := new.entry_price * (1 + (new.target_pct / 100));
    new.stop_price := new.entry_price * (1 - (new.stop_pct / 100));
  end if;
  return new;
end;
$$;

drop trigger if exists trg_positions_compute_prices on public.positions;
create trigger trg_positions_compute_prices
before insert or update of entry_price, target_pct, stop_pct on public.positions
for each row execute function public.compute_target_stop_prices();

-- Evaluate a single open position.
-- NOTE: Postgres cannot fetch market prices by itself. The app should call this RPC with a current_price.
create or replace function public.evaluate_position(
  p_position_id uuid,
  p_current_price numeric,
  p_as_of timestamptz default now()
)
returns public.positions
language plpgsql
security definer
set search_path = public
as $$
declare
  v_row public.positions;
begin
  select * into v_row from public.positions where id = p_position_id for update;

  if v_row.id is null then
    raise exception 'Position not found';
  end if;

  if v_row.user_id <> auth.uid() then
    raise exception 'Not allowed';
  end if;

  if v_row.status <> 'open' then
    return v_row;
  end if;

  if v_row.target_price is not null and p_current_price >= v_row.target_price then
    update public.positions
      set status = 'hit_target',
          status_at = p_as_of,
          status_price = p_current_price
      where id = p_position_id
      returning * into v_row;

    insert into public.position_events(position_id, user_id, event_type, event_at, payload)
      values (p_position_id, auth.uid(), 'hit_target', p_as_of, jsonb_build_object('price', p_current_price));

    return v_row;
  elsif v_row.stop_price is not null and p_current_price <= v_row.stop_price then
    update public.positions
      set status = 'hit_stop',
          status_at = p_as_of,
          status_price = p_current_price
      where id = p_position_id
      returning * into v_row;

    insert into public.position_events(position_id, user_id, event_type, event_at, payload)
      values (p_position_id, auth.uid(), 'hit_stop', p_as_of, jsonb_build_object('price', p_current_price));

    return v_row;
  else
    -- Still open, update observed price/date
    update public.positions
      set status_at = p_as_of,
          status_price = p_current_price
      where id = p_position_id
      returning * into v_row;

    return v_row;
  end if;
end;
$$;

-- User stats view (works with RLS for the current user)
create or replace view public.my_position_stats
with (security_invoker = on)
as
select
  user_id,
  count(*) filter (where status = 'open') as open_count,
  count(*) filter (where status = 'hit_target') as win_count,
  count(*) filter (where status = 'hit_stop') as loss_count,
  count(*) as total_count,
  case when count(*) filter (where status in ('hit_target','hit_stop')) = 0 then 0
       else (count(*) filter (where status = 'hit_target')::numeric / nullif(count(*) filter (where status in ('hit_target','hit_stop')), 0))
  end as win_rate
from public.positions
group by user_id;

grant select on public.my_position_stats to authenticated;

-- Leaderboard: use a SECURITY DEFINER RPC to bypass RLS for aggregation
create or replace function public.get_leaderboard(p_limit int default 50)
returns table (
  user_id uuid,
  username citext,
  display_name text,
  win_count bigint,
  loss_count bigint,
  total_closed bigint,
  win_rate numeric
)
language sql
security definer
set search_path = public
as $$
  select
    p.user_id,
    pr.username,
    pr.display_name,
    count(*) filter (where p.status = 'hit_target') as win_count,
    count(*) filter (where p.status = 'hit_stop') as loss_count,
    count(*) filter (where p.status in ('hit_target','hit_stop')) as total_closed,
    case when count(*) filter (where p.status in ('hit_target','hit_stop')) = 0 then 0
         else (count(*) filter (where p.status = 'hit_target')::numeric / nullif(count(*) filter (where p.status in ('hit_target','hit_stop')), 0))
    end as win_rate
  from public.positions p
  join public.profiles pr on pr.id = p.user_id
  group by p.user_id, pr.username, pr.display_name
  order by win_rate desc, win_count desc
  limit greatest(p_limit, 1);
$$;

grant execute on function public.evaluate_position(uuid, numeric, timestamptz) to authenticated;
grant execute on function public.get_leaderboard(int) to anon, authenticated;

-- Auto-provision profile + settings for new users
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id)
  values (new.id)
  on conflict (id) do nothing;

  insert into public.user_settings (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

-- Default plans (optional seed)
insert into public.pricing_plans (id, name, price_monthly_cents, features, is_active)
values
  ('free', 'Free', 0, jsonb_build_object('savedSymbolsLimit', 25, 'scansPerDay', 20), true),
  ('pro', 'Pro', 999, jsonb_build_object('savedSymbolsLimit', 500, 'scansPerDay', 500, 'prioritySupport', true), true)
on conflict (id) do update set
  name = excluded.name,
  price_monthly_cents = excluded.price_monthly_cents,
  features = excluded.features,
  is_active = excluded.is_active;

-- Stock Prices (Historical Data)
create table if not exists public.stock_prices (
    symbol text not null,
    exchange text not null,
    date date not null,
    open numeric(18,6),
    high numeric(18,6),
    low numeric(18,6),
    close numeric(18,6),
    adjusted_close numeric(18,6),
    volume bigint,
    created_at timestamptz not null default now(),
    primary key (symbol, exchange, date)
);

alter table public.stock_prices enable row level security;
create policy "allow_read_all_prices" on public.stock_prices for select using (true);
grant select on public.stock_prices to anon, authenticated;

-- Stock Fundamentals (Company Info)
create table if not exists public.stock_fundamentals (
    symbol text not null,
    exchange text not null,
    data jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    primary key (symbol, exchange)
);

alter table public.stock_fundamentals enable row level security;
create policy "allow_read_all_fundamentals" on public.stock_fundamentals for select using (true);
grant select on public.stock_fundamentals to anon, authenticated;

-- Data Sync Logs (Auditing)
create table if not exists public.data_sync_logs (
    id uuid primary key default gen_random_uuid(),
    exchange text not null,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    status text not null, -- 'running', 'success', 'failed'
    symbols_updated int default 0,
    prices_updated int default 0,
    triggered_by text default 'admin',
    notes text
);

alter table public.data_sync_logs enable row level security;
create policy "admin_only_all_logs" on public.data_sync_logs for all using (true); -- Simplified for admin use
grant all on public.data_sync_logs to anon, authenticated, service_role;

-- Inventory Stats RPC
create or replace function public.get_inventory_stats()
returns table (
    exchange text,
    price_count bigint,
    fund_count bigint,
    last_update timestamptz
)
language plpgsql
security definer
as $$
BEGIN
    RETURN QUERY
    WITH p AS (
        SELECT s.exchange, count(DISTINCT s.symbol) as p_count, max(s.updated_at) as last_ts
        FROM public.stock_prices s
        GROUP BY s.exchange
    ),
    f AS (
        SELECT s.exchange, count(*) as f_count, max(s.updated_at) as last_upd
        FROM public.stock_fundamentals s
        GROUP BY s.exchange
    )
    SELECT 
        COALESCE(p.exchange, f.exchange)::text as exchange,
        COALESCE(p.p_count, 0)::bigint as price_count,
        COALESCE(f.f_count, 0)::bigint as fund_count,
        COALESCE(f.last_upd, p.last_ts) as last_update
    FROM p
    FULL OUTER JOIN f ON p.exchange = f.exchange;
END;
$$;

create or replace function public.get_active_symbols(p_country text default null)
returns table (symbol text, exchange text, name text, country text)
language sql
security definer
as $$
  select distinct 
    p.symbol, 
    p.exchange, 
    (f.data->>'name')::text as name, 
    (f.data->>'country')::text as country
  from public.stock_prices p
  left join public.stock_fundamentals f on p.symbol = f.symbol and p.exchange = f.exchange
  where (p_country is null or (f.data->>'country' = p_country))
    and not (p.symbol ilike 'fund_%');
$$;

create or replace function public.get_active_countries()
returns table (country text)
language sql
security definer
as $$
  select distinct (f.data->>'country')::text as country
  from public.stock_fundamentals f
  where f.data->>'country' is not null;
$$;

grant execute on function public.get_inventory_stats() to anon, authenticated;
grant execute on function public.get_active_symbols(text) to anon, authenticated;
grant execute on function public.get_active_countries() to anon, authenticated;
-- Pre-calculated Technical Indicators Table
-- This table stores daily pre-calculated technical indicators for all stocks
-- Updated by a background job running nightly

create table if not exists public.stock_technical_indicators (
    symbol text not null,
    exchange text not null,
    date date not null,
    
    -- Price data snapshot
    close numeric(18,6),
    volume bigint,
    
    -- Moving Averages
    ema_20 numeric(18,6),
    ema_50 numeric(18,6),
    ema_200 numeric(18,6),
    sma_20 numeric(18,6),
    sma_50 numeric(18,6),
    sma_200 numeric(18,6),
    
    -- Momentum Indicators
    rsi_14 numeric(10,4),
    rsi_9 numeric(10,4),
    macd numeric(18,6),
    macd_signal numeric(18,6),
    macd_histogram numeric(18,6),
    momentum_10 numeric(10,4),
    roc_12 numeric(10,4),
    
    -- Volatility Indicators
    atr_14 numeric(18,6),
    bb_upper numeric(18,6),
    bb_middle numeric(18,6),
    bb_lower numeric(18,6),
    
    -- Trend Indicators
    adx_14 numeric(10,4),
    plus_di numeric(10,4),
    minus_di numeric(10,4),
    
    -- Stochastic
    stoch_k numeric(10,4),
    stoch_d numeric(10,4),
    
    -- Volume Indicators
    vol_sma20 bigint,
    vwap_20 numeric(18,6),
    r_vol numeric(10,4),  -- Relative volume (current vol / avg vol)
    
    -- Other
    cci_20 numeric(10,4),
    
    -- Daily change
    change_pct numeric(10,4),
    
    -- Metadata
    calculated_at timestamptz not null default now(),
    
    primary key (symbol, exchange, date)
);

-- Index for fast lookups by symbol and recent dates
create index if not exists idx_stock_tech_indicators_symbol_date 
    on public.stock_technical_indicators(symbol, exchange, date desc);

create index if not exists idx_stock_tech_indicators_latest 
    on public.stock_technical_indicators(exchange, date desc);

create index if not exists idx_stock_tech_indicators_rsi 
    on public.stock_technical_indicators(rsi_14);

create index if not exists idx_stock_tech_indicators_adx 
    on public.stock_technical_indicators(adx_14);

alter table public.stock_technical_indicators enable row level security;
create policy "allow_read_all_indicators" on public.stock_technical_indicators for select using (true);
grant select on public.stock_technical_indicators to anon, authenticated;

-- Function to get latest technical indicators for a symbol
create or replace function public.get_latest_tech_indicators(
    p_symbol text,
    p_exchange text
)
returns table (
    symbol text,
    exchange text,
    date date,
    close numeric,
    volume bigint,
    rsi_14 numeric,
    ema_50 numeric,
    ema_200 numeric,
    macd numeric,
    macd_signal numeric,
    adx_14 numeric,
    atr_14 numeric,
    stoch_k numeric,
    stoch_d numeric,
    cci_20 numeric,
    vwap_20 numeric,
    roc_12 numeric,
    vol_sma20 bigint,
    momentum_10 numeric,
    change_pct numeric
)
language sql
security definer
stable
as $$
    select 
        t.symbol,
        t.exchange,
        t.date,
        t.close,
        t.volume,
        t.rsi_14,
        t.ema_50,
        t.ema_200,
        t.macd,
        t.macd_signal,
        t.adx_14,
        t.atr_14,
        t.stoch_k,
        t.stoch_d,
        t.cci_20,
        t.vwap_20,
        t.roc_12,
        t.vol_sma20,
        t.momentum_10,
        t.change_pct
    from public.stock_technical_indicators t
    where t.symbol = p_symbol
        and t.exchange = p_exchange
    order by t.date desc
    limit 1;
$$;

-- Function for technical scanner (optimized for performance)
create or replace function public.scan_technical_indicators(
    p_exchange text default 'CAIRO',
    p_limit int default 50,
    p_rsi_min numeric default null,
    p_rsi_max numeric default null,
    p_adx_min numeric default null,
    p_adx_max numeric default null,
    p_atr_min numeric default null,
    p_atr_max numeric default null,
    p_stoch_k_min numeric default null,
    p_stoch_k_max numeric default null,
    p_roc_min numeric default null,
    p_roc_max numeric default null,
    p_above_ema50 boolean default false,
    p_above_ema200 boolean default false,
    p_above_vwap20 boolean default false,
    p_volume_above_sma20 boolean default false,
    p_golden_cross boolean default false
)
returns table (
    symbol text,
    name text,
    last_close numeric,
    rsi numeric,
    volume bigint,
    ema50 numeric,
    ema200 numeric,
    momentum numeric,
    roc12 numeric,
    vol_sma20 bigint,
    change_p numeric,
    atr14 numeric,
    adx14 numeric,
    stoch_k numeric,
    stoch_d numeric,
    cci20 numeric,
    vwap20 numeric
)
language plpgsql
security definer
stable
as $$
BEGIN
    RETURN QUERY
    WITH latest_indicators AS (
        SELECT DISTINCT ON (t.symbol, t.exchange)
            t.symbol,
            t.exchange,
            t.close,
            t.volume,
            t.rsi_14,
            t.ema_50,
            t.ema_200,
            t.momentum_10,
            t.roc_12,
            t.vol_sma20,
            t.change_pct,
            t.atr_14,
            t.adx_14,
            t.stoch_k,
            t.stoch_d,
            t.cci_20,
            t.vwap_20
        FROM public.stock_technical_indicators t
        WHERE t.exchange = p_exchange
            AND t.date >= current_date - interval '3 days'
            -- Apply filters
            AND (p_rsi_min IS NULL OR t.rsi_14 >= p_rsi_min)
            AND (p_rsi_max IS NULL OR t.rsi_14 <= p_rsi_max)
            AND (p_adx_min IS NULL OR t.adx_14 >= p_adx_min)
            AND (p_adx_max IS NULL OR t.adx_14 <= p_adx_max)
            AND (p_atr_min IS NULL OR t.atr_14 >= p_atr_min)
            AND (p_atr_max IS NULL OR t.atr_14 <= p_atr_max)
            AND (p_stoch_k_min IS NULL OR t.stoch_k >= p_stoch_k_min)
            AND (p_stoch_k_max IS NULL OR t.stoch_k <= p_stoch_k_max)
            AND (p_roc_min IS NULL OR t.roc_12 >= p_roc_min)
            AND (p_roc_max IS NULL OR t.roc_12 <= p_roc_max)
            AND (NOT p_above_ema50 OR t.close > t.ema_50)
            AND (NOT p_above_ema200 OR t.close > t.ema_200)
            AND (NOT p_above_vwap20 OR t.close > t.vwap_20)
            AND (NOT p_volume_above_sma20 OR t.volume > t.vol_sma20)
            AND (NOT p_golden_cross OR (t.ema_50 > t.ema_200))
        ORDER BY t.symbol, t.exchange, t.date DESC
    )
    SELECT 
        l.symbol::text,
        COALESCE(f.data->>'name', l.symbol)::text as name,
        l.close,
        l.rsi_14,
        l.volume,
        l.ema_50,
        l.ema_200,
        l.momentum_10,
        l.roc_12,
        l.vol_sma20,
        l.change_pct,
        l.atr_14,
        l.adx_14,
        l.stoch_k,
        l.stoch_d,
        l.cci_20,
        l.vwap_20
    FROM latest_indicators l
    LEFT JOIN public.stock_fundamentals f 
        ON l.symbol = f.symbol AND l.exchange = f.exchange
    LIMIT p_limit;
END;
$$;

grant execute on function public.get_latest_tech_indicators(text, text) to anon, authenticated;
grant execute on function public.scan_technical_indicators(text, int, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, boolean, boolean, boolean, boolean, boolean) to anon, authenticated;

-- Add comment for documentation
comment on table public.stock_technical_indicators is 'Pre-calculated technical indicators updated daily by background job for fast scanner queries';
comment on function public.scan_technical_indicators is 'High-performance technical scanner using pre-calculated indicators';

-- Model Test History & Analytics (Simplified - Single Table)
create table if not exists public.model_tests (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    
    -- Model & Symbol Info
    model_name text not null,
    symbol text not null,
    exchange text not null,
    symbol_count int not null,  -- عدد الرموز المختبرة
    
    -- Results & Signals
    buy_signals int not null,
    sell_signals int not null,
    execution_time_ms int not null,  -- الوقت الذي استغرقه النموذج (الاتجاه الزمني)
    win_rate numeric(10,4) not null,
    
    -- Test Metadata
    tested_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_model_tests_user_id on public.model_tests(user_id);
create index if not exists idx_model_tests_symbol on public.model_tests(symbol, exchange);
create index if not exists idx_model_tests_tested_at on public.model_tests(tested_at desc);

alter table public.model_tests enable row level security;

create policy "model_tests_select_own" on public.model_tests
for select using (auth.uid() = user_id or user_id is null);

create policy "model_tests_insert_own" on public.model_tests
for insert with check (auth.uid() = user_id);

grant select, insert on public.model_tests to authenticated;

-- Model Test Analytics View
create or replace view public.model_test_stats
with (security_invoker = on)
as
select
    user_id,
    symbol,
    exchange,
    model_name,
    count(*) as total_tests,
    count(distinct tested_at::date) as test_days,
    avg(execution_time_ms) as avg_execution_time_ms,
    max(execution_time_ms) as max_execution_time_ms,
    min(execution_time_ms) as min_execution_time_ms,
    avg(win_rate) as avg_win_rate,
    max(win_rate) as best_win_rate,
    min(win_rate) as worst_win_rate,
    sum(buy_signals) as total_buy_signals,
    sum(sell_signals) as total_sell_signals,
    max(tested_at) as last_tested
from public.model_tests
group by user_id, symbol, exchange, model_name;

grant select on public.model_test_stats to authenticated;

-- Function to save test results (Simplified - Single Table with Execution Time)
create or replace function public.save_model_test(
    p_model_name text,
    p_symbol text,
    p_exchange text,
    p_predictions jsonb,
    p_execution_time_ms int default 0,
    p_symbol_count int default 1
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_test_id uuid;
    v_total int;
    v_buy_count int;
    v_sell_count int;
    v_correct int;
    v_win_rate numeric;
    v_pred jsonb;
begin
    -- Input validation
    if p_predictions is null or jsonb_array_length(p_predictions) = 0 then
        raise exception 'No predictions provided';
    end if;
    if p_model_name is null or trim(p_model_name) = '' then
        raise exception 'Model name is required';
    end if;
    if p_symbol is null or trim(p_symbol) = '' then
        raise exception 'Symbol is required';
    end if;
    if p_exchange is null or trim(p_exchange) = '' then
        raise exception 'Exchange is required';
    end if;
    
    -- Calculate statistics
    v_total := jsonb_array_length(p_predictions);
    v_buy_count := 0;
    v_sell_count := 0;
    v_correct := 0;
    
    for v_pred in select jsonb_array_elements(p_predictions)
    loop
        if (v_pred->>'pred')::int = 1 then
            v_buy_count := v_buy_count + 1;
        else
            v_sell_count := v_sell_count + 1;
        end if;
        
        if (v_pred->>'pred')::int = (v_pred->>'target')::int then
            v_correct := v_correct + 1;
        end if;
    end loop;
    
    -- Calculate win rate
    v_win_rate := case when v_total > 0 then (v_correct::numeric / v_total) * 100 else 0 end;
    
    -- Insert simplified test record with execution time as direction indicator
    insert into public.model_tests (
        user_id,
        model_name,
        symbol,
        exchange,
        symbol_count,
        buy_signals,
        sell_signals,
        execution_time_ms,
        win_rate
    ) values (
        auth.uid(),
        trim(p_model_name),
        trim(p_symbol),
        trim(p_exchange),
        p_symbol_count,
        v_buy_count,
        v_sell_count,
        p_execution_time_ms,
        v_win_rate
    ) returning id into v_test_id;
    
    return v_test_id;
end;
$$;

grant execute on function public.save_model_test(text, text, text, jsonb, int, int) to authenticated;

