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
  end if;

  if v_row.stop_price is not null and p_current_price <= v_row.stop_price then
    update public.positions
      set status = 'hit_stop',
          status_at = p_as_of,
          status_price = p_current_price
      where id = p_position_id
      returning * into v_row;

    insert into public.position_events(position_id, user_id, event_type, event_at, payload)
      values (p_position_id, auth.uid(), 'hit_stop', p_as_of, jsonb_build_object('price', p_current_price));

    return v_row;
  end if;

  return v_row;
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
