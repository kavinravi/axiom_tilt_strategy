-- Dashboard datastore schema. Apply once in the Supabase SQL editor.
-- All tables are written by the VPS publisher (service-role key) and read by the
-- Vercel frontend (read-only key). Data volume is tiny; no indexes beyond PKs needed.

create table if not exists snapshot (
  id              int primary key default 1,
  asof            timestamptz not null,
  nav             double precision not null,
  day_pnl         double precision,
  day_pnl_pct     double precision,
  total_return    double precision,
  spy_return      double precision,
  n_positions     int,
  invested_pct    double precision,
  k_probs         jsonb,
  regime_features jsonb,
  risk            jsonb,
  turnover        jsonb
);
alter table snapshot add column if not exists turnover jsonb;

create table if not exists equity_curve (
  date       date primary key,
  nav        double precision not null,
  spy_close  double precision
);

create table if not exists holdings (
  asof          timestamptz not null,
  ticker        text not null,
  shares        double precision not null,
  price         double precision,
  market_value  double precision,
  weight_actual double precision,
  weight_target double precision
);

create table if not exists weekly_portfolio (
  asof_friday   date not null,
  ticker        text not null,
  target_weight double precision not null,
  k_probs       jsonb,
  primary key (asof_friday, ticker)
);

create table if not exists executions (
  asof           date not null,
  ticker         text not null,
  side           text,
  qty            double precision,
  realized_price double precision,
  midpoint       double precision,
  slippage_bps   double precision
);

-- Enable Row Level Security on every table. The SQL editor does NOT auto-enable
-- RLS (unlike the Table Editor UI), so we do it explicitly here. With RLS on and
-- NO policies, the public/anon key can read nothing — the data is not publicly
-- exposed. Writes (VPS publisher) and reads (Vercel server-side) both use the
-- service-role key, which bypasses RLS, so no policies are needed.
alter table snapshot         enable row level security;
alter table equity_curve     enable row level security;
alter table holdings         enable row level security;
alter table weekly_portfolio enable row level security;
alter table executions       enable row level security;
