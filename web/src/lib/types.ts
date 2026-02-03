export type DateSymbolResult = {
  symbol: string;
  exchange: string;
  name: string;
  rowCount?: number;
};

export type Fundamentals = {
  marketCap?: number | null;
  peRatio?: number | null;
  eps?: number | null;
  sector?: string | null;
  beta?: number | null;
  dividendYield?: number | null;
  high52?: number | null;
  low52?: number | null;
  name?: string | null;
  logoUrl?: string | null;
};

export type TestPredictionRow = {
  date: string;
  close: number;
  pred: 0 | 1;
  target: 0 | 1;
  open?: number;
  high?: number;
  low?: number;
  volume?: number;
  sma50?: number;
  sma200?: number;
  ema50?: number;
  ema200?: number;
  macd?: number;
  macd_signal?: number;
  bb_upper?: number;
  bb_lower?: number;
  rsi?: number;
  momentum?: number;
  outcome?: 'win' | 'loss' | 'pending';
  councilScore?: number;
  consensusRatio?: string;
};

export type ProfitSummary = {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_trade_return_pct: number;
  total_return_pct: number;
};

export type WalkForwardFold = {
  start: string | null;
  end: string | null;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_trade_return_pct: number;
  total_return_pct: number;
};

export type PredictResponse = {
  ticker: string;
  precision: number;
  tomorrowPrediction: 0 | 1;
  signal?: string; // e.g. "STRONG BUY", "BUY", "HOLD", etc.
  lastClose: number;
  lastDate: string;
  fundamentals: Fundamentals;
  testPredictions: TestPredictionRow[];
  executionTime?: number; // milliseconds
  earnPercentage?: number; // cumulative strategy return
  profitSummary?: ProfitSummary | null;
  walkForwardFolds?: WalkForwardFold[];
  topReasons?: string[];
  councilScore?: number;
  consensusRatio?: string;
};
