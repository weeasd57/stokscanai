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
};
