import type { PredictResponse } from "@/lib/types";

export interface KPIStats {
  totalTests: number;
  totalDays: number;
  buySignals: number;
  sellSignals: number;
  buyCorrect: number;
  sellCorrect: number;
  buyAccuracy: number;
  sellAccuracy: number;
  totalCorrect: number;
  totalIncorrect: number;
  winRate: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
}

export interface ClassificationStats {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  precisionBuy: number;
  recallBuy: number;
  f1Buy: number;
  precisionSell: number;
  recallSell: number;
  f1Sell: number;
}

export interface ModelSummary {
  modelName: string;
  result: PredictResponse;
  kpis: KPIStats;
}

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

export interface ModelMetadata {
  name: string;
  kind?: string;
  timestamp?: string;
  num_features?: number;
  numFeatures?: number;
  training_samples?: number;
  trainingSamples?: number;
  n_estimators?: number;
  nEstimators?: number;
  learning_rate?: number;
  target_pct?: number;
  stop_loss_pct?: number;
  look_forward_days?: number;
}

export interface ModelStats {
  name: string;
  trainingSamples: number | null;
  numFeatures: number | null;
  nEstimators: number | null;
}
