"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import { Brain, CalendarRange, Loader2, LineChart, Sparkles, Database, Building2, Search, TrendingUp, TrendingDown, ChevronUp, ChevronDown, CircleDot, BarChart3, Gauge, LayoutDashboard } from "lucide-react";
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from "recharts";
import { predictStock, getLocalModels, getSymbolsByDate, getCountries, getSymbolsForExchange } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import type { DateSymbolResult } from "@/lib/api";
import { useAppState } from "@/contexts/AppStateContext";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import TestModelCandleChart from "./TestModelCandleChart";

const defaultStart = "2023-01-01";
const defaultEnd = new Date().toISOString().slice(0, 10);

function parseModelExchange(modelName: string | null): string | null {
  if (!modelName) return null;

  // 1. Try format model_EXCHANGE.pkl
  let match = modelName.match(/model_(.+?)\.pkl/i);
  if (match) return match[1].toUpperCase();

  // 2. Look for exchange codes anywhere in the name (EGX, US, SA, etc.)
  // Use word boundaries or underscores to avoid partial matches
  const upper = modelName.toUpperCase();
  const exchanges = ["EGX", "USA", "US", "KSA", "SA", "KQ", "BA", "TO", "LSE", "PA", "F"];

  for (const ex of exchanges) {
    // Look for the code surrounded by underscores, spaces, or at poles
    const regex = new RegExp(`(?:^|[\\s_.,])${ex}(?:$|[\\s_.,])`, 'i');
    if (regex.test(upper)) return ex === "USA" ? "US" : ex;
  }

  return null;
}

function signalLabel(value: number) {
  return value === 1 ? "BUY" : "SELL";
}

function signalClass(value: number) {
  return value === 1
    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
    : "bg-rose-500/10 text-rose-400 border-rose-500/20";
}

function calculateKPIs(predictions: any[]) {
  if (!predictions || predictions.length === 0) {
    return {
      totalTests: 0, totalDays: 0, buySignals: 0, sellSignals: 0,
      buyCorrect: 0, sellCorrect: 0, buyAccuracy: 0, sellAccuracy: 0,
      totalCorrect: 0, totalIncorrect: 0, winRate: 0,
      consecutiveWins: 0, consecutiveLosses: 0, maxConsecutiveWins: 0, maxConsecutiveLosses: 0,
    };
  }

  // Filter to only include test rows if the backend provided them
  const rows = predictions.filter(p => p.is_test !== false);
  const dataToUse = rows.length > 0 ? rows : predictions;

  const uniqueDates = new Set(dataToUse.map((p) => p.date)).size;
  const buySignals = dataToUse.filter((p) => p.pred === 1).length;
  const sellSignals = dataToUse.filter((p) => p.pred === 0).length;

  const buyCorrect = dataToUse.filter((p) => p.pred === 1 && p.pred === p.target).length;
  const sellCorrect = dataToUse.filter((p) => p.pred === 0 && p.pred === p.target).length;

  const totalCorrect = dataToUse.filter((p) => p.pred === p.target).length;
  const totalIncorrect = dataToUse.length - totalCorrect;

  let consecutiveWins = 0, consecutiveLosses = 0, maxConsecutiveWins = 0, maxConsecutiveLosses = 0;
  for (const pred of dataToUse) {
    if (pred.pred === pred.target) {
      consecutiveWins++; consecutiveLosses = 0;
      maxConsecutiveWins = Math.max(maxConsecutiveWins, consecutiveWins);
    } else {
      consecutiveLosses++; consecutiveWins = 0;
      maxConsecutiveLosses = Math.max(maxConsecutiveLosses, consecutiveLosses);
    }
  }

  return {
    totalTests: dataToUse.length,
    totalDays: uniqueDates,
    buySignals, sellSignals, buyCorrect, sellCorrect,
    buyAccuracy: buySignals > 0 ? (buyCorrect / buySignals) * 100 : 0,
    sellAccuracy: sellSignals > 0 ? (sellCorrect / sellSignals) * 100 : 0,
    totalCorrect, totalIncorrect,
    winRate: (totalCorrect / dataToUse.length) * 100,
    consecutiveWins, consecutiveLosses, maxConsecutiveWins, maxConsecutiveLosses,
  };
}

function calculateClassification(predictions: any[]) {
  const rows = predictions.filter(p => p.is_test !== false);
  const dataToUse = rows.length > 0 ? rows : (predictions || []);

  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const p of dataToUse) {
    const pred = p?.pred, target = p?.target;
    if (pred === 1 && target === 1) tp++;
    else if (pred === 1 && target === 0) fp++;
    else if (pred === 0 && target === 0) tn++;
    else if (pred === 0 && target === 1) fn++;
  }

  const precisionBuy = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recallBuy = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Buy = precisionBuy + recallBuy > 0 ? (2 * precisionBuy * recallBuy) / (precisionBuy + recallBuy) : 0;
  const precisionSell = tn + fn > 0 ? tn / (tn + fn) : 0;
  const recallSell = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1Sell = precisionSell + recallSell > 0 ? (2 * precisionSell * recallSell) / (precisionSell + recallSell) : 0;

  return { tp, fp, tn, fn, precisionBuy, recallBuy, f1Buy, precisionSell, recallSell, f1Sell };
}

function renderResultContent(result: any) {
  if (!result?.testPredictions?.length) {
    return <div className="text-xs text-zinc-500">No predictions available.</div>;
  }
  const kpis = calculateKPIs(result.testPredictions);
  return (
    <div className="space-y-4">
      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="bg-zinc-900/50 rounded-lg p-2 border border-white/5">
          <div className="text-[9px] text-zinc-500 uppercase tracking-wider">Tests</div>
          <div className="text-base font-bold text-white mt-1">{kpis.totalTests}</div>
        </div>
        <div className="bg-emerald-500/10 rounded-lg p-2 border border-emerald-500/20">
          <div className="text-[9px] text-emerald-400 uppercase">Buy</div>
          <div className="text-base font-bold text-emerald-400 mt-1">{kpis.buySignals}</div>
        </div>
        <div className="bg-rose-500/10 rounded-lg p-2 border border-rose-500/20">
          <div className="text-[9px] text-rose-400 uppercase">Sell</div>
          <div className="text-base font-bold text-rose-400 mt-1">{kpis.sellSignals}</div>
        </div>
        <div className="bg-indigo-500/10 rounded-lg p-2 border border-indigo-500/20">
          <div className="text-[9px] text-indigo-400 uppercase">Win</div>
          <div className="text-base font-bold text-indigo-400 mt-1">{kpis.winRate.toFixed(1)}%</div>
        </div>
        {result.earnPercentage != null && (
          <div className={`rounded-lg p-2 border ${result.earnPercentage >= 0 ? "bg-emerald-500/10 border-emerald-500/20" : "bg-rose-500/10 border-rose-500/20"}`}>
            <div className={`text-[9px] uppercase ${result.earnPercentage >= 0 ? "text-emerald-400" : "text-rose-400"}`}>Earn Rate</div>
            <div className={`text-base font-bold mt-1 ${result.earnPercentage >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {result.earnPercentage >= 0 ? "+" : ""}{result.earnPercentage.toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function groupPredictionsByWeek(predictions: any[]) {
  const groups: { week: string; predictions: any[] }[] = [];
  const weekMap = new Map<string, any[]>();

  predictions.forEach((pred) => {
    const date = new Date(pred.date);
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay());
    const weekKey = weekStart.toISOString().split('T')[0];

    if (!weekMap.has(weekKey)) {
      weekMap.set(weekKey, []);
    }
    weekMap.get(weekKey)!.push(pred);
  });

  weekMap.forEach((preds, week) => {
    groups.push({ week, predictions: preds });
  });

  return groups.sort((a, b) => a.week.localeCompare(b.week));
}

export default function TestModelTab() {
  const [models, setModels] = useState<any[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [startDate, setStartDate] = useState(defaultStart);
  const [endDate, setEndDate] = useState(defaultEnd);
  const { countries: availableExchanges, refreshCountries: refreshAvailableExchanges, inventory } = useAppState();
  const [selectedExchange, setSelectedExchange] = useState<string>("");
  const [symbols, setSymbols] = useState<DateSymbolResult[]>([]);
  const [symbolsLoading, setSymbolsLoading] = useState(false);
  const [symbolsError, setSymbolsError] = useState<string | null>(null);
  const [searchSymbolTerm, setSearchSymbolTerm] = useState<string>("");

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [useMultipleModels, setUseMultipleModels] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<DateSymbolResult | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<PredictResponse | null>(null);
  const [testResults, setTestResults] = useState<Map<string, PredictResponse>>(new Map());
  const [saveLoading, setSaveLoading] = useState(false);

  // Strategy override settings
  const [targetPct, setTargetPct] = useState<number>(0.15);
  const [stopLossPct, setStopLossPct] = useState<number>(0.05);
  const [lookForwardDays, setLookForwardDays] = useState<number>(20);
  const [buyThreshold, setBuyThreshold] = useState<number>(0.40);
  const [isAutoDetected, setIsAutoDetected] = useState(false);

  const [showRSI, setShowRSI] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showSMA200, setShowSMA200] = useState(false);
  const [showEMA50, setShowEMA50] = useState(false);
  const [showEMA200, setShowEMA200] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [showVolume, setShowVolume] = useState(true);
  const [showMACD, setShowMACD] = useState(true);
  const [showBUYSignals, setShowBUYSignals] = useState(true);
  const [showSELLSignals, setShowSELLSignals] = useState(true);
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);

  const selectedModelStats = useMemo(() => {
    if (!models.length) return [] as any[];
    const selected = useMultipleModels ? Array.from(selectedModels) : selectedModel ? [selectedModel] : [];
    if (!selected.length) return [] as any[];

    const byName = new Map(
      models.map((model) => {
        const name = typeof model === "string" ? model : model.name;
        return [name, model] as const;
      })
    );

    return selected
      .map((name) => {
        const model = byName.get(name) as any;
        const numFeatures = typeof model === "object" ? (model.num_features ?? model.numFeatures) : undefined;
        const trainingSamples = typeof model === "object" ? (model.trainingSamples ?? model.training_samples) : undefined;
        const nEstimators = typeof model === "object" ? (model.n_estimators ?? model.nEstimators) : undefined;
        return {
          name: name.replace(/^model_|\.pkl$/gi, ""),
          trainingSamples: typeof trainingSamples === "number" ? trainingSamples : null,
          numFeatures: typeof numFeatures === "number" ? numFeatures : null,
          nEstimators: typeof nEstimators === "number" ? nEstimators : null,
        };
      })
      .filter((item) => item.trainingSamples !== null || item.numFeatures !== null || item.nEstimators !== null);
  }, [models, selectedModel, selectedModels, useMultipleModels]);

  const singleSummary = useMemo(() => {
    if (!testResult || !testResult.testPredictions?.length) return null;
    const kpis = calculateKPIs(testResult.testPredictions);
    const lastPred = testResult.testPredictions[testResult.testPredictions.length - 1];
    const signal = lastPred?.pred ?? 1;
    return { kpis, signal };
  }, [testResult]);

  const multiSummaries = useMemo(() => {
    if (!testResults.size) return [] as {
      modelName: string;
      result: PredictResponse;
      kpis: ReturnType<typeof calculateKPIs>;
    }[];

    const items = Array.from(testResults.entries()).map(([modelName, result]) => {
      const kpis = calculateKPIs(result.testPredictions || []);
      return { modelName, result, kpis };
    });

    items.sort((a, b) => b.kpis.winRate - a.kpis.winRate);
    return items;
  }, [testResults]);

  const predictionsWithOutcome = useMemo(() => {
    if (!testResult?.testPredictions) return [];

    const rows = [...testResult.testPredictions].sort((a, b) => (new Date(a.date).getTime() - new Date(b.date).getTime()));

    return rows.map((row, idx) => {
      if (row.pred !== 1) return { ...row, outcome: undefined };

      const entryPrice = row.close;
      const targetPrice = entryPrice * (1 + targetPct);
      const stopPrice = entryPrice * (1 - stopLossPct);

      // Look forward to find outcome
      for (let i = idx + 1; i < Math.min(idx + lookForwardDays + 1, rows.length); i++) {
        const future = rows[i];
        if ((future.low ?? future.close) <= stopPrice) return { ...row, outcome: 'loss' as const };
        if ((future.high ?? future.close) >= targetPrice) return { ...row, outcome: 'win' as const };
      }

      return { ...row, outcome: 'pending' as const };
    });
  }, [testResult, targetPct, stopLossPct, lookForwardDays]);

  const sortedMultiSummaries = useMemo(() => {
    if (!sortConfig) return multiSummaries;

    return [...multiSummaries].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      const { key } = sortConfig;

      if (key === 'name') {
        aValue = a.modelName;
        bValue = b.modelName;
      } else if (key === 'winRate') {
        aValue = a.kpis.winRate;
        bValue = b.kpis.winRate;
      } else if (key === 'buyAcc') {
        aValue = a.kpis.buyAccuracy;
        bValue = b.kpis.buyAccuracy;
      } else if (key === 'sellAcc') {
        aValue = a.kpis.sellAccuracy;
        bValue = b.kpis.sellAccuracy;
      } else if (key === 'execTime') {
        aValue = (a.result as any).executionTime ?? 0;
        bValue = (b.result as any).executionTime ?? 0;
      } else if (key === 'earnRate') {
        aValue = a.result.earnPercentage ?? -999;
        bValue = b.result.earnPercentage ?? -999;
      } else {
        // Classification metrics
        const clsA = calculateClassification(a.result.testPredictions || []);
        const clsB = calculateClassification(b.result.testPredictions || []);
        if (key === 'precision') { aValue = clsA.precisionBuy; bValue = clsB.precisionBuy; }
        else if (key === 'recall') { aValue = clsA.recallBuy; bValue = clsB.recallBuy; }
        else if (key === 'f1') { aValue = clsA.f1Buy; bValue = clsB.f1Buy; }
      }

      if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [multiSummaries, sortConfig]);

  const toggleSort = (key: string) => {
    setSortConfig(prev => {
      if (prev?.key === key) {
        return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key, direction: 'desc' };
    });
  };

  const SortIcon = ({ column }: { column: string }) => {
    if (sortConfig?.key !== column) return <div className="w-3 h-3 ml-1 opacity-20" />;
    return sortConfig.direction === 'asc' ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />;
  };

  const multiClassificationChart = useMemo(() => {
    if (!testResults.size) return [] as any[];
    return Array.from(testResults.entries())
      .map(([modelName, result]) => {
        const cls = calculateClassification(result.testPredictions || []);
        return {
          name: modelName.replace(/^model_|\.pkl$/gi, ""),
          precision: cls.precisionBuy * 100,
          recall: cls.recallBuy * 100,
          f1: cls.f1Buy * 100,
        };
      })
      .sort((a, b) => (b.f1 ?? 0) - (a.f1 ?? 0));
  }, [testResults]);

  const modelExchange = useMemo(() => parseModelExchange(selectedModel), [selectedModel]);

  useEffect(() => {
    let mounted = true;
    async function loadModels() {
      setModelsLoading(true);
      setModelsError(null);
      try {
        const data = await getLocalModels();
        if (mounted) {
          setModels(data);
          if (!selectedModel && Array.isArray(data) && data.length > 0) {
            const first = data[0] as any;
            const firstName = typeof first === "string" ? first : first?.name;
            if (firstName) {
              setSelectedModel(firstName);
            }
          }
        }
      } catch (err) {
        if (mounted) {
          setModelsError(err instanceof Error ? err.message : "Failed to load models");
        }
      } finally {
        if (mounted) setModelsLoading(false);
      }
    }

    loadModels();
    return () => {
      mounted = false;
    };
  }, [selectedModel]);

  useEffect(() => {
    if (selectedModel && !useMultipleModels) {
      const model = models.find(m => (typeof m === "string" ? m : m.name) === selectedModel);
      if (model && typeof model === "object") {
        if (model.target_pct != null) {
          setTargetPct(model.target_pct);
          setStopLossPct(model.stop_loss_pct ?? 0.05);
          setLookForwardDays(model.look_forward_days ?? 20);
          setIsAutoDetected(true);
        } else {
          setIsAutoDetected(false);
        }
      }
    }
  }, [selectedModel, models, useMultipleModels]);

  useEffect(() => {
    if (modelExchange && !selectedExchange) {
      // Map exchange code back to display name if possible
      const lower = modelExchange.toLowerCase();
      const invItem = inventory.find(item =>
        item.exchange?.toLowerCase() === lower ||
        item.country?.toLowerCase() === lower
      );
      if (invItem?.country) {
        setSelectedExchange(invItem.country);
      } else if (availableExchanges.includes(modelExchange)) {
        setSelectedExchange(modelExchange);
      } else {
        // Fallback to searching case-insensitive in availableExchanges
        const matched = availableExchanges.find(e => e.toLowerCase() === lower);
        if (matched) setSelectedExchange(matched);
        else setSelectedExchange(modelExchange);
      }
    }
    if (!selectedExchange && availableExchanges.length > 0) {
      // Prioritize Egypt as default if available
      const egypt = availableExchanges.find(e => e.toLowerCase() === "egypt");
      setSelectedExchange(egypt || availableExchanges[0]);
    }
  }, [modelExchange, selectedExchange, availableExchanges, inventory]);

  useEffect(() => {
    refreshAvailableExchanges();
  }, [refreshAvailableExchanges]);

  const exchangeCode = useMemo(() => {
    if (!selectedExchange) return "";
    const lower = selectedExchange.toLowerCase();

    // Direct mapping for common ones if inventory is not yet loaded
    if (lower === "egypt" || lower === "egx") return "EGX";
    if (lower === "usa" || lower === "us") return "US";
    if (lower === "saudi arabia" || lower === "sa" || lower === "ksa") return "SA";

    const invItem = inventory.find(item =>
      item.country?.toLowerCase() === lower ||
      item.exchange?.toLowerCase() === lower
    );
    return invItem?.exchange || (selectedExchange.length <= 4 ? selectedExchange.toUpperCase() : selectedExchange);
  }, [selectedExchange, inventory]);

  useEffect(() => {
    let active = true;
    async function loadSymbols() {
      if (!exchangeCode) return;
      setSymbolsLoading(true);
      setSymbolsError(null);
      try {
        let data = await getSymbolsForExchange(exchangeCode);

        if (searchSymbolTerm) {
          const term = searchSymbolTerm.toLowerCase();
          data = data.filter(
            (s) =>
              s.symbol.toLowerCase().includes(term) ||
              s.name.toLowerCase().includes(term)
          );
        }

        if (active) {
          setSymbols(data);
          if (data.length > 0) {
            setSelectedSymbol(data[0]);
          } else {
            setSelectedSymbol(null);
          }
        }
      } catch (err) {
        if (active) {
          setSymbolsError(err instanceof Error ? err.message : "Failed to load symbols");
          setSymbols([]);
          setSelectedSymbol(null);
        }
      } finally {
        if (active) setSymbolsLoading(false);
      }
    }

    loadSymbols();
    return () => {
      active = false;
    };
  }, [exchangeCode, searchSymbolTerm]);

  useEffect(() => {
    if (symbols.length > 0 && !selectedSymbol) {
      setSelectedSymbol(symbols[0]);
    }
  }, [symbols, selectedSymbol]);

  async function runTest() {
    if (!selectedSymbol) {
      setTestError("Select a symbol first");
      return;
    }

    const modelsToTest = useMultipleModels ? Array.from(selectedModels) : (selectedModel ? [selectedModel] : []);
    if (modelsToTest.length === 0) {
      setTestError("Select at least one model");
      return;
    }

    setTestLoading(true);
    setTestError(null);
    setTestResult(null);
    setTestResults(new Map());

    try {
      const resultsMap = new Map<string, PredictResponse>();

      for (const model of modelsToTest) {
        const startTime = performance.now();
        try {
          const response = await predictStock({
            ticker: selectedSymbol.symbol,
            // Use symbol's own exchange if available; then exchange selection; then infer from model filename.
            exchange: selectedSymbol.exchange || exchangeCode || parseModelExchange(model) || undefined,
            includeFundamentals: true,
            forceLocal: true,
            modelName: model,
            targetPct,
            stopLossPct,
            lookForwardDays,
            buyThreshold,
          });
          const endTime = performance.now();
          const executionTime = Math.round(endTime - startTime);
          resultsMap.set(model, {
            ...response,
            executionTime,
          });
        } catch (err) {
          console.error(`Error testing model ${model}:`, err);
        }
      }

      if (resultsMap.size > 0) {
        setTestResults(resultsMap);

        // If multiple models, automatically select the one with best win rate for the detailed chart
        const sorted = Array.from(resultsMap.entries()).sort((a, b) => {
          const kA = calculateKPIs(a[1].testPredictions || []);
          const kB = calculateKPIs(b[1].testPredictions || []);
          return kB.winRate - kA.winRate;
        });

        if (sorted.length > 0) {
          setTestResult(sorted[0][1]);
        }
      } else {
        setTestError("No models tested successfully");
      }
    } catch (err) {
      setTestError(err instanceof Error ? err.message : "Test failed");
    } finally {
      setTestLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-black w-full">
      <div className="flex w-full flex-col gap-4 px-2 py-4 sm:gap-4 sm:px-4 sm:py-4">
        <header className="space-y-3 sm:space-y-4">
          <div className="inline-flex items-center gap-2 sm:gap-3 text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
            <Sparkles className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" />
            AI Model Sandbox
          </div>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-black tracking-tight text-white flex items-center gap-2 sm:gap-3">
            <div className="rounded-2xl bg-gradient-to-br from-indigo-600 to-indigo-700 p-2 sm:p-3 shadow-xl shadow-indigo-600/30">
              <Brain className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
            </div>
            <span>Model Test</span>
          </h1>
          <p className="text-xs sm:text-sm text-zinc-500 max-w-2xl leading-relaxed">
            Pick a trained model, filter symbols, and inspect prediction quality over historical data.
          </p>
        </header>

        <section className="rounded-2xl sm:rounded-3xl border border-white/5 bg-gradient-to-br from-zinc-900/60 to-zinc-950/40 backdrop-blur-xl p-6 sm:p-8 shadow-2xl">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Models Section */}
            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
              <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center gap-2 justify-between mb-3">
                <div className="flex items-center gap-2">
                  <LineChart className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" />
                  Models
                </div>
                <button
                  onClick={() => {
                    setUseMultipleModels(!useMultipleModels);
                    if (useMultipleModels) {
                      setSelectedModels(new Set());
                    } else {
                      setSelectedModel(null);
                    }
                  }}
                  className={`px-3 py-1 rounded-lg text-[9px] font-bold transition-all ${useMultipleModels ? 'bg-emerald-600/30 text-emerald-400 border border-emerald-500/30' : 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
                >
                  {useMultipleModels ? '✓ Multi' : 'Single'}
                </button>
              </div>
              <div className="mt-3 flex flex-col gap-3">
                {modelsLoading ? (
                  <div className="flex items-center gap-2 text-xs text-zinc-400">
                    <Loader2 className="h-4 w-4 animate-spin" /> Loading...
                  </div>
                ) : modelsError ? (
                  <div className="text-xs text-red-400">{modelsError}</div>
                ) : useMultipleModels ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {models.map((model) => {
                      const name = typeof model === "string" ? model : model.name;
                      const numFeatures = typeof model === "object" ? (model as any).num_features ?? (model as any).numFeatures : undefined;
                      const numParams = typeof model === "object" ? (model as any).num_parameters ?? (model as any).numParameters : undefined;
                      const trainingSamples = typeof model === "object" ? (model as any).trainingSamples ?? (model as any).training_samples : undefined;
                      return (
                        <label key={name} className="flex items-center gap-2 p-2 rounded-lg hover:bg-zinc-800/50 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={selectedModels.has(name)}
                            onChange={(e) => {
                              const newSet = new Set(selectedModels);
                              if (e.target.checked) {
                                newSet.add(name);
                              } else {
                                newSet.delete(name);
                              }
                              setSelectedModels(newSet);
                            }}
                            className="w-4 h-4 rounded"
                          />
                          <span className="flex-1 flex flex-col gap-0.5">
                            <span className="text-xs text-zinc-300 truncate">{name}</span>
                            {(numFeatures || numParams || trainingSamples) && (
                              <span className="flex flex-wrap gap-1 text-[9px] text-zinc-500">
                                {numFeatures && (
                                  <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-700">
                                    Feat: {numFeatures}
                                  </span>
                                )}
                                {numParams && (
                                  <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-700">
                                    Params: {numParams}
                                  </span>
                                )}
                                {trainingSamples && (
                                  <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-700">
                                    Samples: {trainingSamples}
                                  </span>
                                )}
                              </span>
                            )}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                ) : (
                  <Select
                    value={selectedModel ?? ""}
                    onValueChange={setSelectedModel}
                  >
                    <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                      {models.map((model) => {
                        const name = typeof model === "string" ? model : model.name;
                        const numFeatures = typeof model === "object" ? (model as any).num_features ?? (model as any).numFeatures : undefined;
                        const numParams = typeof model === "object" ? (model as any).num_parameters ?? (model as any).numParameters : undefined;
                        const trainingSamples = typeof model === "object" ? (model as any).trainingSamples ?? (model as any).training_samples : undefined;
                        return (
                          <SelectItem key={name} value={name} className="text-xs font-bold uppercase tracking-widest flex items-center justify-between gap-2">
                            <span className="truncate">{name}</span>
                            {(numFeatures || numParams || trainingSamples) && (
                              <span className="flex gap-1 text-[9px] text-zinc-500">
                                {numFeatures && <span>F:{numFeatures}</span>}
                                {numParams && <span>P:{numParams}</span>}
                                {trainingSamples && <span>S:{trainingSamples}</span>}
                              </span>
                            )}
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                )}
              </div>
            </div>

            {/* Exchange Section */}
            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5 space-y-4">
              <div className="flex items-center gap-2 text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
                <Building2 className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Exchange
              </div>
              <Select
                value={selectedExchange}
                onValueChange={setSelectedExchange}
              >
                <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                  <SelectValue placeholder="Select exchange" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                  {availableExchanges.map((exchangeOpt) => (
                    <SelectItem key={exchangeOpt} value={exchangeOpt} className="text-xs font-bold uppercase tracking-widest">
                      {exchangeOpt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
              <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center gap-2 mb-3">
                <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Model Stats
              </div>
              {selectedModelStats.length > 0 ? (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={selectedModelStats} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis
                        dataKey="name"
                        stroke="rgba(255,255,255,0.4)"
                        tick={{ fontSize: 10 }}
                        angle={-20}
                        textAnchor="end"
                        height={40}
                      />
                      <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 10 }} allowDecimals={false} />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0a", border: "1px solid rgba(255,255,255,0.08)", color: "#fff" }}
                      />
                      <Bar dataKey="trainingSamples" fill="#38bdf8" name="Samples" />
                      <Bar dataKey="numFeatures" fill="#a855f7" name="Features" />
                      <Bar dataKey="nEstimators" fill="#34d399" name="Estimators" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-56 flex items-center justify-center text-xs text-zinc-500">
                  Select a model to view stats.
                </div>
              )}
            </div>

            {/* Strategy Settings Section */}
            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
              <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" />
                  Strategy Window
                </div>
                {isAutoDetected && !useMultipleModels && (
                  <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                    Auto-Detected
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <Label className="text-[9px] uppercase text-zinc-500">Target %</Label>
                  <Select value={String(targetPct ?? 0.15)} onValueChange={(v: string) => { setTargetPct(Number(v)); setIsAutoDetected(false); }}>
                    <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                      {[0.01, 0.05, 0.10, 0.15, 0.20, 0.30].map((v: number) => (
                        <SelectItem key={v} value={v.toString()}>{(v * 100).toFixed(0)}%</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[9px] uppercase text-zinc-500">Stop %</Label>
                  <Select value={String(stopLossPct ?? 0.05)} onValueChange={(v: string) => { setStopLossPct(Number(v)); setIsAutoDetected(false); }}>
                    <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                      {[0.01, 0.03, 0.05, 0.07, 0.10].map((v: number) => (
                        <SelectItem key={v} value={v.toString()}>{(v * 100).toFixed(0)}%</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[9px] uppercase text-zinc-500">Days</Label>
                  <Select value={String(lookForwardDays ?? 20)} onValueChange={(v: string) => { setLookForwardDays(Number(v)); setIsAutoDetected(false); }}>
                    <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                      {[10, 15, 20, 30].map((v: number) => (
                        <SelectItem key={v} value={v.toString()}>{v} Days</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[9px] uppercase text-zinc-500">Sensitivity</Label>
                  <Select value={String(buyThreshold ?? 0.40)} onValueChange={(v: string) => { setBuyThreshold(Number(v)); setIsAutoDetected(false); }}>
                    <SelectTrigger className="h-9 sm:h-10 text-[10px] sm:text-xs bg-zinc-950/50 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                      {[0.10, 0.20, 0.30, 0.40, 0.45, 0.50].map((v: number) => (
                        <SelectItem key={v} value={v.toString()}>{(100 - v * 100).toFixed(0)}% (prob {v})</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <p className="text-[9px] text-zinc-600 mt-3 leading-tight italic">
                {isAutoDetected && !useMultipleModels
                  ? "Values detected from model file. You can override them above."
                  : "Specify the strategy target/stop for calculating Precision and Earn metrics."}
              </p>
            </div>
          </div>
        </section>

        {/* Symbols Section */}
        <section className="rounded-2xl sm:rounded-3xl border border-white/5 bg-gradient-to-br from-zinc-900/60 to-zinc-950/40 backdrop-blur-xl p-6 sm:p-8 shadow-2xl">
          <div className="space-y-4 sm:space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2 sm:gap-3 text-[11px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
                <Database className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Symbols
              </div>
              <div className="text-[9px] sm:text-[10px] uppercase tracking-[0.3em] text-zinc-600 bg-zinc-950/50 px-2 sm:px-3 py-1 rounded-full">
                {symbols.length} symbols
              </div>
            </div>

            <Label className="flex flex-col gap-2 text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
              Search
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
                <Input
                  type="text"
                  value={searchSymbolTerm}
                  onChange={(e) => setSearchSymbolTerm(e.target.value)}
                  placeholder="Symbol or name..."
                  className="h-10 sm:h-11 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 pl-9 sm:pl-10 pr-3 text-xs font-bold text-zinc-200 outline-none transition-colors hover:bg-zinc-950"
                />
              </div>
            </Label>

            {symbolsLoading ? (
              <div className="flex items-center gap-2 text-xs text-zinc-400 py-4">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading symbols...
              </div>
            ) : symbolsError ? (
              <div className="text-xs text-red-400 py-4">{symbolsError}</div>
            ) : symbols.length === 0 ? (
              <div className="text-xs text-zinc-500 py-4">No symbols found.</div>
            ) : (
              <Select
                value={selectedSymbol ? `${selectedSymbol.symbol}|${selectedSymbol.exchange}` : ""}
                onValueChange={(value: string) => {
                  const [symbol, exchangeValue] = value.split("|");
                  const match = symbols.find((s) => s.symbol === symbol && s.exchange === exchangeValue);
                  setSelectedSymbol(match ?? null);
                }}
              >
                <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                  <SelectValue placeholder="Select symbol" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200 max-h-[300px]">
                  {symbols.map((s) => (
                    <SelectItem key={`${s.symbol}-${s.exchange}`} value={`${s.symbol}|${s.exchange}`} className="text-xs font-bold uppercase tracking-widest">
                      <span className="flex items-center justify-between gap-3">
                        <span>{s.symbol} {s.name ? `- ${s.name}` : ""}</span>
                        {s.rowCount && <span className="text-zinc-500">({s.rowCount})</span>}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            <Button
              type="button"
              onClick={runTest}
              disabled={testLoading || !selectedSymbol}
              className="w-full h-10 sm:h-12 rounded-lg sm:rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 px-4 sm:px-6 text-xs sm:text-sm font-bold uppercase tracking-[0.2em] text-white shadow-xl shadow-indigo-600/30 transition-all hover:shadow-indigo-600/50 hover:from-indigo-500 hover:to-indigo-600 disabled:cursor-not-allowed disabled:from-indigo-600/30 disabled:to-indigo-700/30 disabled:shadow-none"
            >
              {testLoading ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" /> Running...
                </span>
              ) : (
                "Run Model Test"
              )}
            </Button>

            {testError && <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{testError}</div>}
          </div>
        </section>

        {/* Results Section */}
        <section className="rounded-2xl sm:rounded-3xl border border-white/5 bg-gradient-to-br from-zinc-900/60 to-zinc-950/40 backdrop-blur-xl p-6 sm:p-8 shadow-2xl">
          {testResults.size > 0 ? (
            <div className="space-y-6">
              {/* Multi-Model Results */}
              {testResults.size > 1 && (
                <>
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs sm:text-sm font-black uppercase tracking-[0.3em] text-zinc-500">
                      <LineChart className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-400 inline mr-2" /> Results Comparison ({testResults.size} Models)
                    </h3>
                  </div>

                  {/* Comparative win-rate chart */}
                  <div className="bg-zinc-950/40 rounded-xl border border-white/5 p-4 mb-4">
                    <h4 className="text-[10px] sm:text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-3">
                      Win Rate by Model
                    </h4>
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={Array.from(testResults.entries()).map(([modelName, result]) => {
                            const k = calculateKPIs(result.testPredictions || []);
                            return {
                              name: modelName.replace(/^model_|\.pkl$/gi, ""),
                              winRate: k.winRate,
                              buyAcc: k.buyAccuracy,
                              sellAcc: k.sellAccuracy,
                            };
                          })}
                          margin={{ top: 10, right: 20, left: 0, bottom: 30 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                          <XAxis
                            dataKey="name"
                            stroke="rgba(255,255,255,0.4)"
                            tick={{ fontSize: 10 }}
                            angle={-20}
                            textAnchor="end"
                            height={40}
                          />
                          <YAxis
                            stroke="rgba(255,255,255,0.4)"
                            tick={{ fontSize: 10 }}
                            tickFormatter={(v) => `${v.toFixed(0)}%`}
                          />
                          <Tooltip
                            formatter={(value: any, key: any) =>
                              typeof value === "number" ? `${value.toFixed(1)}%` : value
                            }
                            contentStyle={{
                              backgroundColor: "rgba(9,9,11,0.98)",
                              border: "1px solid rgba(99,102,241,0.3)",
                              borderRadius: 10,
                              padding: "8px 12px",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Bar dataKey="winRate" name="Win Rate" fill="#6366f1" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="buyAcc" name="Buy Acc" fill="#22c55e" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="sellAcc" name="Sell Acc" fill="#f97373" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Detailed Results Table */}
                  <div className="bg-zinc-950/40 rounded-xl border border-white/5 overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="bg-white/5">
                            {[
                              { label: 'Model', key: 'name' },
                              { label: 'Win Rate', key: 'winRate' },
                              { label: 'Buy Acc', key: 'buyAcc' },
                              { label: 'Sell Acc', key: 'sellAcc' },
                              { label: 'Prec.', key: 'precision' },
                              { label: 'Earn', key: 'earnRate' },
                              { label: 'Recall', key: 'recall' },
                              { label: 'F1', key: 'f1' },
                              { label: 'Time', key: 'execTime' },
                            ].map((col) => (
                              <th
                                key={col.key}
                                onClick={() => toggleSort(col.key)}
                                className="px-4 py-3 text-[10px] font-black uppercase tracking-widest text-zinc-500 cursor-pointer hover:bg-white/5 transition-colors"
                              >
                                <div className="flex items-center">
                                  {col.label}
                                  <SortIcon column={col.key} />
                                </div>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {sortedMultiSummaries.map((item) => {
                            const cls = calculateClassification(item.result.testPredictions || []);
                            const execTime = (item.result as any).executionTime;
                            return (
                              <tr key={item.modelName} className="hover:bg-white/5 transition-colors">
                                <td className="px-4 py-3 text-xs font-bold text-zinc-300 truncate max-w-[150px]">
                                  {item.modelName.replace(/^model_|\.pkl$/gi, "")}
                                </td>
                                <td className="px-4 py-3 text-xs font-bold text-indigo-400">
                                  {item.kpis.winRate.toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-xs font-bold text-emerald-400">
                                  {item.kpis.buyAccuracy.toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-xs font-bold text-rose-400">
                                  {item.kpis.sellAccuracy.toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-xs text-zinc-400">
                                  {(cls.precisionBuy * 100).toFixed(1)}%
                                </td>
                                <td className={`px-4 py-3 text-xs font-bold ${item.result.earnPercentage != null ? (item.result.earnPercentage >= 0 ? "text-emerald-400" : "text-rose-400") : "text-zinc-500"}`}>
                                  {item.result.earnPercentage != null ? `${item.result.earnPercentage >= 0 ? '+' : ''}${item.result.earnPercentage.toFixed(1)}%` : '-'}
                                </td>
                                <td className="px-4 py-3 text-xs text-zinc-400">
                                  {(cls.recallBuy * 100).toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-xs text-zinc-400">
                                  {(cls.f1Buy * 100).toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-xs text-zinc-500 italic">
                                  {execTime ? `${execTime}ms` : '-'}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="bg-zinc-950/40 rounded-xl border border-white/5 p-4 mb-4">

                    <h4 className="text-[10px] sm:text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-3">
                      Classification (BUY) by Model
                    </h4>
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={multiClassificationChart} margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                          <XAxis
                            dataKey="name"
                            stroke="rgba(255,255,255,0.4)"
                            tick={{ fontSize: 10 }}
                            angle={-20}
                            textAnchor="end"
                            height={40}
                          />
                          <YAxis
                            stroke="rgba(255,255,255,0.4)"
                            tick={{ fontSize: 10 }}
                            tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                          />
                          <Tooltip
                            formatter={(value: any) => (typeof value === "number" ? `${value.toFixed(1)}%` : value)}
                            contentStyle={{
                              backgroundColor: "rgba(9,9,11,0.98)",
                              border: "1px solid rgba(99,102,241,0.3)",
                              borderRadius: 10,
                              padding: "8px 12px",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Bar dataKey="precision" name="Precision" fill="#22c55e" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="recall" name="Recall" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="f1" name="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {multiSummaries.map(({ modelName, result, kpis }, idx) => (
                      <div
                        key={modelName}
                        className={`rounded-xl p-4 border bg-zinc-900/50 ${idx === 0
                          ? "border-emerald-500/60 shadow-lg shadow-emerald-500/30"
                          : "border-white/5"
                          }`}
                      >
                        <div className="mb-3 pb-3 border-b border-white/5 flex items-center justify-between gap-2">
                          <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-indigo-400 truncate">
                            {modelName}
                          </h4>
                          {idx === 0 && (
                            <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-emerald-400 bg-emerald-500/10 border border-emerald-500/40 rounded-full px-2 py-1">
                              Best Win Rate
                            </span>
                          )}
                        </div>

                        <div className="mb-3 grid grid-cols-2 sm:grid-cols-3 gap-2 text-[10px] sm:text-xs text-zinc-400">
                          <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-white/5">
                            <div className="uppercase tracking-wider text-[9px] text-zinc-500">Win%</div>
                            <div className="font-semibold text-indigo-300">
                              {kpis.winRate.toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-emerald-500/30">
                            <div className="uppercase tracking-wider text-[9px] text-emerald-400">Buy Acc</div>
                            <div className="font-semibold text-emerald-300">
                              {kpis.buyAccuracy.toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-rose-500/30">
                            <div className="uppercase tracking-wider text-[9px] text-rose-400">Sell Acc</div>
                            <div className="font-semibold text-rose-300">
                              {kpis.sellAccuracy.toFixed(1)}%
                            </div>
                          </div>
                          <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-white/10">
                            <div className="uppercase tracking-wider text-[9px] text-zinc-500">Tests</div>
                            <div className="font-semibold text-zinc-200">
                              {kpis.totalTests}
                            </div>
                          </div>
                          <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-yellow-500/30">
                            <div className="uppercase tracking-wider text-[9px] text-yellow-400">Max Streak</div>
                            <div className="font-semibold text-yellow-300">
                              {kpis.maxConsecutiveWins}
                            </div>
                          </div>
                          {result.executionTime != null && (
                            <div className="bg-zinc-900/70 rounded-lg px-2 py-1.5 border border-sky-500/30">
                              <div className="uppercase tracking-wider text-[9px] text-sky-400">Latency</div>
                              <div className="font-semibold text-sky-300">
                                {result.executionTime} ms
                              </div>
                            </div>
                          )}
                        </div>

                        {renderResultContent(result)}
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* Single Model or First Model Details */}
              {testResult && testResult.testPredictions?.length ? (
                <div className="space-y-6">
                  <div className="flex flex-wrap items-center justify-between gap-3 sm:gap-4">
                    <div className="flex items-center gap-2 text-xs sm:text-sm font-bold uppercase tracking-[0.2em] text-zinc-400">
                      <LineChart className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-400" /> Detailed Results
                    </div>
                    {singleSummary && (
                      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                        <div className="flex flex-wrap items-center gap-2 text-[9px] sm:text-[10px] uppercase tracking-[0.2em] text-zinc-600">
                          <div>
                            Precision: {" "}
                            <span className="text-emerald-400">
                              {(testResult.precision * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="hidden sm:inline">
                            Close:{" "}
                            <span className="text-zinc-300">
                              ${testResult.lastClose?.toFixed?.(2)}
                            </span>
                          </div>
                          <div className="hidden md:inline">
                            Date:{" "}
                            <span className="text-zinc-300">{testResult.lastDate}</span>
                          </div>
                        </div>
                        <div
                          className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm tracking-wide ${singleSummary.signal === 1
                            ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                            : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                            }`}
                        >
                          {singleSummary.signal === 1 ? (
                            <TrendingUp className="h-5 w-5" />
                          ) : (
                            <TrendingDown className="h-5 w-5" />
                          )}
                          {signalLabel(singleSummary.signal)} ({singleSummary.kpis.winRate.toFixed(1)}%)
                        </div>
                      </div>
                    )}
                  </div>

                  {singleSummary && (
                    <div className="grid gap-3 sm:gap-4 grid-cols-2 sm:grid-cols-4 lg:grid-cols-8">
                      <div className="bg-zinc-900/50 rounded-lg p-3 border border-white/5">
                        <div className="text-[9px] text-zinc-500 uppercase tracking-wider">Days</div>
                        <div className="text-lg sm:text-xl font-bold text-white mt-1">
                          {singleSummary.kpis.totalDays}
                        </div>
                      </div>
                      <div className="bg-zinc-900/50 rounded-lg p-3 border border-white/5">
                        <div className="text-[9px] text-zinc-500 uppercase tracking-wider">Tests</div>
                        <div className="text-lg sm:text-xl font-bold text-white mt-1">
                          {singleSummary.kpis.totalTests}
                        </div>
                      </div>
                      <div className="bg-emerald-500/10 rounded-lg p-3 border border-emerald-500/20">
                        <div className="text-[9px] text-emerald-400 uppercase tracking-wider">Buy</div>
                        <div className="text-lg sm:text-xl font-bold text-emerald-400 mt-1">
                          {singleSummary.kpis.buySignals}
                        </div>
                      </div>
                      <div className="bg-rose-500/10 rounded-lg p-3 border border-rose-500/20">
                        <div className="text-[9px] text-rose-400 uppercase tracking-wider">Sell</div>
                        <div className="text-lg sm:text-xl font-bold text-rose-400 mt-1">
                          {singleSummary.kpis.sellSignals}
                        </div>
                      </div>
                      <div className="bg-indigo-500/10 rounded-lg p-3 border border-indigo-500/20">
                        <div className="text-[9px] text-indigo-400 uppercase tracking-wider">Win Rate</div>
                        <div className="text-lg sm:text-xl font-bold text-indigo-400 mt-1">
                          {singleSummary.kpis.winRate.toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-emerald-500/10 rounded-lg p-3 border border-emerald-500/20">
                        <div className="text-[9px] text-emerald-400 uppercase tracking-wider">Buy Acc</div>
                        <div className="text-lg sm:text-xl font-bold text-emerald-400 mt-1">
                          {singleSummary.kpis.buyAccuracy.toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-rose-500/10 rounded-lg p-3 border border-rose-500/20">
                        <div className="text-[9px] text-rose-400 uppercase tracking-wider">Sell Acc</div>
                        <div className="text-lg sm:text-xl font-bold text-rose-400 mt-1">
                          {singleSummary.kpis.sellAccuracy.toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-yellow-500/10 rounded-lg p-3 border border-yellow-500/20">
                        <div className="text-[9px] text-yellow-400 uppercase tracking-wider">Streak</div>
                        <div className="text-lg sm:text-xl font-bold text-yellow-400 mt-1">
                          {singleSummary.kpis.maxConsecutiveWins}
                        </div>
                      </div>
                      <div className={`rounded-lg p-3 border ${testResult.earnPercentage != null ? (testResult.earnPercentage >= 0 ? "bg-emerald-500/10 border-emerald-500/20" : "bg-rose-500/10 border-rose-500/20") : "bg-zinc-900/50 border-white/5"}`}>
                        <div className={`text-[9px] uppercase tracking-wider ${testResult.earnPercentage != null ? (testResult.earnPercentage >= 0 ? "text-emerald-400" : "text-rose-400") : "text-zinc-500"}`}>Earn Rate</div>
                        <div className={`text-lg sm:text-xl font-bold mt-1 ${testResult.earnPercentage != null ? (testResult.earnPercentage >= 0 ? "text-emerald-400" : "text-rose-400") : "text-white"}`}>
                          {testResult.earnPercentage != null ? `${testResult.earnPercentage >= 0 ? '+' : ''}${testResult.earnPercentage.toFixed(1)}%` : '-'}
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="bg-zinc-950/40 rounded-xl p-4 sm:p-6 border border-white/5">
                    <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500 mb-4">
                      Classification Performance
                    </h3>
                    {(() => {
                      const cls = calculateClassification(testResult!.testPredictions || []);
                      const prData = [
                        { name: "BUY", precision: cls.precisionBuy * 100, recall: cls.recallBuy * 100, f1: cls.f1Buy * 100 },
                        { name: "SELL", precision: cls.precisionSell * 100, recall: cls.recallSell * 100, f1: cls.f1Sell * 100 },
                      ];
                      return (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          <div className="h-56">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={prData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 11 }} />
                                <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
                                <Tooltip
                                  formatter={(value: any) => (typeof value === "number" ? `${value.toFixed(1)}%` : value)}
                                  contentStyle={{
                                    backgroundColor: "rgba(9,9,11,0.98)",
                                    border: "1px solid rgba(99,102,241,0.3)",
                                    borderRadius: 10,
                                    padding: "8px 12px",
                                  }}
                                />
                                <Legend wrapperStyle={{ fontSize: 10 }} />
                                <Bar dataKey="precision" name="Precision" fill="#22c55e" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="recall" name="Recall" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="f1" name="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>

                          <div className="grid grid-cols-2 gap-3">
                            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3">
                              <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-400">TP (BUY correct)</div>
                              <div className="mt-1 text-2xl font-black text-emerald-300">{cls.tp}</div>
                            </div>
                            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3">
                              <div className="text-[10px] uppercase tracking-[0.2em] text-rose-400">FP (BUY wrong)</div>
                              <div className="mt-1 text-2xl font-black text-rose-300">{cls.fp}</div>
                            </div>
                            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3">
                              <div className="text-[10px] uppercase tracking-[0.2em] text-amber-400">FN (missed BUY)</div>
                              <div className="mt-1 text-2xl font-black text-amber-300">{cls.fn}</div>
                            </div>
                            <div className="rounded-xl border border-sky-500/30 bg-sky-500/10 p-3">
                              <div className="text-[10px] uppercase tracking-[0.2em] text-sky-400">TN (SELL correct)</div>
                              <div className="mt-1 text-2xl font-black text-sky-300">{cls.tn}</div>
                            </div>
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  {/* Chart */}
                  <div className="bg-zinc-950/40 rounded-xl p-4 sm:p-6 border border-white/5">
                    <div className="flex flex-col gap-4 mb-6">
                      <div className="flex items-center justify-between">
                        <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500">
                          Price & Signals Chart (Candlestick)
                        </h3>
                      </div>

                      {/* Signal Controls */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
                        <Button
                          variant={showBUYSignals ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowBUYSignals(!showBUYSignals)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <TrendingUp className="w-3 h-3 text-emerald-400 mr-1.5" />
                          BUY
                        </Button>
                        <Button
                          variant={showVolume ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowVolume(!showVolume)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <BarChart3 className="w-3 h-3 text-blue-400 mr-1.5" />
                          Volume
                        </Button>
                      </div>

                      {/* Indicator Controls */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
                        <Button
                          variant={showSMA50 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowSMA50(!showSMA50)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <LineChart className="w-3 h-3 text-amber-500 mr-1.5" />
                          SMA50
                        </Button>
                        <Button
                          variant={showSMA200 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowSMA200(!showSMA200)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <LineChart className="w-3 h-3 text-cyan-500 mr-1.5" />
                          SMA200
                        </Button>
                        <Button
                          variant={showEMA50 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowEMA50(!showEMA50)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <TrendingUp className="w-3 h-3 text-orange-500 mr-1.5" />
                          EMA50
                        </Button>
                        <Button
                          variant={showEMA200 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowEMA200(!showEMA200)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <TrendingUp className="w-3 h-3 text-sky-500 mr-1.5" />
                          EMA200
                        </Button>
                        <Button
                          variant={showBB ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowBB(!showBB)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <CircleDot className="w-3 h-3 text-violet-500 mr-1.5" />
                          BB
                        </Button>
                        <Button
                          variant={showRSI ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowRSI(!showRSI)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <Gauge className="w-3 h-3 text-purple-500 mr-1.5" />
                          RSI
                        </Button>
                        <Button
                          variant={showMACD ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowMACD(!showMACD)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <LayoutDashboard className="w-3 h-3 text-indigo-500 mr-1.5" />
                          MACD
                        </Button>
                      </div>
                    </div>

                    {/* Candlestick Chart */}
                    <TestModelCandleChart
                      rows={predictionsWithOutcome}
                      showBuySignals={showBUYSignals}
                      showSellSignals={showSELLSignals}
                      showSMA50={showSMA50}
                      showSMA200={showSMA200}
                      showEMA50={showEMA50}
                      showEMA200={showEMA200}
                      showBB={showBB}
                      showRSI={showRSI}
                      showMACD={showMACD}
                      showVolume={showVolume}
                    />
                  </div>

                </div>
              ) : null}
            </div>
          ) : (
            <div className="text-xs sm:text-sm text-zinc-500 py-8 text-center">
              No test rows returned.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
