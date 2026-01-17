"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import { Brain, CalendarRange, Loader2, LineChart, Sparkles, Database, Building2, Search, TrendingUp, TrendingDown } from "lucide-react";
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from "recharts";
import { predictStock, getLocalModels, getSymbolsByDate, getCountries, getSymbolsForExchange } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import type { DateSymbolResult } from "@/lib/api";
import { useAppState } from "@/contexts/AppStateContext";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

const defaultStart = "2023-01-01";
const defaultEnd = new Date().toISOString().slice(0, 10);

function parseModelExchange(modelName: string | null): string | null {
  if (!modelName) return null;
  const match = modelName.match(/model_(.+?)\\.pkl/i);
  return match ? match[1].toUpperCase() : null;
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
      totalTests: 0,
      totalDays: 0,
      buySignals: 0,
      sellSignals: 0,
      buyCorrect: 0,
      sellCorrect: 0,
      buyAccuracy: 0,
      sellAccuracy: 0,
      totalCorrect: 0,
      totalIncorrect: 0,
      winRate: 0,
      consecutiveWins: 0,
      consecutiveLosses: 0,
      maxConsecutiveWins: 0,
      maxConsecutiveLosses: 0,
    };
  }

  const uniqueDates = new Set(predictions.map((p) => p.date)).size;
  const buySignals = predictions.filter((p) => p.pred === 1).length;
  const sellSignals = predictions.filter((p) => p.pred === 0).length;
  
  const buyCorrect = predictions.filter((p) => p.pred === 1 && p.pred === p.target).length;
  const sellCorrect = predictions.filter((p) => p.pred === 0 && p.pred === p.target).length;
  
  const totalCorrect = predictions.filter((p) => p.pred === p.target).length;
  const totalIncorrect = predictions.length - totalCorrect;

  let consecutiveWins = 0;
  let consecutiveLosses = 0;
  let maxConsecutiveWins = 0;
  let maxConsecutiveLosses = 0;

  for (const pred of predictions) {
    if (pred.pred === pred.target) {
      consecutiveWins++;
      consecutiveLosses = 0;
      maxConsecutiveWins = Math.max(maxConsecutiveWins, consecutiveWins);
    } else {
      consecutiveLosses++;
      consecutiveWins = 0;
      maxConsecutiveLosses = Math.max(maxConsecutiveLosses, consecutiveLosses);
    }
  }

  return {
    totalTests: predictions.length,
    totalDays: uniqueDates,
    buySignals,
    sellSignals,
    buyCorrect,
    sellCorrect,
    buyAccuracy: buySignals > 0 ? (buyCorrect / buySignals) * 100 : 0,
    sellAccuracy: sellSignals > 0 ? (sellCorrect / sellSignals) * 100 : 0,
    totalCorrect,
    totalIncorrect,
    winRate: (totalCorrect / predictions.length) * 100,
    consecutiveWins,
    consecutiveLosses,
    maxConsecutiveWins,
    maxConsecutiveLosses,
  };
}

function calculateClassification(predictions: any[]) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (const p of predictions || []) {
    const pred = p?.pred;
    const target = p?.target;
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

  return {
    tp,
    fp,
    tn,
    fn,
    precisionBuy,
    recallBuy,
    f1Buy,
    precisionSell,
    recallSell,
    f1Sell,
  };
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

  const [showRSI, setShowRSI] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showSMA200, setShowSMA200] = useState(false);
  const [showBUYSignals, setShowBUYSignals] = useState(true);
  const [showSELLSignals, setShowSELLSignals] = useState(true);
  const [chartType, setChartType] = useState<"line" | "scatter">("line");

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
    if (modelExchange && !selectedExchange) {
      setSelectedExchange(modelExchange);
    }
    if (!selectedExchange && availableExchanges.length > 0) {
      setSelectedExchange(availableExchanges[0]);
    }
  }, [modelExchange, selectedExchange, availableExchanges]);

  useEffect(() => {
    refreshAvailableExchanges();
  }, [refreshAvailableExchanges]);

  const exchangeCode = useMemo(() => {
    const invItem = inventory.find(item => item.country === selectedExchange);
    return invItem?.exchange || selectedExchange;
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
            // Use explicit exchange selection if available; otherwise infer from model filename.
            exchange: exchangeCode || parseModelExchange(model) || undefined,
            includeFundamentals: true,
            modelName: model,
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
        if (resultsMap.size === 1) {
          setTestResult(Array.from(resultsMap.values())[0]);
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
                          {(numFeatures || numParams) && (
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
                            </span>
                          )}
                        </span>
                      </label>
                    );})}
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
                        return (
                        <SelectItem key={name} value={name} className="text-xs font-bold uppercase tracking-widest flex items-center justify-between gap-2">
                          <span className="truncate">{name}</span>
                          {(numFeatures || numParams) && (
                            <span className="flex gap-1 text-[9px] text-zinc-500">
                              {numFeatures && <span>F:{numFeatures}</span>}
                              {numParams && <span>P:{numParams}</span>}
                            </span>
                          )}
                        </SelectItem>
                      );})}
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
                        className={`rounded-xl p-4 border bg-zinc-900/50 ${
                          idx === 0
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
                          className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm tracking-wide ${
                            singleSummary.signal === 1
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
                          Price & Signals Chart
                        </h3>
                        <div className="flex gap-2">
                          <Select value={chartType} onValueChange={(v: any) => setChartType(v)}>
                            <SelectTrigger className="w-32 h-8 text-xs">
                              <SelectValue placeholder="Chart Type" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="line">Line Chart</SelectItem>
                              <SelectItem value="scatter">Scatter Plot</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <Button
                          variant={showBUYSignals ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowBUYSignals(!showBUYSignals)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <div className="w-2 h-2 rounded-full bg-emerald-500 mr-1.5" />
                          BUY Signals
                        </Button>
                        <Button
                          variant={showSELLSignals ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowSELLSignals(!showSELLSignals)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <div className="w-2 h-2 rotate-45 bg-red-500 mr-1.5" />
                          SELL Signals
                        </Button>
                        <Button
                          variant={showSMA50 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowSMA50(!showSMA50)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <div className="w-2 h-2 bg-amber-500 mr-1.5" />
                          SMA50
                        </Button>
                        <Button
                          variant={showSMA200 ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowSMA200(!showSMA200)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8"
                        >
                          <div className="w-2 h-2 bg-cyan-500 mr-1.5" />
                          SMA200
                        </Button>
                        <Button
                          variant={showRSI ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowRSI(!showRSI)}
                          className="text-[10px] sm:text-xs h-7 sm:h-8 col-span-2 sm:col-span-1"
                        >
                          <div className="w-2 h-2 bg-purple-500 mr-1.5" />
                          RSI
                        </Button>
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={450}>
                      <RechartsLineChart
                        data={testResult!.testPredictions.map((row) => ({
                          ...row,
                          date: row.date,
                          signalColor: row.pred === 1 ? "#10b981" : "#ef4444",
                          signalName: row.pred === 1 ? "BUY" : "SELL",
                        }))}
                        margin={{ top: 10, right: 60, left: 10, bottom: 40 }}
                      >
                        <defs>
                          <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0.01} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis
                          dataKey="date"
                          stroke="rgba(255,255,255,0.4)"
                          tick={{ fontSize: 11 }}
                          interval={Math.floor(
                            Math.max(0, testResult!.testPredictions.length - 1) / 8
                          )}
                          angle={-45}
                          textAnchor="end"
                          height={60}
                        />
                        <YAxis
                          stroke="rgba(255,255,255,0.4)"
                          tick={{ fontSize: 11 }}
                          yAxisId="left"
                          label={{ value: "Price", angle: -90, position: "insideLeft" }}
                        />
                        <YAxis
                          stroke="rgba(255,255,255,0.4)"
                          tick={{ fontSize: 11 }}
                          yAxisId="right"
                          orientation="right"
                          label={{ value: "RSI", angle: 90, position: "insideRight" }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "rgba(9,9,11,0.98)",
                            border: "1px solid rgba(99,102,241,0.3)",
                            borderRadius: "10px",
                            padding: "8px 12px",
                          }}
                          labelStyle={{ color: "rgba(255,255,255,0.9)" }}
                          formatter={(value: any) =>
                            typeof value === "number" ? value.toFixed(2) : value
                          }
                          cursor={{ stroke: "rgba(99,102,241,0.2)", strokeWidth: 1 }}
                        />
                        <Legend
                          verticalAlign="top"
                          height={20}
                          wrapperStyle={{ paddingBottom: "10px" }}
                        />
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="close"
                          stroke="#6366f1"
                          dot={false}
                          strokeWidth={2.5}
                          isAnimationActive={false}
                          name="Close Price"
                        />
                        {showSMA50 &&
                          testResult!.testPredictions.some((p) => p.sma50) && (
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="sma50"
                              stroke="#f59e0b"
                              dot={false}
                              strokeWidth={1.5}
                              strokeDasharray="5 5"
                              isAnimationActive={false}
                              name="SMA50"
                            />
                          )}
                        {showSMA200 &&
                          testResult!.testPredictions.some((p) => p.sma200) && (
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="sma200"
                              stroke="#06b6d4"
                              dot={false}
                              strokeWidth={1.5}
                              strokeDasharray="5 5"
                              isAnimationActive={false}
                              name="SMA200"
                            />
                          )}
                        {showRSI && testResult!.testPredictions.some((p) => p.rsi) && (
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="rsi"
                            stroke="#8b5cf6"
                            dot={false}
                            strokeWidth={1.5}
                            strokeDasharray="5 5"
                            isAnimationActive={false}
                            name="RSI"
                          />
                        )}
                        {showBUYSignals && (
                          <Scatter
                            yAxisId="left"
                            dataKey="close"
                            data={testResult!.testPredictions.filter((p) => p.pred === 1)}
                            fill="#10b981"
                            stroke="#059669"
                            strokeWidth={2}
                            name="BUY Signal (●)"
                            shape="circle"
                          />
                        )}
                        {showSELLSignals && (
                          <Scatter
                            yAxisId="left"
                            dataKey="close"
                            data={testResult!.testPredictions.filter((p) => p.pred === 0)}
                            fill="#ef4444"
                            stroke="#dc2626"
                            strokeWidth={2}
                            name="SELL Signal (◆)"
                            shape="diamond"
                          />
                        )}
                      </RechartsLineChart>
                    </ResponsiveContainer>

                    <div className="flex flex-wrap gap-4 mt-6 text-xs">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-indigo-500 border border-indigo-600" />
                        <span className="text-zinc-400">Close Price</span>
                      </div>
                      {showSMA50 &&
                        testResult!.testPredictions.some((p) => p.sma50) && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-1 bg-amber-500 border border-amber-600" />
                            <span className="text-zinc-400">SMA50</span>
                          </div>
                        )}
                      {showSMA200 &&
                        testResult!.testPredictions.some((p) => p.sma200) && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-1 bg-cyan-500 border border-cyan-600" />
                            <span className="text-zinc-400">SMA200</span>
                          </div>
                        )}
                      {showRSI && testResult!.testPredictions.some((p) => p.rsi) && (
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-1 bg-purple-500 border border-purple-600" />
                          <span className="text-zinc-400">RSI</span>
                        </div>
                      )}
                      {showBUYSignals && (
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-emerald-500 border border-emerald-600" />
                          <span className="text-zinc-400">BUY Signal (●)</span>
                        </div>
                      )}
                      {showSELLSignals && (
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rotate-45 bg-red-500 border border-red-600" />
                          <span className="text-zinc-400">SELL Signal (◆)</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Save button + weekly breakdown table */}
                  <Button
                    type="button"
                    onClick={async () => {
                      setSaveLoading(true);
                      try {
                        const response = await fetch("/api/model-test/save", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            modelName: selectedModel,
                            symbol: selectedSymbol?.symbol,
                            exchange: exchangeCode,
                            predictions: testResult!.testPredictions,
                            executionTime: testResult!.executionTime || 0,
                          }),
                        });
                        if (response.ok) {
                          alert("✅ Test results saved successfully!");
                        }
                      } catch (err) {
                        alert("❌ Failed to save results");
                      } finally {
                        setSaveLoading(false);
                      }
                    }}
                    disabled={saveLoading}
                    className="w-full h-10 sm:h-12 rounded-lg sm:rounded-xl bg-gradient-to-r from-amber-600 to-amber-700 px-4 sm:px-6 text-xs sm:text-sm font-bold uppercase tracking-[0.2em] text-white shadow-xl shadow-amber-600/30 transition-all hover:shadow-amber-600/50 disabled:opacity-50"
                  >
                    {saveLoading ? "Saving..." : "💾 Save Results"}
                  </Button>

                  <div className="bg-zinc-950/40 rounded-xl border border-white/5 overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs sm:text-sm">
                        <thead className="bg-zinc-950/80 text-[9px] sm:text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500 sticky top-0">
                          <tr>
                            <th className="px-2 sm:px-4 py-3">Period</th>
                            <th className="px-2 sm:px-4 py-3">Symbol</th>
                            <th className="px-2 sm:px-4 py-3 text-center">Buy Signals</th>
                            <th className="px-2 sm:px-4 py-3 text-center">Sell Signals</th>
                            <th className="px-2 sm:px-4 py-3 text-center">Correct</th>
                            <th className="px-2 sm:px-4 py-3 text-right">Win Rate</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {groupPredictionsByWeek(testResult!.testPredictions).map(
                            (group, idx) => {
                              const weekKPIs = calculateKPIs(group.predictions);
                              return (
                                <tr
                                  key={`week-${idx}`}
                                  className="text-xs sm:text-sm text-zinc-300 hover:bg-zinc-900/30 transition-colors"
                                >
                                  <td className="px-2 sm:px-4 py-3 font-mono text-zinc-400">
                                    {new Date(group.week).toLocaleDateString("en-US", {
                                      month: "short",
                                      day: "numeric",
                                    })}
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 font-bold text-white">
                                    {selectedSymbol?.symbol || "-"}
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 text-center">
                                    <span className="inline-flex items-center h-6 px-2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px] sm:text-[10px] font-bold">
                                      {weekKPIs.buySignals}
                                    </span>
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 text-center">
                                    <span className="inline-flex items-center h-6 px-2 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[9px] sm:text-[10px] font-bold">
                                      {weekKPIs.sellSignals}
                                    </span>
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 text-center">
                                    <span className="inline-flex items-center h-6 px-2 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[9px] sm:text-[10px] font-bold">
                                      {weekKPIs.totalCorrect}/{weekKPIs.totalTests}
                                    </span>
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 text-right font-mono text-amber-400">
                                    {weekKPIs.winRate.toFixed(1)}%
                                  </td>
                                </tr>
                              );
                            }
                          )}
                        </tbody>
                      </table>
                    </div>
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
