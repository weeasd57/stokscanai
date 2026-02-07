"use client";

import { useEffect, useMemo, useReducer, useState } from "react";
import { Brain, Sparkles } from "lucide-react";
import { predictStock, getLocalModels, getSymbolsForExchange } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import type { DateSymbolResult } from "@/lib/api";
import { useAppState } from "@/contexts/AppStateContext";

// Decomposed Components
import ModelSelector from "./test-model/ModelSelector";
import ExchangeSelector from "./test-model/ExchangeSelector";
import StrategySettings from "./test-model/StrategySettings";
import ModelStatsChart from "./test-model/ModelStatsChart";
import ResultsComparison from "./test-model/ResultsComparison";
import DetailedResults from "./test-model/DetailedResults";
import AIBrainVisualizerSection from "./test-model/AIBrainVisualizerSection";

// Utilities
import {
  parseModelExchange,
  calculateKPIs,
  calculateClassification
} from "./test-model/utils";
import { SortConfig } from "./test-model/types";

const defaultStart = "2023-01-01";
const defaultEnd = new Date().toISOString().slice(0, 10);

interface TestModelState {
  models: any[];
  modelsLoading: boolean;
  modelsError: string | null;
  selectedExchange: string;
  symbols: DateSymbolResult[];
  symbolsLoading: boolean;
  symbolsError: string | null;
  searchSymbolTerm: string;
  selectedModel: string | null;
  selectedModels: Set<string>;
  useMultipleModels: boolean;
  selectedSymbol: DateSymbolResult | null;
  testLoading: boolean;
  testError: string | null;
  testResult: PredictResponse | null;
  testResults: Map<string, PredictResponse>;
  targetPct: number;
  stopLossPct: number;
  lookForwardDays: number;
  buyThreshold: number;
  asOfDate: string;
  isAutoDetected: boolean;
  uiSettings: {
    showRSI: boolean;
    showSMA50: boolean;
    showSMA200: boolean;
    showEMA50: boolean;
    showEMA200: boolean;
    showBB: boolean;
    showVolume: boolean;
    showMACD: boolean;
    showBUYSignals: boolean;
    showSELLSignals: boolean;
  };
  sortConfig: SortConfig | null;
  lastRunTimestamp: number;
}

type TestModelAction =
  | { type: 'SET_MODELS'; payload: any[] }
  | { type: 'SET_MODELS_LOADING'; payload: boolean }
  | { type: 'SET_MODELS_ERROR'; payload: string | null }
  | { type: 'SET_EXCHANGE'; payload: string }
  | { type: 'SET_SYMBOLS'; payload: DateSymbolResult[] }
  | { type: 'SET_SYMBOLS_LOADING'; payload: boolean }
  | { type: 'SET_SYMBOLS_ERROR'; payload: string | null }
  | { type: 'SET_SEARCH_TERM'; payload: string }
  | { type: 'SET_SELECTED_MODEL'; payload: string | null }
  | { type: 'SET_SELECTED_MODELS'; payload: Set<string> }
  | { type: 'TOGGLE_MULTI_MODE'; payload: boolean }
  | { type: 'SET_SELECTED_SYMBOL'; payload: DateSymbolResult | null }
  | { type: 'SET_TEST_LOADING'; payload: boolean }
  | { type: 'SET_TEST_ERROR'; payload: string | null }
  | { type: 'SET_SINGLE_RESULT'; payload: PredictResponse | null }
  | { type: 'SET_MULTI_RESULTS'; payload: Map<string, PredictResponse> }
  | { type: 'UPDATE_STRATEGY'; payload: Partial<Pick<TestModelState, 'targetPct' | 'stopLossPct' | 'lookForwardDays' | 'buyThreshold' | 'asOfDate' | 'isAutoDetected'>> }
  | { type: 'SET_UI', payload: { key: keyof TestModelState['uiSettings']; value: boolean } }
  | { type: 'SET_SORT'; payload: SortConfig | null }
  | { type: 'TRIGGER_RUN_UPDATE'; payload: number };

function testModelReducer(state: TestModelState, action: TestModelAction): TestModelState {
  switch (action.type) {
    case 'SET_MODELS': return { ...state, models: action.payload };
    case 'SET_MODELS_LOADING': return { ...state, modelsLoading: action.payload };
    case 'SET_MODELS_ERROR': return { ...state, modelsError: action.payload };
    case 'SET_EXCHANGE': return { ...state, selectedExchange: action.payload };
    case 'SET_SYMBOLS': return { ...state, symbols: action.payload };
    case 'SET_SYMBOLS_LOADING': return { ...state, symbolsLoading: action.payload };
    case 'SET_SYMBOLS_ERROR': return { ...state, symbolsError: action.payload };
    case 'SET_SEARCH_TERM': return { ...state, searchSymbolTerm: action.payload };
    case 'SET_SELECTED_MODEL': return { ...state, selectedModel: action.payload };
    case 'SET_SELECTED_MODELS': return { ...state, selectedModels: action.payload };
    case 'TOGGLE_MULTI_MODE': return { ...state, useMultipleModels: action.payload };
    case 'SET_SELECTED_SYMBOL': return { ...state, selectedSymbol: action.payload };
    case 'SET_TEST_LOADING': return { ...state, testLoading: action.payload };
    case 'SET_TEST_ERROR': return { ...state, testError: action.payload };
    case 'SET_SINGLE_RESULT': return { ...state, testResult: action.payload };
    case 'SET_MULTI_RESULTS': return { ...state, testResults: action.payload };
    case 'UPDATE_STRATEGY': return { ...state, ...action.payload };
    case 'SET_UI': return { ...state, uiSettings: { ...state.uiSettings, [action.payload.key]: action.payload.value } };
    case 'SET_SORT': return { ...state, sortConfig: action.payload };
    case 'TRIGGER_RUN_UPDATE': return { ...state, lastRunTimestamp: action.payload };
    default: return state;
  }
}

const initialState: TestModelState = {
  models: [],
  modelsLoading: false,
  modelsError: null,
  selectedExchange: "",
  symbols: [],
  symbolsLoading: false,
  symbolsError: null,
  searchSymbolTerm: "",
  selectedModel: null,
  selectedModels: new Set(),
  useMultipleModels: false,
  selectedSymbol: null,
  testLoading: false,
  testError: null,
  testResult: null,
  testResults: new Map(),
  targetPct: 0.15,
  stopLossPct: 0.05,
  lookForwardDays: 20,
  buyThreshold: 0.45,
  asOfDate: defaultEnd,
  isAutoDetected: false,
  uiSettings: {
    showRSI: true,
    showSMA50: true,
    showSMA200: false,
    showEMA50: false,
    showEMA200: false,
    showBB: false,
    showVolume: true,
    showMACD: true,
    showBUYSignals: true,
    showSELLSignals: true,
  },
  sortConfig: null,
  lastRunTimestamp: 0,
};

export default function TestModelTab() {
  const [state, dispatch] = useReducer(testModelReducer, initialState);
  const { countries: availableExchanges, refreshCountries: refreshAvailableExchanges, inventory } = useAppState();

  // Derived Values
  const selectedModelStats = useMemo(() => {
    if (!state.models.length) return [];
    const selected = state.useMultipleModels ? Array.from(state.selectedModels) : state.selectedModel ? [state.selectedModel] : [];
    if (!selected.length) return [];

    const byName = new Map(state.models.map((m) => [typeof m === "string" ? m : m.name, m]));

    return selected
      .map((name) => {
        const model = byName.get(name) as any;
        const numFeatures = model?.num_features ?? model?.numFeatures;
        const trainingSamples = model?.trainingSamples ?? model?.training_samples;
        const nEstimators = model?.n_estimators ?? model?.nEstimators;
        return {
          name: name.replace(/^model_|\.pkl$/gi, ""),
          trainingSamples: typeof trainingSamples === "number" ? trainingSamples : null,
          numFeatures: typeof numFeatures === "number" ? numFeatures : null,
          nEstimators: typeof nEstimators === "number" ? nEstimators : null,
        };
      })
      .filter((item) => item.trainingSamples !== null || item.numFeatures !== null || item.nEstimators !== null);
  }, [state.models, state.selectedModel, state.selectedModels, state.useMultipleModels]);

  const multiSummaries = useMemo(() => {
    if (!state.testResults.size) return [];
    const items = Array.from(state.testResults.entries()).map(([modelName, result]) => {
      const kpis = calculateKPIs(result.testPredictions || []);
      return { modelName, result, kpis };
    });
    return items.sort((a, b) => b.kpis.winRate - a.kpis.winRate);
  }, [state.testResults]);

  const predictionsWithOutcome = useMemo(() => {
    if (!state.testResult?.testPredictions) return [];
    const rows = [...state.testResult.testPredictions].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

    return rows.map((row, idx) => {
      if (row.pred !== 1) return { ...row, outcome: undefined };
      const entryPrice = row.close;
      const targetPrice = entryPrice * (1 + state.targetPct);
      const stopPrice = entryPrice * (1 - state.stopLossPct);

      for (let i = idx + 1; i < Math.min(idx + state.lookForwardDays + 1, rows.length); i++) {
        const future = rows[i];
        if ((future.low ?? future.close) <= stopPrice) return { ...row, outcome: "loss" as const };
        if ((future.high ?? future.close) >= targetPrice) return { ...row, outcome: "win" as const };
      }
      return { ...row, outcome: "pending" as const };
    });
  }, [state.testResult, state.targetPct, state.stopLossPct, state.lookForwardDays]);

  const sortedMultiSummaries = useMemo(() => {
    if (!state.sortConfig) return multiSummaries;
    return [...multiSummaries].sort((a, b) => {
      let aValue: any, bValue: any;
      const { key } = state.sortConfig!;
      if (key === "name") { aValue = a.modelName; bValue = b.modelName; }
      else if (key === "winRate") { aValue = a.kpis.winRate; bValue = b.kpis.winRate; }
      else if (key === "buyAcc") { aValue = a.kpis.buyAccuracy; bValue = b.kpis.buyAccuracy; }
      else if (key === "sellAcc") { aValue = a.kpis.sellAccuracy; bValue = b.kpis.sellAccuracy; }
      else if (key === "execTime") { aValue = (a.result as any).executionTime ?? 0; bValue = (b.result as any).executionTime ?? 0; }
      else if (key === "earnRate") { aValue = a.result.earnPercentage ?? -999; bValue = b.result.earnPercentage ?? -999; }
      else {
        const clsA = calculateClassification(a.result.testPredictions || []);
        const clsB = calculateClassification(b.result.testPredictions || []);
        if (key === "precision") { aValue = clsA.precisionBuy; bValue = clsB.precisionBuy; }
        else if (key === "recall") { aValue = clsA.recallBuy; bValue = clsB.recallBuy; }
        else if (key === "f1") { aValue = clsA.f1Buy; bValue = clsB.f1Buy; }
      }
      if (aValue < bValue) return state.sortConfig!.direction === "asc" ? -1 : 1;
      if (aValue > bValue) return state.sortConfig!.direction === "asc" ? 1 : -1;
      return 0;
    });
  }, [multiSummaries, state.sortConfig]);

  const multiClassificationChart = useMemo(() => {
    return Array.from(state.testResults.entries())
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
  }, [state.testResults]);

  const modelExchangeCode = useMemo(() => parseModelExchange(state.selectedModel), [state.selectedModel]);

  // Effects
  useEffect(() => {
    async function loadModels() {
      dispatch({ type: 'SET_MODELS_LOADING', payload: true });
      try {
        const data = await getLocalModels();
        dispatch({ type: 'SET_MODELS', payload: data });
        if (!state.selectedModel && data.length > 0) {
          dispatch({ type: 'SET_SELECTED_MODEL', payload: typeof data[0] === "string" ? data[0] : data[0].name });
        }
      } catch (err) {
        dispatch({ type: 'SET_MODELS_ERROR', payload: err instanceof Error ? err.message : "Failed to load models" });
      } finally {
        dispatch({ type: 'SET_MODELS_LOADING', payload: false });
      }
    }
    loadModels();
  }, [state.selectedModel]);

  useEffect(() => {
    if (state.selectedModel && !state.useMultipleModels) {
      const model = state.models.find((m) => (typeof m === "string" ? m : m.name) === state.selectedModel);
      if (model && typeof model === "object") {
        if (model.target_pct != null) {
          dispatch({
            type: 'UPDATE_STRATEGY', payload: {
              targetPct: model.target_pct,
              stopLossPct: model.stop_loss_pct ?? 0.05,
              lookForwardDays: model.look_forward_days ?? 20,
              isAutoDetected: true
            }
          });
        } else {
          dispatch({ type: 'UPDATE_STRATEGY', payload: { isAutoDetected: false } });
        }
      }
    }
  }, [state.selectedModel, state.models, state.useMultipleModels]);

  useEffect(() => {
    refreshAvailableExchanges();
  }, [refreshAvailableExchanges]);

  const currentExchangeCode = useMemo(() => {
    if (!state.selectedExchange) return "";
    const lower = state.selectedExchange.toLowerCase();
    const invItem = inventory.find((item) =>
      item.country?.toLowerCase() === lower || item.exchange?.toLowerCase() === lower
    );
    return invItem?.exchange || (state.selectedExchange.length <= 4 ? state.selectedExchange.toUpperCase() : state.selectedExchange);
  }, [state.selectedExchange, inventory]);

  useEffect(() => {
    async function loadSymbols() {
      if (!currentExchangeCode) return;
      dispatch({ type: 'SET_SYMBOLS_LOADING', payload: true });
      try {
        let data = await getSymbolsForExchange(currentExchangeCode);
        if (state.searchSymbolTerm) {
          const term = state.searchSymbolTerm.toLowerCase();
          data = data.filter((s) => s.symbol.toLowerCase().includes(term) || s.name.toLowerCase().includes(term));
        }
        dispatch({ type: 'SET_SYMBOLS', payload: data });
        if (data.length > 0) dispatch({ type: 'SET_SELECTED_SYMBOL', payload: data[0] });
      } catch (err) {
        dispatch({ type: 'SET_SYMBOLS_ERROR', payload: "Failed to load symbols" });
        dispatch({ type: 'SET_SYMBOLS', payload: [] });
      } finally {
        dispatch({ type: 'SET_SYMBOLS_LOADING', payload: false });
      }
    }
    loadSymbols();
  }, [currentExchangeCode, state.searchSymbolTerm]);

  // Handlers
  async function runTest() {
    if (!state.selectedSymbol) return;
    const modelsToTest = state.useMultipleModels ? Array.from(state.selectedModels) : state.selectedModel ? [state.selectedModel] : [];
    if (modelsToTest.length === 0) return;

    dispatch({ type: 'SET_TEST_LOADING', payload: true });
    dispatch({ type: 'SET_TEST_ERROR', payload: null });
    dispatch({ type: 'SET_MULTI_RESULTS', payload: new Map() });

    try {
      const resultsMap = new Map<string, PredictResponse>();
      for (const model of modelsToTest) {
        const start = performance.now();
        const res = await predictStock({
          ticker: state.selectedSymbol.symbol,
          exchange: state.selectedSymbol.exchange || currentExchangeCode || parseModelExchange(model) || undefined,
          modelName: model,
          toDate: state.asOfDate || undefined,
          targetPct: state.targetPct,
          stopLossPct: state.stopLossPct,
          lookForwardDays: state.lookForwardDays,
          buyThreshold: state.buyThreshold,
          forceLocal: true,
        });
        (res as any).executionTime = Math.round(performance.now() - start);
        resultsMap.set(model, res);
      }
      dispatch({ type: 'SET_MULTI_RESULTS', payload: resultsMap });
      if (resultsMap.size > 0) {
        // Pick best win rate for auto-selection
        const best = Array.from(resultsMap.entries()).sort((a, b) =>
          calculateKPIs(b[1].testPredictions || []).winRate - calculateKPIs(a[1].testPredictions || []).winRate
        )[0];
        dispatch({ type: 'SET_SINGLE_RESULT', payload: best[1] });
        dispatch({ type: 'TRIGGER_RUN_UPDATE', payload: Date.now() });
      }
    } catch (err) {
      dispatch({ type: 'SET_TEST_ERROR', payload: "Test failed" });
    } finally {
      dispatch({ type: 'SET_TEST_LOADING', payload: false });
    }
  }

  const toggleSort = (key: string) => {
    dispatch({
      type: 'SET_SORT', payload: {
        key,
        direction: state.sortConfig?.key === key && state.sortConfig.direction === "desc" ? "asc" : "desc",
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-black w-full text-zinc-200">
      <div className="flex w-full flex-col gap-4 px-2 py-4 sm:gap-4 sm:px-4 sm:py-4 max-w-7xl mx-auto">
        <header className="space-y-3">
          <div className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">
            <Sparkles className="h-3 w-3 text-indigo-400" /> AI Sandbox
          </div>
          <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white flex items-center gap-3">
            <div className="rounded-2xl bg-gradient-to-br from-indigo-600 to-indigo-700 p-2 shadow-xl shadow-indigo-600/30">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <span>Model Test</span>
          </h1>
        </header>

        <div className="grid gap-6">
          {/* Config Grid */}
          <section className="rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-2xl space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <ModelSelector
                models={state.models}
                modelsLoading={state.modelsLoading}
                modelsError={state.modelsError}
                useMultipleModels={state.useMultipleModels}
                setUseMultipleModels={(v) => dispatch({ type: 'TOGGLE_MULTI_MODE', payload: v })}
                selectedModel={state.selectedModel}
                setSelectedModel={(v) => dispatch({ type: 'SET_SELECTED_MODEL', payload: v })}
                selectedModels={state.selectedModels}
                setSelectedModels={(v) => dispatch({ type: 'SET_SELECTED_MODELS', payload: v })}
              />
              <ModelStatsChart selectedModelStats={selectedModelStats} />
            </div>

            <StrategySettings
              targetPct={state.targetPct}
              setTargetPct={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { targetPct: v } })}
              stopLossPct={state.stopLossPct}
              setStopLossPct={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { stopLossPct: v } })}
              lookForwardDays={state.lookForwardDays}
              setLookForwardDays={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { lookForwardDays: v } })}
              buyThreshold={state.buyThreshold}
              setBuyThreshold={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { buyThreshold: v } })}
              asOfDate={state.asOfDate}
              setAsOfDate={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { asOfDate: v } })}
              isAutoDetected={state.isAutoDetected}
              setIsAutoDetected={(v) => dispatch({ type: 'UPDATE_STRATEGY', payload: { isAutoDetected: v } })}
              useMultipleModels={state.useMultipleModels}
            />
          </section>

          {/* Symbols & Action */}
          <section className="rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-2xl">
            <ExchangeSelector
              availableExchanges={availableExchanges}
              selectedExchange={state.selectedExchange}
              setSelectedExchange={(v) => dispatch({ type: 'SET_EXCHANGE', payload: v })}
              symbols={state.symbols}
              symbolsLoading={state.symbolsLoading}
              symbolsError={state.symbolsError}
              searchSymbolTerm={state.searchSymbolTerm}
              setSearchSymbolTerm={(v) => dispatch({ type: 'SET_SEARCH_TERM', payload: v })}
              selectedSymbol={state.selectedSymbol}
              setSelectedSymbol={(v) => dispatch({ type: 'SET_SELECTED_SYMBOL', payload: v })}
            />
            <div className="mt-6">
              <button
                onClick={runTest}
                disabled={state.testLoading || !state.selectedSymbol}
                className="w-full h-12 rounded-xl bg-indigo-600 font-bold uppercase tracking-widest text-white shadow-lg shadow-indigo-600/20 hover:bg-indigo-500 disabled:opacity-50 transition-all"
              >
                {state.testLoading ? "Running Test..." : "Run Analysis"}
              </button>
              {state.testError && <div className="mt-3 text-xs text-red-400 font-bold">{state.testError}</div>}
            </div>
          </section>

          {/* Results Comparison (if multi) */}
          {state.testResults.size > 1 && (
            <section className="rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-2xl">
              <ResultsComparison
                sortedMultiSummaries={sortedMultiSummaries as any}
                multiClassificationChart={multiClassificationChart}
                toggleSort={toggleSort}
                sortConfig={state.sortConfig}
              />
            </section>
          )}

          {/* Detailed View */}
          {state.testResult && (
            <section className="rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-xl p-6 shadow-2xl">
              <DetailedResults
                testResult={state.testResult}
                predictionsWithOutcome={predictionsWithOutcome}
                showBUYSignals={state.uiSettings.showBUYSignals}
                setShowBUYSignals={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showBUYSignals', value: v } })}
                showSELLSignals={state.uiSettings.showSELLSignals}
                setShowSELLSignals={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showSELLSignals', value: v } })}
                showVolume={state.uiSettings.showVolume}
                setShowVolume={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showVolume', value: v } })}
                showSMA50={state.uiSettings.showSMA50}
                setShowSMA50={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showSMA50', value: v } })}
                showSMA200={state.uiSettings.showSMA200}
                setShowSMA200={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showSMA200', value: v } })}
                showEMA50={state.uiSettings.showEMA50}
                setShowEMA50={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showEMA50', value: v } })}
                showEMA200={state.uiSettings.showEMA200}
                setShowEMA200={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showEMA200', value: v } })}
                showBB={state.uiSettings.showBB}
                setShowBB={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showBB', value: v } })}
                showRSI={state.uiSettings.showRSI}
                setShowRSI={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showRSI', value: v } })}
                showMACD={state.uiSettings.showMACD}
                setShowMACD={(v) => dispatch({ type: 'SET_UI', payload: { key: 'showMACD', value: v } })}
                kpis={calculateKPIs(state.testResult.testPredictions || [])}
              />
            </section>
          )}

          {/* AI Brain Visualizer Integration */}
          {state.selectedModel && !state.useMultipleModels && (
            <AIBrainVisualizerSection
              selectedModel={state.selectedModel}
              targetSymbol={state.selectedSymbol?.symbol}
              lastRunTimestamp={state.lastRunTimestamp}
            />
          )}
        </div>
      </div>
    </div>
  );
}
