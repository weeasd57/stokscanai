import type { TestPredictionRow } from "./types";

export type IndicatorStats = {
    buySignals: number;
    sellSignals: number;
    buyWinRate: string;
    sellWinRate: string;
};

export type FullIndicatorStats = {
    rsi: IndicatorStats;
    macd: IndicatorStats;
    ema: IndicatorStats;
    bb: IndicatorStats;
};

export type IndicatorSignals = {
    rsiSignal: "buy" | "sell" | "neutral";
    macdSignal: "buy" | "sell" | "neutral";
    emaSignal: "buy" | "sell" | "neutral";
    bbSignal: "buy" | "sell" | "neutral";
};

// Calculate indicator signals
export function getIndicatorSignals(row: TestPredictionRow, prevRow?: TestPredictionRow): IndicatorSignals {
    // RSI Signal: < 30 = Oversold (Buy), > 70 = Overbought (Sell)
    let rsiSignal: "buy" | "sell" | "neutral" = "neutral";
    if (row.rsi !== undefined) {
        if (row.rsi < 30) rsiSignal = "buy";
        else if (row.rsi > 70) rsiSignal = "sell";
    }

    // MACD Signal: MACD crosses above Signal = Buy, below = Sell
    let macdSignal: "buy" | "sell" | "neutral" = "neutral";
    if (row.macd !== undefined && row.macd_signal !== undefined && prevRow?.macd !== undefined && prevRow?.macd_signal !== undefined) {
        const currDiff = row.macd - row.macd_signal;
        const prevDiff = prevRow.macd - prevRow.macd_signal;
        if (currDiff > 0 && prevDiff <= 0) macdSignal = "buy";
        else if (currDiff < 0 && prevDiff >= 0) macdSignal = "sell";
    }

    // EMA Signal: Price > EMA50 > EMA200 = Buy, Price < EMA50 < EMA200 = Sell
    let emaSignal: "buy" | "sell" | "neutral" = "neutral";
    if (row.ema50 !== undefined && row.ema200 !== undefined) {
        if (row.close > row.ema50 && row.ema50 > row.ema200) emaSignal = "buy";
        else if (row.close < row.ema50 && row.ema50 < row.ema200) emaSignal = "sell";
    }

    // Bollinger Bands: Price at lower band = Buy, at upper band = Sell
    let bbSignal: "buy" | "sell" | "neutral" = "neutral";
    if (row.bb_lower !== undefined && row.bb_upper !== undefined) {
        const bbRange = row.bb_upper - row.bb_lower;
        if (bbRange > 0) {
            const position = (row.close - row.bb_lower) / bbRange;
            if (position <= 0.1) bbSignal = "buy";
            else if (position >= 0.9) bbSignal = "sell";
        }
    }

    return { rsiSignal, macdSignal, emaSignal, bbSignal };
}

// Calculate statistics for indicators
export function calculateIndicatorStats(rows: TestPredictionRow[]): FullIndicatorStats {
    const rowsWithSignals = rows.map((row, idx) => ({
        row,
        signals: getIndicatorSignals(row, rows[idx - 1])
    }));

    const calcStats = (signalKey: keyof IndicatorSignals): IndicatorStats => {
        let buySignals = 0;
        let sellSignals = 0;
        let buyWins = 0;
        let sellWins = 0;

        rowsWithSignals.forEach(({ row, signals }) => {
            const signal = signals[signalKey];
            if (signal === "buy") {
                buySignals++;
                if (row.target === 1) buyWins++;
            } else if (signal === "sell") {
                sellSignals++;
                if (row.target === 0) sellWins++;
            }
        });

        return {
            buySignals,
            sellSignals,
            buyWinRate: buySignals > 0 ? ((buyWins / buySignals) * 100).toFixed(1) : "0.0",
            sellWinRate: sellSignals > 0 ? ((sellWins / sellSignals) * 100).toFixed(1) : "0.0",
        };
    };

    return {
        rsi: calcStats("rsiSignal"),
        macd: calcStats("macdSignal"),
        ema: calcStats("emaSignal"),
        bb: calcStats("bbSignal"),
    };
}
