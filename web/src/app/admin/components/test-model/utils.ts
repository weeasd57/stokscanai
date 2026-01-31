import { KPIStats, ClassificationStats } from "./types";

export function parseModelExchange(modelName: string | null): string | null {
    if (!modelName) return null;
    let match = modelName.match(/model_(.+?)\.pkl/i);
    if (match) return match[1].toUpperCase();

    const upper = modelName.toUpperCase();
    const exchanges = ["EGX", "USA", "US", "KSA", "SA", "KQ", "BA", "TO", "LSE", "PA", "F"];

    for (const ex of exchanges) {
        const regex = new RegExp(`(?:^|[\\s_.,])${ex}(?:$|[\\s_.,])`, 'i');
        if (regex.test(upper)) return ex === "USA" ? "US" : ex;
    }
    return null;
}

export function calculateKPIs(predictions: any[]): KPIStats {
    if (!predictions || predictions.length === 0) {
        return {
            totalTests: 0, totalDays: 0, buySignals: 0, sellSignals: 0,
            buyCorrect: 0, sellCorrect: 0, buyAccuracy: 0, sellAccuracy: 0,
            totalCorrect: 0, totalIncorrect: 0, winRate: 0,
            consecutiveWins: 0, consecutiveLosses: 0, maxConsecutiveWins: 0, maxConsecutiveLosses: 0,
        };
    }

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

export function calculateClassification(predictions: any[]): ClassificationStats {
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

export function groupPredictionsByWeek(predictions: any[]) {
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
