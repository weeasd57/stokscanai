"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Language = "en";

interface LanguageContextType {
    language: Language;
    setLanguage: (lang: Language) => void;
    t: (key: string) => string;
    dir: "ltr";
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const translations: Record<string, Record<Language, string>> = {
    "app.title": { en: "Artoro" },
    "app.subtitle": { en: "AI-powered stock insights to help you spot opportunities with confidence" },
    "ticker.label": { en: "Ticker" },
    "ticker.placeholder": { en: "AAPL" },
    "btn.run": { en: "Run prediction" },
    "btn.running": { en: "Running..." },
    "btn.browse": { en: "Browse" },
    "browse.title": { en: "Browse Symbols" },
    "dialog.country": { en: "Country" },
    "dialog.select_country": { en: "Select a country..." },
    "dialog.search_placeholder": { en: "Search by symbol or name..." },
    "dialog.search_btn": { en: "Search" },
    "result.precision": { en: "Precision" },
    "result.window": { en: "Test window" },
    "result.last_close": { en: "Last Close" },
    "result.date": { en: "Date" },
    "result.signal": { en: "Tomorrow Signal" },
    "chart.title": { en: "{symbol} Chart Analysis" },
    "signal.up": { en: "Up (buy candidate)" },
    "signal.down": { en: "Down/flat (avoid)" },
    "nav.scanner.ai": { en: "AI Scanner" },
    "nav.scanner.tech": { en: "Technical Scanner" },
    "nav.scanner.compare": { en: "Comparison" },
    "nav.home": { en: "Home" },
    "nav.leaderboard": { en: "Leaderboard" },
    "nav.pricing": { en: "Pricing" },
    "nav.profile": { en: "Profile" },
    "auth.login": { en: "Login" },
    "auth.logout": { en: "Logout" },
    "header.pro_analysis": { en: "Pro Analysis" },
    "account.label": { en: "Account" },
    "compare.symbol": { en: "Symbol" },
    "compare.precision": { en: "AI Precision" },
    "compare.rsi": { en: "RSI Stats" },
    "compare.macd": { en: "MACD Stats" },
    "compare.ema": { en: "EMA Cross" },
    "compare.bb": { en: "Bollinger" },
    "compare.actions": { en: "Actions" },
    "compare.empty": { en: "Add symbols to compare their performance statistics." },
    "compare.fetching": { en: "Fetching historical data and calculating statistics..." },
    "compare.winrate_info": { en: "Win rates are calculated by matching indicator signals against actual next-day price movement (UP/DOWN)." },
    "tech.title": { en: "Technical Scanner" },
    "tech.subtitle": { en: "Advanced technical screener with real-time indicators." },
    "tech.config": { en: "Scanner Config" },
    "tech.market": { en: "Market" },
    "tech.rsi": { en: "RSI (14)" },
    "tech.adx": { en: "ADX (14)" },
    "tech.atr": { en: "ATR (14)" },
    "tech.stoch": { en: "Stoch %K (14)" },
    "tech.roc": { en: "ROC (12)" },
    "tech.price_above_ema50": { en: "Price > EMA 50" },
    "tech.price_above_ema200": { en: "Price > EMA 200" },
    "tech.golden_cross": { en: "Golden Cross (50 > 200)" },
    "tech.price_above_vwap20": { en: "Price > VWAP 20" },
    "tech.volume_spike": { en: "Volume Spike (> SMA20)" },
    "tech.start_scan": { en: "Start Scan" },
    "tech.stop_scan": { en: "Stop Scanning" },
    "tech.quick_search": { en: "Quick search..." },
    "tech.found_matches": { en: "Found {count} matches" },
    "tech.clear_results": { en: "Clear Results" },
    "tech.restore_last": { en: "Restore Last" },
    "tech.ready": { en: "Ready to scan. Configure filters and press Start." },
    "tech.no_matches": { en: "No stocks match your criteria." },
    "tech.table.symbol": { en: "Symbol" },
    "tech.table.price": { en: "Price" },
    "tech.table.momentum": { en: "Momentum" },
    "tech.table.save": { en: "Save" },
    "pagination.page": { en: "Page" },
    "dash.title": { en: "Market Strategy Dashboard" },
    "dash.subtitle": { en: "Aggregate success rates for key technical indicators across {country} market listings." },
    "dash.winrate": { en: "Avg Win Rate" },
    "dash.signals": { en: "Total Signals" },
    "dash.scanned": { en: "Scanned {count} tickers" },
    "dash.refresh": { en: "Refreshed {time} ago" },
    "dash.filter_market": { en: "Filter Market" },
    "ai.title": { en: "AI Market Scanner" },
    "ai.subtitle": { en: "Scans the market using the Random Forest model to find high-probability BUY signals." },
    "ai.scan_all": { en: "Scan All Market" },
    "ai.start_scan": { en: "Start AI Scan" },
    "ai.stop_scan": { en: "Stop Analysis" },
    "ai.model_preset": { en: "Model Preset" },
    "ai.model_options": { en: "Model Options" },
    "ai.precision_info": { en: "Precision means: among all signals produced in backtest, how many were correct. High precision usually indicates higher quality." },
    "ai.matches": { en: "Matches" },
    "ai.table.symbol": { en: "Symbol" },
    "ai.table.name": { en: "Name" },
    "ai.table.price": { en: "Last Price" },
    "ai.table.precision": { en: "AI Precision" },
    "ai.table.save": { en: "Save" },
    "ai.no_matches": { en: "No high-confidence opportunities found in this batch." },
    "ai.chart_ctrl": { en: "Chart View" },
    "ai.indicators": { en: "Indicators" },
    "scanner.templates.kicker": { en: "One-Click Strategies" },
    "scanner.templates.title": { en: "Scanner Templates" },
    "scanner.templates.subtitle": { en: "Pick a strategy to launch a focused scan instantly." },
    "scanner.templates.ai_growth.title": { en: "AI Smart Pick" },
    "scanner.templates.ai_growth.desc": { en: "AI expects upside based on a Random Forest model trained on two years of data." },
    "scanner.templates.macd_cross.title": { en: "MACD Golden Cross" },
    "scanner.templates.macd_cross.desc": { en: "Classic entry when MACD crosses its signal line to the upside." },
    "scanner.templates.rsi_oversold.title": { en: "RSI Oversold" },
    "scanner.templates.rsi_oversold.desc": { en: "Find tickers under RSI 30 that may be ready to rebound." },
    "scanner.templates.volume_breakout.title": { en: "Volume Breakout" },
    "scanner.templates.volume_breakout.desc": { en: "Detect unusual volume spikes signaling institutional interest." },
    "scanner.templates.sma_200_breakout.title": { en: "Trend Breakout" },
    "scanner.templates.sma_200_breakout.desc": { en: "Price breaks above the 200-day SMA to confirm a long-term uptrend." },
    "scanner.templates.risk.low": { en: "Low Risk" },
    "scanner.templates.risk.medium": { en: "Medium Risk" },
    "scanner.templates.risk.high": { en: "High Risk" },
    "scanner.templates.risk.very_high": { en: "Very High Risk" },
    "profile.track": { en: "Track positions, targets, stop-loss and performance." },
    "profile.stats.open": { en: "Open" },
    "profile.stats.wins": { en: "Wins" },
    "profile.stats.losses": { en: "Losses" },
    "profile.stats.winrate": { en: "Win Rate" },
    "profile.defaults.title": { en: "Trading Defaults" },
    "profile.defaults.subtitle": { en: "Used when saving a new symbol to your watchlist." },
    "profile.defaults.target": { en: "Default Target %" },
    "profile.defaults.stop": { en: "Default Stop-Loss %" },
    "profile.ai.title": { en: "AI Assistant Settings" },
    "profile.ai.subtitle": { en: "Configure API keys for the smart chat assistant." },
    "profile.ai.gemini": { en: "Gemini API Key" },
    "profile.ai.rules": { en: "Custom Rules / Instructions" },
    "profile.positions.title": { en: "Trading Positions" },
    "profile.positions.subtitle": { en: "Manage your open and closed positions. Evaluation updates status to Win/Loss." },
    "profile.positions.evaluate": { en: "Evaluate Open Positions" },
    "profile.table.symbol": { en: "Symbol" },
    "profile.table.added": { en: "Added" },
    "profile.table.status": { en: "Status" },
    "profile.table.entry": { en: "Entry" },
    "profile.table.target": { en: "Target" },
    "profile.table.stop": { en: "Stop" },
    "profile.table.actions": { en: "Actions" },
    "home.insights.title": { en: "Smart AI Insights" },
    "home.insights.subtitle": { en: "Combined Fundamental & Technical Analysis" },
    "home.insights.valuation": { en: "Valuation" },
    "home.insights.volatility": { en: "Volatility Risk" },
    "home.insights.mcap": { en: "Market Cap" },
    "home.insights.undervalued": { en: "Undervalued. Good value stock candidate." },
    "home.insights.fair": { en: "Fair Valuation. Market standard." },
    "home.insights.expensive": { en: "Growth/Expensive. Expect volatility." },
    "home.insights.bubble": { en: "Very Expensive (Bubble risk?). AI is cautious." },
    "home.insights.low_vol": { en: "Low Volatility. Defensive / Safe haven." },
    "home.insights.avg_vol": { en: "Average Market Risk." },
    "home.insights.high_vol": { en: "High Volatility. High risk/reward." },
    "home.insights.large_cap": { en: "Large Cap. Stable & Established." },
    "home.insights.mid_cap": { en: "Mid Cap. Balanced Growth." },
    "home.insights.small_cap": { en: "Small Cap. Higher growth potential but risky." },
    "home.insights.strong_buy": { en: "Strong Buy Signal: Technical Uptrend + Reasonable Valuation." },
    "home.insights.cautious_buy": { en: "Cautious Buy: Technical Uptrend, but Valuation is expensive." },
    "home.chart.title": { en: "Price & AI Buy Signals" },
    "home.chart.subtitle": { en: "Green dots are model buy predictions" },
    "home.fundamentals.title": { en: "Fundamentals" },
    "home.footer.disclaimer": { en: "Model output is not financial advice." },
    "dash.days.30": { en: "Last 30 Days" },
    "dash.days.60": { en: "Last 60 Days" },
    "dash.days.90": { en: "Last 90 Days" },
    "compare.signal": { en: "Tomorrow Signal" },
    "compare.save": { en: "Save" },
    "compare.saved": { en: "Saved" },
    "compare.chart": { en: "Chart" },
    "chart.close": { en: "Close" },
};

export function LanguageProvider({ children }: { children: ReactNode }) {
    const [language] = useState<Language>("en");

    // Force English and LTR direction
    useEffect(() => {
        localStorage.setItem("app-language", "en");
        document.documentElement.dir = "ltr";
        document.documentElement.lang = "en";
    }, []);

    const t = (key: string) => {
        const entry = translations[key];
        if (!entry) return key;
        return entry["en"];
    };

    return (
        <LanguageContext.Provider
            value={{
                language,
                setLanguage: () => { }, // No-op as we only support English
                t,
                dir: "ltr",
            }}
        >
            {children}
        </LanguageContext.Provider>
    );
}

export function useLanguage() {
    const context = useContext(LanguageContext);
    if (!context) {
        throw new Error("useLanguage must be used within a LanguageProvider");
    }
    return context;
}
