"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Language = "en" | "ar";

interface LanguageContextType {
    language: Language;
    setLanguage: (lang: Language) => void;
    t: (key: string) => string;
    dir: "ltr" | "rtl";
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const translations: Record<string, Record<Language, string>> = {
    "app.title": {
        en: "Artoro",
        ar: "Artoro",
    },
    "app.subtitle": {
        en: "AI-powered stock insights to help you spot opportunities with confidence",
        ar: "رؤى أسهم ذكية بالذكاء الاصطناعي لمساعدتك على اكتشاف الفرص بثقة",
    },
    "ticker.label": {
        en: "Ticker",
        ar: "الرمز",
    },
    "ticker.placeholder": {
        en: "AAPL",
        ar: "أدخل الرمز",
    },
    "btn.run": {
        en: "Run prediction",
        ar: "شغل التوقع",
    },
    "btn.running": {
        en: "Running...",
        ar: "جاري العمل...",
    },
    "btn.browse": {
        en: "Browse",
        ar: "تصفح",
    },
    "browse.title": {
        en: "Browse Symbols",
        ar: "تصفح الرموز",
    },
    "dialog.country": {
        en: "Country",
        ar: "الدولة",
    },
    "dialog.select_country": {
        en: "Select a country...",
        ar: "اختر دولة...",
    },
    "dialog.search_placeholder": {
        en: "Search by symbol or name...",
        ar: "ابحث بالرمز أو الاسم...",
    },
    "dialog.search_btn": {
        en: "Search",
        ar: "بحث",
    },
    "result.precision": {
        en: "Precision",
        ar: "الدقة",
    },
    "result.window": {
        en: "Test window",
        ar: "نافذة الاختبار",
    },
    "result.last_close": {
        en: "Last Close",
        ar: "آخر إغلاق",
    },
    "result.date": {
        en: "Date",
        ar: "التاريخ",
    },
    "result.signal": {
        en: "Tomorrow Signal",
        ar: "إشارة الغد",
    },
    "chart.title": {
        en: "{symbol} Chart Analysis",
        ar: "تحليل الرسم البياني {symbol}",
    },
    "signal.up": {
        en: "Up (buy candidate)",
        ar: "صعود (مرشح للشراء)",
    },
    "signal.down": {
        en: "Down/flat (avoid)",
        ar: "هبوط/ثبات (تجنب)",
    },
    "nav.scanner.ai": {
        en: "AI Scanner",
        ar: "ماسح الذكاء الاصطناعي",
    },
    "nav.scanner.tech": {
        en: "Technical Scanner",
        ar: "الماسح الفني",
    },
    "nav.scanner.compare": {
        en: "Comparison",
        ar: "مقارنة الأسهم",
    },
    "nav.home": {
        en: "Home",
        ar: "الرئيسية",
    },
    "nav.leaderboard": {
        en: "Leaderboard",
        ar: "لوحة المتصدرين",
    },
    "nav.pricing": {
        en: "Pricing",
        ar: "الأسعار",
    },
    "nav.profile": {
        en: "Profile",
        ar: "الملف الشخصي",
    },
    "auth.login": {
        en: "Login",
        ar: "تسجيل الدخول",
    },
    "auth.logout": {
        en: "Logout",
        ar: "تسجيل الخروج",
    },
    "header.pro_analysis": {
        en: "Pro Analysis",
        ar: "تحليل احترافي",
    },
    "account.label": {
        en: "Account",
        ar: "الحساب",
    },
    "compare.symbol": { en: "Symbol", ar: "الرمز" },
    "compare.precision": { en: "AI Precision", ar: "دقة الذكاء الاصطناعي" },
    "compare.rsi": { en: "RSI Stats", ar: "إحصائيات RSI" },
    "compare.macd": { en: "MACD Stats", ar: "إحصائيات MACD" },
    "compare.ema": { en: "EMA Cross", ar: "تقاطع EMA" },
    "compare.bb": { en: "Bollinger", ar: "البولنجر" },
    "compare.actions": { en: "Actions", ar: "إجراءات" },
    "compare.empty": { en: "Add symbols to compare their performance statistics.", ar: "أضف رموزًا لمقارنة إحصائيات الأداء الخاصة بها." },
    "compare.fetching": { en: "Fetching historical data and calculating statistics...", ar: "جاري جلب البيانات التاريخية وحساب الإحصائيات..." },
    "compare.winrate_info": { en: "Win rates are calculated by matching indicator signals against actual next-day price movement (UP/DOWN).", ar: "يتم حساب معدلات الفوز من خلال مطابقة إشارات المؤشر مع حركة السعر الفعلية لليوم التالي (صعود/هبوط)." },
    "tech.title": { en: "Technical Scanner", ar: "الماسح الفني" },
    "tech.subtitle": { en: "Advanced technical screener with real-time indicators.", ar: "ماسح فني متقدم مع مؤشرات لحظية." },
    "tech.config": { en: "Scanner Config", ar: "إعدادات الماسح" },
    "tech.market": { en: "Market", ar: "السوق" },
    "tech.rsi": { en: "RSI (14)", ar: "مؤشر RSI" },
    "tech.adx": { en: "ADX (14)", ar: "مؤشر ADX" },
    "tech.atr": { en: "ATR (14)", ar: "مؤشر ATR" },
    "tech.stoch": { en: "Stoch %K (14)", ar: "مؤشر Stoch" },
    "tech.roc": { en: "ROC (12)", ar: "مؤشر ROC" },
    "tech.price_above_ema50": { en: "Price > EMA 50", ar: "السعر > EMA 50" },
    "tech.price_above_ema200": { en: "Price > EMA 200", ar: "السعر > EMA 200" },
    "tech.golden_cross": { en: "Golden Cross (50 > 200)", ar: "التقاطع الذهبي (50 > 200)" },
    "tech.price_above_vwap20": { en: "Price > VWAP 20", ar: "السعر > VWAP 20" },
    "tech.volume_spike": { en: "Volume Spike (> SMA20)", ar: "طفرة في حجم التداول (> SMA20)" },
    "tech.start_scan": { en: "Start Scan", ar: "بدء الفحص" },
    "tech.stop_scan": { en: "Stop Scanning", ar: "إيقاف الفحص" },
    "tech.quick_search": { en: "Quick search...", ar: "بحث سريع..." },
    "tech.found_matches": { en: "Found {count} matches", ar: "تم العثور على {count} تطابقاً" },
    "tech.clear_results": { en: "Clear Results", ar: "مسح النتائج" },
    "tech.restore_last": { en: "Restore Last", ar: "استعادة الأخير" },
    "tech.ready": { en: "Ready to scan. Configure filters and press Start.", ar: "جاهز للفحص. قم بضبط الفلاتر واضغط على بدء." },
    "tech.no_matches": { en: "No stocks match your criteria.", ar: "لا توجد أسهم تطابق معاييرك." },
    "tech.table.symbol": { en: "Symbol", ar: "الرمز" },
    "tech.table.price": { en: "Price", ar: "السعر" },
    "tech.table.momentum": { en: "Momentum", ar: "الزخم" },
    "tech.table.save": { en: "Save", ar: "حفظ" },
    "pagination.page": { en: "Page", ar: "صفحة" },
    "dash.title": { en: "Market Strategy Dashboard", ar: "لوحة تحكم استراتيجيات السوق" },
    "dash.subtitle": { en: "Aggregate success rates for key technical indicators across {country} market listings.", ar: "معدلات النجاح المجمعة للمؤشرات الفنية الرئيسية عبر مدرجات سوق {country}." },
    "dash.winrate": { en: "Avg Win Rate", ar: "متوسط معدل الفوز" },
    "dash.signals": { en: "Total Signals", ar: "إجمالي الإشارات" },
    "dash.scanned": { en: "Scanned {count} tickers", ar: "تم فحص {count} رمزاً" },
    "dash.refresh": { en: "Refreshed {time} ago", ar: "تم التحديث منذ {time}" },
    "dash.filter_market": { en: "Filter Market", ar: "تصفية السوق" },
    "ai.title": { en: "AI Market Scanner", ar: "ماسح السوق بالذكاء الاصطناعي" },
    "ai.subtitle": { en: "Scans the market using the Random Forest model to find high-probability BUY signals.", ar: "يفحص السوق باستخدام نموذج الغابة العشوائية (Random Forest) للعثور على إشارات شراء عالية الاحتمالية." },
    "ai.scan_all": { en: "Scan All Market", ar: "مسح السوق بالكامل" },
    "ai.start_scan": { en: "Start AI Scan", ar: "بدء فحص الذكاء الاصطناعي" },
    "ai.stop_scan": { en: "Stop Analysis", ar: "إيقاف التحليل" },
    "ai.model_preset": { en: "Model Preset", ar: "الإعداد المسبق للنموذج" },
    "ai.model_options": { en: "Model Options", ar: "خيارات النموذج" },
    "ai.precision_info": { en: "Precision means: among all signals produced in backtest, how many were correct. High precision usually indicates higher quality.", ar: "تعني الدقة: من بين جميع الإشارات المنتجة في الاختبار الخلفي، كم عدد الإشارات الصحيحة. تشير الدقة العالية عادةً إلى جودة أعلى." },
    "ai.matches": { en: "Matches", ar: "التطابقات" },
    "ai.table.symbol": { en: "Symbol", ar: "الرمز" },
    "ai.table.name": { en: "Name", ar: "الاسم" },
    "ai.table.price": { en: "Last Price", ar: "آخر سعر" },
    "ai.table.precision": { en: "AI Precision", ar: "دقة الذكاء الاصطناعي" },
    "ai.table.save": { en: "Save", ar: "حفظ" },
    "ai.no_matches": { en: "No high-confidence opportunities found in this batch.", ar: "لم يتم العثور على فرص عالية الثقة في هذه الدفعة." },
    "ai.chart_ctrl": { en: "Chart View", ar: "عرض الرسم البياني" },
    "ai.indicators": { en: "Indicators", ar: "المؤشرات" },
    "scanner.templates.kicker": { en: "One-Click Strategies", ar: "One-Click Strategies" },
    "scanner.templates.title": { en: "Scanner Templates", ar: "Scanner Templates" },
    "scanner.templates.subtitle": { en: "Pick a strategy to launch a focused scan instantly.", ar: "Pick a strategy to launch a focused scan instantly." },
    "scanner.templates.ai_growth.title": { en: "AI Smart Pick", ar: "AI Smart Pick" },
    "scanner.templates.ai_growth.desc": { en: "AI expects upside based on a Random Forest model trained on two years of data.", ar: "AI expects upside based on a Random Forest model trained on two years of data." },
    "scanner.templates.macd_cross.title": { en: "MACD Golden Cross", ar: "MACD Golden Cross" },
    "scanner.templates.macd_cross.desc": { en: "Classic entry when MACD crosses its signal line to the upside.", ar: "Classic entry when MACD crosses its signal line to the upside." },
    "scanner.templates.rsi_oversold.title": { en: "RSI Oversold", ar: "RSI Oversold" },
    "scanner.templates.rsi_oversold.desc": { en: "Find tickers under RSI 30 that may be ready to rebound.", ar: "Find tickers under RSI 30 that may be ready to rebound." },
    "scanner.templates.volume_breakout.title": { en: "Volume Breakout", ar: "Volume Breakout" },
    "scanner.templates.volume_breakout.desc": { en: "Detect unusual volume spikes signaling institutional interest.", ar: "Detect unusual volume spikes signaling institutional interest." },
    "scanner.templates.sma_200_breakout.title": { en: "Trend Breakout", ar: "Trend Breakout" },
    "scanner.templates.sma_200_breakout.desc": { en: "Price breaks above the 200-day SMA to confirm a long-term uptrend.", ar: "Price breaks above the 200-day SMA to confirm a long-term uptrend." },
    "scanner.templates.risk.low": { en: "Low Risk", ar: "Low Risk" },
    "scanner.templates.risk.medium": { en: "Medium Risk", ar: "Medium Risk" },
    "scanner.templates.risk.high": { en: "High Risk", ar: "High Risk" },
    "scanner.templates.risk.very_high": { en: "Very High Risk", ar: "Very High Risk" },
    "profile.track": { en: "Track positions, targets, stop-loss and performance.", ar: "تتبع المراكز والأهداف ووقف الخسارة والأداء." },
    "profile.stats.open": { en: "Open", ar: "مفتوحة" },
    "profile.stats.wins": { en: "Wins", ar: "أرباح" },
    "profile.stats.losses": { en: "Losses", ar: "خسائر" },
    "profile.stats.winrate": { en: "Win Rate", ar: "معدل الربح" },
    "profile.defaults.title": { en: "Trading Defaults", ar: "إعدادات التداول الافتراضية" },
    "profile.defaults.subtitle": { en: "Used when saving a new symbol to your watchlist.", ar: "تُستخدم عند حفظ رمز جديد في قائمة المراقبة الخاصة بك." },
    "profile.defaults.target": { en: "Default Target %", ar: "الهدف الافتراضي %" },
    "profile.defaults.stop": { en: "Default Stop-Loss %", ar: "وقف الخسارة الافتراضي %" },
    "profile.ai.title": { en: "AI Assistant Settings", ar: "إعدادات مساعد الذكاء الاصطناعي" },
    "profile.ai.subtitle": { en: "Configure API keys for the smart chat assistant.", ar: "تكوين مفاتيح API للمساعد الذكي." },
    "profile.ai.gemini": { en: "Gemini API Key", ar: "مفتاح Gemini API" },
    "profile.ai.rules": { en: "Custom Rules / Instructions", ar: "القواعد / التعليمات المخصصة" },
    "profile.positions.title": { en: "Trading Positions", ar: "مراكز التداول" },
    "profile.positions.subtitle": { en: "Manage your open and closed positions. Evaluation updates status to Win/Loss.", ar: "إدارة مراكزك المفتوحة والمغلقة. يقوم التقييم بتحديث الحالة إلى ربح/خسارة." },
    "profile.positions.evaluate": { en: "Evaluate Open Positions", ar: "تقييم المراكز المفتوحة" },
    "profile.table.symbol": { en: "Symbol", ar: "الرمز" },
    "profile.table.added": { en: "Added", ar: "أضيف في" },
    "profile.table.status": { en: "Status", ar: "الحالة" },
    "profile.table.entry": { en: "Entry", ar: "الدخول" },
    "profile.table.target": { en: "Target", ar: "الهدف" },
    "profile.table.stop": { en: "Stop", ar: "الوقف" },
    "profile.table.actions": { en: "Actions", ar: "إجراءات" },
    "home.insights.title": { en: "Smart AI Insights", ar: "رؤى الذكاء الاصطناعي الذكية" },
    "home.insights.subtitle": { en: "Combined Fundamental & Technical Analysis", ar: "تحليل أساسي وفني مشترك" },
    "home.insights.valuation": { en: "Valuation", ar: "التقييم" },
    "home.insights.volatility": { en: "Volatility Risk", ar: "مخاطر التقلب" },
    "home.insights.mcap": { en: "Market Cap", ar: "القيمة السوقية" },
    "home.insights.undervalued": { en: "Undervalued. Good value stock candidate.", ar: "أقل من قيمته الحقيقية. مرشح جيد كأداة استثمارية ذات قيمة." },
    "home.insights.fair": { en: "Fair Valuation. Market standard.", ar: "تقييم عادل. معيار السوق." },
    "home.insights.expensive": { en: "Growth/Expensive. Expect volatility.", ar: "نمو/مرتفع الثمن. توقع تقلبات." },
    "home.insights.bubble": { en: "Very Expensive (Bubble risk?). AI is cautious.", ar: "غالٍ جداً (خطر الفقاعة؟). الذكاء الاصطناعي حذر." },
    "home.insights.low_vol": { en: "Low Volatility. Defensive / Safe haven.", ar: "تقلب منخفض. دفاعي / ملاذ آمن." },
    "home.insights.avg_vol": { en: "Average Market Risk.", ar: "مخاطر السوق المتوسطة." },
    "home.insights.high_vol": { en: "High Volatility. High risk/reward.", ar: "تقلب عالي. مخاطر/عوائد عالية." },
    "home.insights.large_cap": { en: "Large Cap. Stable & Established.", ar: "شركة كبيرة. مستقرة وراسخة." },
    "home.insights.mid_cap": { en: "Mid Cap. Balanced Growth.", ar: "شركة متوسطة. نمو متوازن." },
    "home.insights.small_cap": { en: "Small Cap. Higher growth potential but risky.", ar: "شركة صغيرة. إمكانات نمو أعلى ولكنها محفوفة بالمخاطر." },
    "home.insights.strong_buy": { en: "Strong Buy Signal: Technical Uptrend + Reasonable Valuation.", ar: "إشارة شراء قوية: اتجاه صعودي فني + تقييم معقول." },
    "home.insights.cautious_buy": { en: "Cautious Buy: Technical Uptrend, but Valuation is expensive.", ar: "شراء حذر: اتجاه صعودي فني، ولكن التقييم مرتفع." },
    "home.chart.title": { en: "Price & AI Buy Signals", ar: "السعر وإشارات شراء الذكاء الاصطناعي" },
    "home.chart.subtitle": { en: "Green dots are model buy predictions", ar: "النقاط الخضراء هي توقعات الشراء من النموذج" },
    "home.fundamentals.title": { en: "Fundamentals", ar: "البيانات الأساسية" },
    "home.footer.disclaimer": { en: "Model output is not financial advice.", ar: "مخرجات النموذج ليست نصيحة مالية." },
    "dash.days.30": { en: "Last 30 Days", ar: "آخر 30 يوم" },
    "dash.days.60": { en: "Last 60 Days", ar: "آخر 60 يوم" },
    "dash.days.90": { en: "Last 90 Days", ar: "آخر 90 يوم" },
    // New Translations
    "compare.signal": { en: "Tomorrow Signal", ar: "إشارة الغد" },
    "compare.save": { en: "Save", ar: "حفظ" },
    "compare.saved": { en: "Saved", ar: "تم الحفظ" },
    "compare.chart": { en: "Chart", ar: "رسم بياني" },
    "chart.close": { en: "Close", ar: "إغلاق" },
};

export function LanguageProvider({ children }: { children: ReactNode }) {
    const [language, setLanguage] = useState<Language>("en");

    // Load saved language
    useEffect(() => {
        const saved = localStorage.getItem("app-language") as Language;
        if (saved) setLanguage(saved);
    }, []);

    // Save language and update document direction
    useEffect(() => {
        localStorage.setItem("app-language", language);
        document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
        document.documentElement.lang = language;
    }, [language]);

    const t = (key: string) => {
        const entry = translations[key];
        if (!entry) return key;
        return entry[language];
    };

    return (
        <LanguageContext.Provider
            value={{
                language,
                setLanguage,
                t,
                dir: language === "ar" ? "rtl" : "ltr",
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
