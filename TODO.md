# Project Roadmap & Improvements

** move fast scanner ai section to admin . and need can select date range for scann 

** add in supabase and in ai scanner page feature be can show last scann result for show old scann and close thim if have close signal and show profit and win rate too - need tap in ai scanner page for every model should from admin page control for this feature
 


## 🚀 Priority Features (Inspired by Danelfin Analysis)

### 1. AI Score System (1-10)
- [ ] Convert model prediction probabilities into a user-friendly score from 1 to 10.
  - 90%+ Probability -> **Strong Buy (10/10)**
  - 70%-80% Probability -> **Buy (7-8/10)**
- [ ] Update frontend UI to display this score prominently for each signal.

### 2. Explainable AI (Transparency)
- [ ] Implement `feature_importance` extraction from LightGBM after each prediction.
- [ ] Display the **Top 3 Reasons** why a stock was selected (e.g., "Low RSI", "Bullish MACD Cross", "High Volume Spark").
- [ ] Add tooltips explaining what each technical indicator means for the user.


## 🔍 Competitor Analysis: Danelfin vs. Elztona

| Feature | Danelfin | Elztona (Our Tool) |
| --- | --- | --- |
| **Strategy** | Broad market coverage (1-10 score) | Sniper approach (High Precision) |
| **Win Rate** | ~60% | **81% - 95%** |
| **Key Advantage** | Sentiment & Transparency | **Surgical Accuracy** |

## ✅ Done
- [x] Initial LightGBM model implementation.
- [x] Multi-exchange data syncing (EGX, etc.).
- [x] Vercel deployment for Next.js & Python backend.
- [x] Middleware error handling and debugging instrumentation.
