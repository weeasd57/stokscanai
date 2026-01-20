# Project Roadmap & Improvements


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

### 3. Scanner & Admin Improvements
- [ ] **Fast Scanner Migration**: Move the "Fast Scanner AI" section to the Admin panel.
- [ ] **Date Range Scanning**: Add functionality to select a specific date range for scanning.
- [ ] **Scan History & Performance**:
  - [ ] Store scan results in Supabase.
  - [ ] Add a feature to show old scan results and "close" them if a close signal appears.
  - [ ] Calculate and display profit/loss and win rate for previous scans.
- [ ] **Multi-Model Tabs**: Add tabs in the AI Scanner page for every model, with visibility controlled from the Admin page.


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
