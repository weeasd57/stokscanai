# AI Stocks Predictor

A modern, AI-powered stock analysis and prediction platform. This application combines EOD technical analysis with Machine Learning (Random Forest) to identify high-probability trading opportunities.

![Dashboard Preview](docs/dashboard.png)

## 🚀 Key Features

### 🧠 AI Scanner
- **Market-wide Analysis**: Scans thousands of stocks across multiple exchanges (US, EGX, UK, etc.).
- **Machine Learning**: Uses Random Forest classification to predict next-day price movement.
- **Backtesting Precision**: Displays historical precision for each stock (~60-80% accuracy).
- **Customizable Models**: Advanced users can tweak Random Forest parameters (trees, depth, split).

### 📈 Technical Scanner
- **Real-time Filters**: Filter by RSI, MACD, EMA Crossovers, Bollinger Bands, and more.
- **Smart Screener**: Find stocks in "oversold" or "overbought" conditions with momentum confirmation.
- **Visual Analysis**: Built-in interactive charts (Candlestick/Area) with overlay indicators.

### ⚖️ Comparison Tool
- **Side-by-Side Analysis**: Compare multiple stocks on key performance indicators.
- **Win Rate Statistics**: See historical win rates for individual indicators (e.g., "How often does RSI < 30 lead to profit for Apple?").

### 💼 Portfolio & Watchlist
- **Track Positions**: Save interesting stocks to your watchlist.
- **Performance Tracking**: Monitor the "AI Signal" vs "Actual Price" for your saved symbols.

## 🛠️ Technology Stack

- **Frontend**: Next.js 14 (App Router), React, TailwindCSS, Lucide Icons.
- **Backend API**: Python (FastAPI), Pandas, Scikit-Learn.
- **Database**: Supabase (PostgreSQL).
- **Data Source**: EODHD API / TradingView (via custom scrapers).

## ⚡ Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- Supabase Account

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/ai-stocks.git
   cd ai-stocks
   ```

2. **Frontend Setup**
   ```bash
   cd web
   npm install
   npm run dev
   ```

3. **Backend Setup**
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

## 🌍 Supported Markets
- **US** (NYSE, NASDAQ, AMEX)
- **Egypt** (EGX)
- **UK** (LSE)
- **France** (Euronext)
- And 50+ other global exchanges.

## 🤝 Contributing
Contributions are welcome! Please create a Pull Request for any bug fixes or new features.

## 📄 License
MIT License.
