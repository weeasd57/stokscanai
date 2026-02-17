# Artoro

Advanced AI-driven stock analysis platform. Combining RandomForest models with multi-source fundamentals to give you the edge.

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

## 🚀 Deployment Architecture

This project uses a **Split Deployment Strategy** to optimize performance and overcome platform limitations:

### 1. Frontend (Vercel)
- **Hosted on**: [Vercel](https://vercel.com)
- **Reason**: Best-in-class performance for Next.js, Edge Network, and SEO.
- **Configuration**: Uses standard Vercel auto-detection for Next.js in the `web/` directory.

### 2. Backend (Railway)
- **Hosted on**: [Railway](https://railway.app)
- **Method**: Docker Container (via `api/Dockerfile`)
- **Reason**: The Python backend requires significant RAM/Disk for ML models (`scikit-learn`, `pandas`) which exceeds Vercel's Serverless Function limits (250MB). Railway provides a persistent environment perfect for heavy AI workloads.
- **Docker Config**: Explicitly uses Python 3.12 and installs dependencies from `api/requirements.txt`.

## 🌍 Supported Markets
- **US** (NYSE, NASDAQ, AMEX)
- **Egypt** (EGX)
- **UK** (LSE)
- **France** (Euronext)
- And 50+ other global exchanges.

## 🏛️ Council Validator (Meta-Model)

This repo also supports training a lightweight validator that learns when a base model’s BUY candidates tend to fail, then blocks those trades.

- Train: `py train_council.py --primary-model "api/models/KING 👑.pkl"`
- Use in backtest: `py api/backtest_radar.py --exchange EGX --model "collector 🎁.pkl" --validator "The_Council_Validator.pkl"`


## � Deployment (Hugging Face Spaces)

The backend is deployed as a Docker container on Hugging Face Spaces. The `api/` folder is the single source of truth.

### How to Deploy Updates

1. **Navigate to the API folder**:
   ```powershell
   cd "C:\Users\MR_CODER\Desktop\AI stocks\api"
   ```

2. **Commit and Push**:
   ```powershell
   git add .
   git commit -m "Update API features"
   git push hf master:main --force
   ```

> [!NOTE]
> The `AI_BOT` folder was removed to avoid duplication. Always work within the root `api/` directory.


## 🤝 Contributing
Contributions are welcome! Please create a Pull Request for any bug fixes or new features.

## 📄 License
MIT License.
الإعداد	🛡️ الدفاعي (Defensive)	🔄 الهجين (Hybrid)	⚔️ الهجومي (Aggressive)
صعوبة دخول KING	أصعب بـ +0.10	طبيعي (0.0)	أسهل بـ -0.10
صعوبة دخول COUNCIL	أصعب بـ +0.05	طبيعي (0.0)	أسهل بـ -0.05
فلتر السيولة (Volume)	صارم جداً (1.5x)	مرن (0.7x)	مرن جداً (0.3x)
فلتر الاتجاه (SMA20)	مفعل دائماً	مفعل	معطل (يخترق الترند)
أدنى جودة للإشارة	75% فأعلى	55% فأعلى	50% فأعلى
حجم الصفقة (Sideways)	مُخفض جداً (30%)	مُخفض (70%)	كامل (100%)
حجم الصفقة (BEAR)	صفر (لا يتداول)	مُخفض جداً (30%)	نصف حجم (50%)