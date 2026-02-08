#!/usr/bin/env python3
"""
Performance Analysis Script للبوت المتقدم
يحلل الأداء ويولد تقارير مفصلة
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List


def load_json(path: str) -> dict:
    """تحميل ملف JSON"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def analyze_trades(trades_path: str = "logs/trades.json") -> Dict:
    """تحليل الصفقات"""
    data = load_json(trades_path)
    trades = data.get("trades", [])
    
    if not trades:
        return {"error": "No trades found"}
    
    # فصل صفقات الشراء والبيع
    buys = [t for t in trades if t.get("action") == "BUY"]
    sells = [t for t in trades if t.get("action") == "SELL"]
    
    # حساب الإحصائيات
    total_trades = len(buys)
    completed_trades = len(sells)
    
    wins = [t for t in sells if t.get("pnl", 0) > 0]
    losses = [t for t in sells if t.get("pnl", 0) <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / completed_trades * 100) if completed_trades > 0 else 0
    
    total_pnl = sum(t.get("pnl", 0) for t in sells)
    avg_win = sum(t.get("pnl", 0) for t in wins) / win_count if wins else 0
    avg_loss = sum(t.get("pnl", 0) for t in losses) / loss_count if losses else 0
    
    # أفضل وأسوأ صفقة
    best_trade = max(sells, key=lambda x: x.get("pnl", 0)) if sells else None
    worst_trade = min(sells, key=lambda x: x.get("pnl", 0)) if sells else None
    
    # أسباب الخروج
    exit_reasons = {}
    for trade in sells:
        reason = trade.get("reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # الصفقات حسب العملة
    symbols = {}
    for trade in sells:
        symbol = trade.get("symbol", "Unknown")
        if symbol not in symbols:
            symbols[symbol] = {"count": 0, "pnl": 0, "wins": 0}
        symbols[symbol]["count"] += 1
        symbols[symbol]["pnl"] += trade.get("pnl", 0)
        if trade.get("pnl", 0) > 0:
            symbols[symbol]["wins"] += 1
    
    return {
        "total_trades": total_trades,
        "completed_trades": completed_trades,
        "open_positions": total_trades - completed_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "exit_reasons": exit_reasons,
        "symbols": symbols,
    }


def analyze_daily_performance(perf_path: str = "logs/performance.json") -> Dict:
    """تحليل الأداء اليومي"""
    data = load_json(perf_path)
    
    if not data or "date" not in data:
        return {"error": "No daily performance data"}
    
    starting = data.get("starting_balance", 0)
    current = data.get("current_balance", 0)
    
    daily_return = ((current - starting) / starting * 100) if starting > 0 else 0
    
    return {
        "date": data.get("date"),
        "trades_count": data.get("trades_count", 0),
        "wins": data.get("wins", 0),
        "losses": data.get("losses", 0),
        "total_pnl": data.get("total_pnl", 0),
        "starting_balance": starting,
        "current_balance": current,
        "daily_return": daily_return,
        "max_drawdown": data.get("max_drawdown", 0),
    }


def analyze_alerts(alerts_path: str = "logs/alerts.json") -> Dict:
    """تحليل التنبيهات"""
    data = load_json(alerts_path)
    alerts = data.get("alerts", [])
    
    # آخر 10 تنبيهات
    recent = alerts[-10:] if len(alerts) > 10 else alerts
    
    # عدد التنبيهات حسب النوع
    by_type = {}
    for alert in alerts:
        alert_type = alert.get("type", "Unknown")
        by_type[alert_type] = by_type.get(alert_type, 0) + 1
    
    return {
        "total_alerts": len(alerts),
        "recent_alerts": recent,
        "by_type": by_type,
    }


def print_section(title: str):
    """طباعة عنوان القسم"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")


def print_report():
    """طباعة التقرير الكامل"""
    print("\n" + "="*70)
    print("TRADING BOT PERFORMANCE REPORT".center(70))
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70))
    print("="*70)
    
    # 1. تحليل الصفقات
    print_section("📊 TRADES ANALYSIS")
    trades_analysis = analyze_trades()
    
    if "error" in trades_analysis:
        print(f"  ⚠️  {trades_analysis['error']}")
    else:
        print(f"  Total Trades:        {trades_analysis['total_trades']}")
        print(f"  Completed:           {trades_analysis['completed_trades']}")
        print(f"  Open Positions:      {trades_analysis['open_positions']}")
        print(f"  Wins:                {trades_analysis['wins']} ({trades_analysis['win_rate']:.1f}%)")
        print(f"  Losses:              {trades_analysis['losses']}")
        print(f"  Total P&L:           ${trades_analysis['total_pnl']:,.2f}")
        print(f"  Avg Win:             ${trades_analysis['avg_win']:,.2f}")
        print(f"  Avg Loss:            ${trades_analysis['avg_loss']:,.2f}")
        print(f"  Profit Factor:       {trades_analysis['profit_factor']:.2f}")
        
        if trades_analysis['best_trade']:
            best = trades_analysis['best_trade']
            print(f"\n  Best Trade:          {best['symbol']} - ${best['pnl']:,.2f}")
        
        if trades_analysis['worst_trade']:
            worst = trades_analysis['worst_trade']
            print(f"  Worst Trade:         {worst['symbol']} - ${worst['pnl']:,.2f}")
        
        print("\n  Exit Reasons:")
        for reason, count in trades_analysis['exit_reasons'].items():
            print(f"    {reason:20s} {count:3d}")
        
        print("\n  Performance by Symbol:")
        for symbol, stats in sorted(trades_analysis['symbols'].items(), 
                                    key=lambda x: x[1]['pnl'], reverse=True):
            wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"    {symbol:10s} Trades: {stats['count']:2d} | "
                  f"Win Rate: {wr:5.1f}% | P&L: ${stats['pnl']:+8.2f}")
    
    # 2. الأداء اليومي
    print_section("📈 DAILY PERFORMANCE")
    daily = analyze_daily_performance()
    
    if "error" in daily:
        print(f"  ⚠️  {daily['error']}")
    else:
        print(f"  Date:                {daily['date']}")
        print(f"  Trades Today:        {daily['trades_count']}")
        print(f"  Wins/Losses:         {daily['wins']}/{daily['losses']}")
        print(f"  Starting Balance:    ${daily['starting_balance']:,.2f}")
        print(f"  Current Balance:     ${daily['current_balance']:,.2f}")
        print(f"  Daily Return:        {daily['daily_return']:+.2f}%")
        print(f"  Total P&L:           ${daily['total_pnl']:+,.2f}")
        print(f"  Max Drawdown:        ${daily['max_drawdown']:,.2f}")
    
    # 3. التنبيهات
    print_section("🔔 ALERTS")
    alerts = analyze_alerts()
    
    print(f"  Total Alerts:        {alerts['total_alerts']}")
    
    if alerts['by_type']:
        print("\n  Alerts by Type:")
        for alert_type, count in alerts['by_type'].items():
            print(f"    {alert_type:20s} {count:3d}")
    
    if alerts['recent_alerts']:
        print("\n  Recent Alerts:")
        for alert in alerts['recent_alerts'][-5:]:
            timestamp = alert.get('timestamp', 'N/A')[:19]
            alert_type = alert.get('type', 'Unknown')
            message = alert.get('message', '')[:50]
            print(f"    [{timestamp}] {alert_type}: {message}")
    
    # 4. توصيات
    print_section("💡 RECOMMENDATIONS")
    
    if "error" not in trades_analysis:
        if trades_analysis['win_rate'] < 40:
            print("  ⚠️  Low win rate - Consider:")
            print("      • Increasing KING_THRESHOLD")
            print("      • Increasing COUNCIL_THRESHOLD")
            print("      • Enabling more filters")
        
        if trades_analysis['profit_factor'] < 1.5:
            print("  ⚠️  Low profit factor - Consider:")
            print("      • Tightening stop loss")
            print("      • Increasing take profit target")
            print("      • Using trailing stops")
        
        if trades_analysis['completed_trades'] < 5:
            print("  ℹ️  Low trading activity - Consider:")
            print("      • Decreasing thresholds")
            print("      • Adding more trading pairs")
            print("      • Decreasing poll interval")
    
    if "error" not in daily:
        if daily['daily_return'] < -5:
            print("  🚨 High daily loss - Consider:")
            print("      • Stopping trading for today")
            print("      • Reviewing your strategy")
            print("      • Checking market conditions")
    
    print("\n" + "="*70)
    print("END OF REPORT".center(70))
    print("="*70 + "\n")


def export_to_csv():
    """تصدير البيانات إلى CSV"""
    import csv
    
    trades_data = load_json("logs/trades.json")
    trades = trades_data.get("trades", [])
    
    if not trades:
        print("No trades to export")
        return
    
    output_file = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
    
    print(f"✓ Trades exported to {output_file}")


def main():
    """الوظيفة الرئيسية"""
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_to_csv()
    else:
        print_report()


if __name__ == "__main__":
    main()
