#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Automation & Optimization System
==========================================
Automated parameter grid search for finding optimal trading configurations.
"""

import subprocess
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os
import re
from pathlib import Path


class BacktestOptimizer:
    """
    Manages automated batch backtesting with parameter grid search.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the optimizer.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = base_dir or os.getcwd()
        self.results = []
        self.current_job_id = None
        
    def _run_single_backtest(self, params: Dict) -> Optional[Dict]:
        """
        Execute a single backtest using backtest_radar.py
        
        Args:
            params: Backtest parameters
            
        Returns:
            Parsed results dictionary or None on failure
        """
        try:
            # Build command
            cmd = [
                "py", "-m", "api.backtest_radar",
                "--model", params.get("model", "KING 👑.pkl"),
                "--exchange", params.get("exchange", "CRYPTO"),
                "--start", params["start_date"],
                "--end", params["end_date"],
                "--capital", str(params.get("capital", 100000)),
                "--meta-threshold", str(params.get("wave_confluence", 0.8)),
                "--target-pct", str(params.get("target_percent", 10) / 100),
                "--stop-loss-pct", str(params.get("stop_loss_percent", 5) / 100),
            ]
            
            # Add council filter if specified
            if params.get("council_filter") and params["council_filter"] != "Direct Execution (No Council)":
                cmd.extend(["--council", params["council_filter"]])
                if params.get("validator_threshold") is not None:
                    cmd.extend(["--validator-threshold", str(params["validator_threshold"])])
            
            # Add timeframe if specified
            if params.get("timeframe"):
                tf_map = {"1H": "1h", "4H": "4h", "1D": "1d"}
                tf = tf_map.get(params["timeframe"].split()[0], "1h")
                cmd.extend(["--timeframe", tf])
            
            # Execute backtest
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=600,  # Increased timeout to 10 minutes
                encoding='utf-8'
            )
            
            # Parse output
            parsed = self._parse_backtest_output(result.stdout, result.stderr)
            if parsed is None:
                print(f"❌ Failed to parse backtest output for params: {params}")
                if result.stderr:
                    print(f"❌ STDERR:\n{result.stderr}")
                else:
                    print(f"❌ STDOUT (first 500 chars):\n{result.stdout[:500]}...")
            return parsed
            
        except subprocess.TimeoutExpired:
            print(f"⚠️ Backtest timeout for params: {params}")
            return None
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            return None
    
    def _parse_backtest_output(self, stdout: str, stderr: str) -> Optional[Dict]:
        """
        Parse backtest output to extract metrics.
        
        Args:
            stdout: Standard output from backtest
            stderr: Standard error from backtest
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Extract JSON trades log
            json_match = re.search(
                r'--- JSON TRADES LOG START ---\s*(.*?)\s*--- JSON TRADES LOG END ---',
                stdout,
                re.DOTALL
            )
            
            if not json_match:
                return None
            
            trades_json = json_match.group(1).strip()
            if not trades_json or trades_json == "[]":
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_cash": 0,
                    "profit_percent": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                }
            
            trades = json.loads(trades_json)
            df = pd.DataFrame(trades)
            
            # Calculate metrics
            accepted = df[df.get("Status", "Accepted").str.lower() == "accepted"]
            total_trades = len(accepted)
            
            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_cash": 0,
                    "profit_percent": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                }
            
            wins = accepted[accepted["PnL_Pct"] > 0]
            losses = accepted[accepted["PnL_Pct"] <= 0]
            
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate profit
            if "Profit_Cash" in accepted.columns:
                profit_cash = accepted["Profit_Cash"].sum()
            else:
                profit_cash = 0
            
            # Extract capital from output
            capital_match = re.search(r'capital[:\s]+(\d+)', stdout, re.IGNORECASE)
            capital = float(capital_match.group(1)) if capital_match else 100000
            
            profit_percent = (profit_cash / capital) * 100 if capital > 0 else 0
            
            # Calculate drawdown
            if "Cumulative_Profit" in accepted.columns:
                cumulative = accepted["Cumulative_Profit"].values
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / (running_max + 1e-9)
                max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0
            
            # Calculate Sharpe ratio (simplified)
            if "PnL_Pct" in accepted.columns and len(accepted) > 1:
                returns = accepted["PnL_Pct"].values
                sharpe_ratio = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Extract Council Stats
            pre_council_match = re.search(r'Pre-Council Trades:\s+(\d+)', stdout)
            post_council_match = re.search(r'Post-Council Trades:\s+(\d+)', stdout)
            
            pre_trades = int(pre_council_match.group(1)) if pre_council_match else total_trades
            post_trades = int(post_council_match.group(1)) if post_council_match else total_trades

            return {
                "total_trades": post_trades,
                "pre_council_trades": pre_trades,
                "post_council_trades": post_trades,
                "win_rate": win_rate,
                "profit_cash": profit_cash,
                "profit_percent": profit_percent,
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
            }
            
        except Exception as e:
            print(f"❌ Critical error parsing backtest output: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_parameters(self,
                          wave_values: List[float],
                          target_values: List[int],
                          stoploss_values: List[int],
                          base_params: Dict,
                          progress_callback=None) -> pd.DataFrame:
        """
        Run grid search across parameter combinations.
        
        Args:
            wave_values: Wave confluence thresholds to test
            target_values: Target percentages to test
            stoploss_values: Stop loss percentages to test
            base_params: Fixed parameters (model, exchange, dates, etc.)
            progress_callback: Optional callback function(current, total, result)
            
        Returns:
            DataFrame with all results
        """
        validator_values = base_params.get("validator_values", [None])
        if not validator_values:
            validator_values = [None]

        # Generate all combinations (4D: Wave x Target x Stoploss x Validator)
        combinations = list(product(wave_values, target_values, stoploss_values, validator_values))
        total_tests = len(combinations)
        
        print(f"🚀 Starting {total_tests} backtest combinations (KING x Validator grid)...")
        print("=" * 70)
        
        self.results = []
        
        for idx, (wave, target, stoploss, validator) in enumerate(combinations, 1):
            # Prepare parameters
            test_params = base_params.copy()
            test_params.update({
                'wave_confluence': wave,
                'target_percent': target,
                'stop_loss_percent': stoploss,
                'validator_threshold': validator
            })
            
            val_str = f", Val={validator}" if validator is not None else ""
            print(f"\n[{idx}/{total_tests}] Testing: Wave={wave}, Target={target}%, StopLoss={stoploss}%{val_str}")
            
            # Run backtest
            result = self._run_single_backtest(test_params)
            
            if result:
                # Store results
                self.results.append({
                    'test_number': idx,
                    'wave_confluence': wave,
                    'validator_threshold': validator,
                    'target_percent': target,
                    'stop_loss_percent': stoploss,
                    'profit_cash': result.get('profit_cash', 0),
                    'profit_percent': result.get('profit_percent', 0),
                    'win_rate': result.get('win_rate', 0),
                    'total_trades': result.get('total_trades', 0),
                    'pre_council_trades': result.get('pre_council_trades', 0),
                    'post_council_trades': result.get('post_council_trades', 0),
                    'winning_trades': result.get('winning_trades', 0),
                    'losing_trades': result.get('losing_trades', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"✅ Profit: {result.get('profit_cash', 0):,.2f} ({result.get('profit_percent', 0):.2f}%)")
                print(f"✅ Win Rate: {result.get('win_rate', 0):.1f}%")
            else:
                print("⚠️ Test failed")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(idx, total_tests, result)
        
        print("\n" + "=" * 70)
        print("✅ All tests completed!")
        
        return pd.DataFrame(self.results)
    
    def save_results(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save results to CSV file.
        
        Args:
            df: Results DataFrame
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.csv"
        
        output_path = os.path.join(self.base_dir, "logs", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Results saved to: {output_path}")
        
        return output_path
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Report text
        """
        if df.empty:
            return "No results to report."
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("📊 BACKTEST OPTIMIZATION REPORT")
        report.append("=" * 70)
        
        # Top 5 by profit percentage
        report.append("\n🏆 TOP 5 CONFIGURATIONS BY PROFIT %:")
        report.append("-" * 70)
        top_profit = df.nlargest(5, 'profit_percent')
        for idx, row in top_profit.iterrows():
            report.append(f"\n#{row['test_number']}:")
            val_str = f", Val: {row['validator_threshold']}" if row['validator_threshold'] is not None else ""
            report.append(f"  Wave: {row['wave_confluence']}{val_str}, Target: {row['target_percent']}%, StopLoss: {row['stop_loss_percent']}%")
            report.append(f"  💰 Profit: {row['profit_cash']:,.2f} ({row['profit_percent']:.2f}%)")
            report.append(f"  📈 Win Rate: {row['win_rate']:.1f}%")
            report.append(f"  🎯 Trades: {row['total_trades']}")
            report.append(f"  📉 Max Drawdown: {row['max_drawdown']:.2f}%")
        
        # Top 5 by win rate
        report.append("\n\n🎯 TOP 5 CONFIGURATIONS BY WIN RATE:")
        report.append("-" * 70)
        top_winrate = df.nlargest(5, 'win_rate')
        for idx, row in top_winrate.iterrows():
            report.append(f"\n#{row['test_number']}:")
            val_str = f", Val: {row['validator_threshold']}" if row['validator_threshold'] is not None else ""
            report.append(f"  Wave: {row['wave_confluence']}{val_str}, Target: {row['target_percent']}%, StopLoss: {row['stop_loss_percent']}%")
            report.append(f"  📈 Win Rate: {row['win_rate']:.1f}%")
            report.append(f"  💰 Profit: {row['profit_cash']:,.2f} ({row['profit_percent']:.2f}%)")
            report.append(f"  🎯 Trades: {row['total_trades']}")
        
        # Statistics
        report.append("\n\n📊 OVERALL STATISTICS:")
        report.append("-" * 70)
        report.append(f"Total Tests: {len(df)}")
        report.append(f"Average Profit: {df['profit_percent'].mean():.2f}%")
        report.append(f"Best Profit: {df['profit_percent'].max():.2f}%")
        report.append(f"Worst Profit: {df['profit_percent'].min():.2f}%")
        report.append(f"Average Win Rate: {df['win_rate'].mean():.1f}%")
        report.append(f"Average Sharpe Ratio: {df['sharpe_ratio'].mean():.2f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def run_optimization_job(job_params: Dict) -> Dict:
    """
    Run a complete optimization job.
    
    Args:
        job_params: Job configuration
        
    Returns:
        Job results
    """
    optimizer = BacktestOptimizer()
    
    # Extract parameters
    wave_values = job_params.get("wave_values", [0.7, 0.8, 0.9])
    validator_values = job_params.get("validator_values", [None])
    target_values = job_params.get("target_values", [5, 10, 15])
    stoploss_values = job_params.get("stoploss_values", [3, 5, 7])
    base_params = job_params.get("base_params", {})
    
    # Run optimization
    results_df = optimizer.optimize_parameters(
        wave_values=wave_values,
        target_values=target_values,
        stoploss_values=stoploss_values,
        base_params={**base_params, "validator_values": validator_values}
    )
    
    # Save results
    csv_path = optimizer.save_results(results_df)
    
    # Generate report
    report = optimizer.generate_report(results_df)
    
    # Save report
    report_path = csv_path.replace('.csv', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    
    return {
        "status": "completed",
        "total_tests": len(results_df),
        "csv_path": csv_path,
        "report_path": report_path,
        "best_config": results_df.nlargest(1, 'profit_percent').to_dict('records')[0] if not results_df.empty else None
    }
