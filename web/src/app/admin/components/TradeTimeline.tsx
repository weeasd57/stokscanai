"use client";

import React from 'react';
import dynamic from 'next/dynamic';

// Use dynamic import to avoid SSR issues with window in apexcharts
// and allow passing props without strict TS typing complaints.
const ReactApexChart = dynamic<any>(
  () => import('react-apexcharts').then((m) => m.default as any),
  {
    ssr: false,
    loading: () => (
      <div className="p-8 rounded-xl border border-white/10 bg-zinc-900/40 text-center text-zinc-400">
        Loading chart engine…
      </div>
    )
  }
);

export const TradeTimeline = ({ trades }: { trades: any[] }) => {
  if (!trades || trades.length === 0) {
    return (
      <div className="p-6 rounded-2xl border border-white/5 text-center text-zinc-500">
        No trade data available
      </div>
    );
  }

  const [isPreparing, setIsPreparing] = React.useState(true);
  const [seriesData, setSeriesData] = React.useState<any[]>([]);
  const [winRatePct, setWinRatePct] = React.useState<number>(0);
  const [avgDurationDays, setAvgDurationDays] = React.useState<number>(0);
  const [requiredBalance, setRequiredBalance] = React.useState<number>(0);

  React.useEffect(() => {
    let cancelled = false;

    const run = () => {
      setIsPreparing(true);

      const safeTrades = Array.isArray(trades) ? trades : [];
      const nextSeries: any[] = [];

      let wins = 0;
      let durationSum = 0;
      const dailyExposure: Record<string, number> = {};
      let maxExposure = 0;

      const acceptedTrades = safeTrades.filter(t => {
        const status = t.features?.backtest_status || t.Status || t.status || 'Accepted';
        return status === 'Accepted';
      });

      for (const t of acceptedTrades) {
        const profitLossPct = Number(t.profit_loss_pct ?? 0) || 0;
        const isWin = profitLossPct > 0;
        if (isWin) wins += 1;

        const entryRaw = t.features?.entry_date || t.features?.trade_date || t.Entry_Date || t.created_at;
        const exitRaw = t.features?.exit_date || t.Exit_Date || t.features?.trade_date || t.created_at;
        const entryMs = new Date(entryRaw).getTime();
        const exitMs = new Date(exitRaw).getTime();
        if (!Number.isFinite(entryMs) || !Number.isFinite(exitMs)) continue;

        const startMs = Math.min(entryMs, exitMs);
        const endMs = Math.max(entryMs, exitMs);
        const days = Math.max(0, Math.ceil((endMs - startMs) / (1000 * 60 * 60 * 24)));
        durationSum += days;

        const positionCash = Number(
          t.features?.position_cash ??
          t.Position_Cash ??
          t.position_cash ??
          0
        ) || 0;

        if (positionCash > 0) {
          const cursor = new Date(startMs);
          const end = new Date(endMs);
          let steps = 0;
          const maxDays = 400;

          while (cursor <= end && steps < maxDays) {
            const key = cursor.toISOString().slice(0, 10);
            dailyExposure[key] = (dailyExposure[key] ?? 0) + positionCash;
            if (dailyExposure[key] > maxExposure) maxExposure = dailyExposure[key];
            cursor.setDate(cursor.getDate() + 1);
            steps += 1;
          }
        }

        nextSeries.push({
          x: t.symbol || 'UNKNOWN',
          y: [startMs, endMs],
          fillColor: isWin ? '#00E396' : '#FF4560',
          meta: {
            pnl: `${isWin ? '+' : ''}${profitLossPct.toFixed(2)}%`,
            pnlCash: `${Number(t.features?.profit_cash || 0).toLocaleString()} EGP`,
            entryPrice: (t.entry_price || 0).toFixed(2),
            exitPrice: (t.exit_price || 0).toFixed(2),
            days,
          }
        });
      }

      const nextWinRate = acceptedTrades.length > 0 ? (wins / acceptedTrades.length) * 100 : 0;
      const nextAvgDuration = acceptedTrades.length > 0 ? durationSum / acceptedTrades.length : 0;


      if (cancelled) return;
      setSeriesData(nextSeries);
      setWinRatePct(nextWinRate);
      setAvgDurationDays(nextAvgDuration);
      setRequiredBalance(maxExposure);
      setIsPreparing(false);
    };

    const w = globalThis as any;
    const ric: any = w.requestIdleCallback;
    const cancelRic: any = w.cancelIdleCallback;

    let idleId: any = null;
    if (typeof ric === "function") {
      idleId = ric(run, { timeout: 800 });
    } else {
      idleId = setTimeout(run, 0);
    }

    return () => {
      cancelled = true;
      if (typeof cancelRic === "function" && idleId) cancelRic(idleId);
      if (idleId) clearTimeout(idleId);
    };
  }, [trades]);

  const series = [
    {
      name: 'Trades',
      data: seriesData
    }
  ];

  const options: any = {
    chart: {
      type: 'rangeBar',
      height: 350,
      background: 'transparent',
      selection: { enabled: false },
      toolbar: {
        show: true,
        tools: {
          download: true,
          pan: true,
          selection: false,
          zoom: false,
          zoomin: false,
          zoomout: false,
          reset: false
        },
        autoSelected: 'pan'
      },
      zoom: { enabled: false },
      animations: {
        enabled: false,
        easing: 'easeinout',
        speed: 800,
        animateGradually: {
          enabled: true,
          delay: 150
        },
        dynamicAnimation: {
          enabled: true,
          speed: 350
        }
      }
    },
    plotOptions: {
      bar: {
        horizontal: true, // عشان تكون أشرطة أفقية زي itch.io
        barHeight: '70%',
        rangeBarGroupRows: false, // كل صفقة في سطر منفصل
        distributed: true,
        dataLabels: {
          hideOverflowingLabels: false
        }
      }
    },
    xaxis: {
      type: 'datetime', // محور الوقت
      labels: {
        style: { colors: '#9CA3AF', fontSize: '11px' },
        datetimeUTC: false
      },
      title: {
        text: 'Timeline',
        style: { color: '#9CA3AF', fontSize: '12px' }
      }
    },
    yaxis: {
      labels: {
        style: { colors: '#9CA3AF', fontSize: '11px' },
        minWidth: 60
      },
      title: {
        text: 'Symbols',
        style: { color: '#9CA3AF', fontSize: '12px' }
      }
    },
    tooltip: {
      theme: 'dark',
      style: {
        fontSize: '12px',
        fontFamily: 'monospace'
      },
      custom: function ({ seriesIndex, dataPointIndex, w }: any) {
        const data = w.config.series[seriesIndex].data[dataPointIndex];
        const start = new Date(data.y[0]).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric'
        });
        const end = new Date(data.y[1]).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric'
        });

        return (
          '<div class="p-3 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl" style="min-width: 200px;">' +
          '<div class="font-bold text-white mb-2">' + data.x + '</div>' +
          '<div class="text-zinc-300 text-sm mb-1">📅 ' + start + ' → ' + end + ' (' + data.meta.days + ' days)</div>' +
          '<div class="text-zinc-300 text-sm mb-1">💰 PnL: <span class="' + (data.fillColor === '#00E396' ? 'text-emerald-400' : 'text-red-400') + ' font-bold">' + data.meta.pnl + '</span></div>' +
          '<div class="text-zinc-300 text-sm mb-1">💵 Cash: <span class="' + (data.fillColor === '#00E396' ? 'text-emerald-400' : 'text-red-400') + ' font-bold">' + data.meta.pnlCash + '</span></div>' +
          '<div class="text-zinc-400 text-xs">Entry: ' + data.meta.entryPrice + ' → Exit: ' + data.meta.exitPrice + '</div>' +
          '</div>'
        );
      }
    },
    grid: {
      borderColor: '#374151',
      xaxis: {
        lines: {
          show: true
        }
      },
      yaxis: {
        lines: {
          show: false
        }
      }
    },
    legend: {
      show: false
    },
    dataLabels: {
      enabled: false
    },
    states: {
      hover: {
        filter: {
          type: 'lighten',
          value: 0.1
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl border border-white/5 bg-white/5">
          <div className="text-[9px] uppercase font-black text-zinc-500 tracking-widest">Total Trades</div>
          <div className="text-2xl font-black text-white mt-1">{isPreparing ? '—' : seriesData.length}</div>
        </div>
        <div className="p-4 rounded-2xl border border-white/5 bg-white/5">
          <div className="text-[9px] uppercase font-black text-zinc-500 tracking-widest">Win Rate</div>
          <div className="text-2xl font-black text-emerald-400 mt-1">
            {isPreparing ? '—' : `${winRatePct.toFixed(1)}%`}
          </div>
        </div>
        <div className="p-4 rounded-2xl border border-white/5 bg-white/5">
          <div className="text-[9px] uppercase font-black text-zinc-500 tracking-widest">Avg Duration</div>
          <div className="text-2xl font-black text-white mt-1">
            {isPreparing ? '—' : `${avgDurationDays.toFixed(1)}d`}
          </div>
        </div>
        <div className="p-4 rounded-2xl border border-white/5 bg-white/5">
          <div className="text-[9px] uppercase font-black text-zinc-500 tracking-widest">Required Balance</div>
          <div className="text-2xl font-black text-emerald-400 mt-1">
            {isPreparing ? '—' : (requiredBalance > 0 ? Math.round(requiredBalance).toLocaleString() : '—')}
          </div>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="rounded-xl overflow-hidden shadow-lg border border-white/10 bg-zinc-900/40">
        <div className="bg-zinc-800/40 p-4 border-b border-white/10">
          <h3 className="text-white font-bold text-sm">⏳ Trade Duration Timeline</h3>
          <p className="text-zinc-400 text-xs mt-1">Horizontal bars show trade duration and profitability</p>
        </div>
        <div className="p-4">
          {isPreparing ? (
            <div className="p-10 rounded-2xl border border-white/10 bg-zinc-950/40 text-center text-zinc-400">
              Preparing chart data…
            </div>
          ) : (
            <ReactApexChart
              options={options}
              series={series}
              type="rangeBar"
              height={Math.min(600, Math.max(320, seriesData.length * 26 + 140))}
            />
          )}
        </div>
        <div className="bg-zinc-800/40 p-3 border-t border-white/10">
          <div className="flex items-center justify-center gap-6 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-emerald-500" />
              <span className="text-zinc-400">Profitable Trade</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-red-500" />
              <span className="text-zinc-400">Loss Trade</span>
            </div>
            <div className="text-zinc-500">
              Showing: {seriesData.length} trades | {seriesData.filter(d => d.fillColor === '#00E396').length} wins
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
