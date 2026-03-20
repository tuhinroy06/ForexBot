"""
Backtester — Simulate signal strategy on historical daily data.

Usage:
    python backtest.py                    # backtest all pairs, last 12 months
    python backtest.py --pair EURUSD      # single pair
    python backtest.py --months 24        # extend lookback

Output:
    - Win rate, profit factor, expectancy, max drawdown
    - Equity curve saved to backtest_results/equity_<pair>.csv
    - Summary saved to backtest_results/summary.txt
"""

import asyncio
import aiohttp
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import Config

config = Config()
API_KEY = config.ALPHA_VANTAGE_API_KEY
BASE_URL = "https://www.alphavantage.co/query"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")

PAIRS = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "XAUUSD": ("XAU", "USD"),
}

PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,  "XAUUSD": 0.10}
SL_PIPS  = {"EURUSD": 20,     "GBPUSD": 25,      "USDJPY": 20,    "XAUUSD": 50}
RR1 = 1.5
RISK_PER_TRADE = 0.01
INITIAL_BALANCE = 10_000
FORWARD_BARS = 20


# ── Data ─────────────────────────────────────────────────────────────────────

async def fetch_daily(pair: str, months: int) -> pd.DataFrame:
    if pair == "XAUUSD":
        return await _fetch_xauusd_daily(months)
    from_s, to_s = PAIRS[pair]
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_s, "to_symbol": to_s,
        "outputsize": "full", "apikey": API_KEY,
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()

    key = "Time Series FX (Daily)"
    if key not in data:
        return pd.DataFrame()

    records = [{"time": pd.to_datetime(t),
                "open": float(v["1. open"]), "high": float(v["2. high"]),
                "low":  float(v["3. low"]),  "close": float(v["4. close"])}
               for t, v in data[key].items()]

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=months)
    return df[df["time"] >= cutoff].reset_index(drop=True)


async def _fetch_xauusd_daily(months: int) -> pd.DataFrame:
    params = {"function": "TIME_SERIES_DAILY", "symbol": "GLD",
              "outputsize": "full", "apikey": API_KEY}
    async with aiohttp.ClientSession() as s:
        async with s.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()

    key = "Time Series (Daily)"
    if key not in data:
        return pd.DataFrame()

    records = [{"time": pd.to_datetime(t),
                "open": float(v["1. open"]) * 10, "high": float(v["2. high"]) * 10,
                "low":  float(v["3. low"]) * 10,  "close": float(v["4. close"]) * 10}
               for t, v in data[key].items()]

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=months)
    return df[df["time"] >= cutoff].reset_index(drop=True)


# ── Indicators ────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].copy()

    d = c.diff()
    g = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
    l = (-d.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    df["macd_hist"] = m - m.ewm(span=9, adjust=False).mean()

    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    df["ema_bull"] = ((e9 > e21) & (e21 > e50)).astype(int)
    df["ema_bear"] = ((e9 < e21) & (e21 < e50)).astype(int)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pos"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    h, lo, cp = df["high"], df["low"], c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    up  = df["high"].diff()
    dn  = -df["low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / (df["atr"] + 1e-9)
    ndi = 100 * ndm.ewm(span=14, adjust=False).mean() / (df["atr"] + 1e-9)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    df["adx"] = dx.ewm(span=14, adjust=False).mean()

    return df


# ── Signal Logic ──────────────────────────────────────────────────────────────

def generate_signal(row: pd.Series) -> str:
    score = 0
    rsi = row["rsi"]
    if rsi < 30:   score += 2
    elif rsi < 45: score += 1
    elif rsi > 70: score -= 2
    elif rsi > 55: score -= 1

    if row["macd_hist"] > 0: score += 2
    else:                     score -= 2

    if row["ema_bull"]:   score += 2
    elif row["ema_bear"]: score -= 2

    if row["bb_pos"] < 0.15:   score += 1
    elif row["bb_pos"] > 0.85: score -= 1

    if row["adx"] > 25:
        score = int(score * 1.2)

    if score > 0: return "BUY"
    if score < 0: return "SELL"
    return "NONE"


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, pair: str) -> dict:
    pip  = PIP_SIZE.get(pair, 0.0001)
    sl_d = SL_PIPS.get(pair, 20) * pip

    balance = INITIAL_BALANCE
    equity  = [balance]
    trades  = []
    peak    = balance
    max_dd  = 0.0

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    for i in range(55, len(df) - FORWARD_BARS):
        signal = generate_signal(df.iloc[i])
        if signal == "NONE":
            continue

        entry   = closes[i]
        atr_sl  = df["atr"].iloc[i] * 1.2
        sl_dist = max(sl_d, atr_sl)
        tp1     = entry + sl_dist * RR1 if signal == "BUY" else entry - sl_dist * RR1
        sl      = entry - sl_dist        if signal == "BUY" else entry + sl_dist

        result   = "OPEN"
        pnl_pct  = 0.0
        exit_bar = FORWARD_BARS

        for j in range(1, FORWARD_BARS + 1):
            idx = i + j
            if signal == "BUY":
                if highs[idx] >= tp1:
                    result = "WIN";  pnl_pct = RR1 * RISK_PER_TRADE; exit_bar = j; break
                if lows[idx]  <= sl:
                    result = "LOSS"; pnl_pct = -RISK_PER_TRADE;       exit_bar = j; break
            else:
                if lows[idx]  <= tp1:
                    result = "WIN";  pnl_pct = RR1 * RISK_PER_TRADE; exit_bar = j; break
                if highs[idx] >= sl:
                    result = "LOSS"; pnl_pct = -RISK_PER_TRADE;       exit_bar = j; break

        if result == "OPEN":
            continue

        pnl      = balance * pnl_pct
        balance += pnl
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100
        max_dd   = max(max_dd, dd)
        equity.append(balance)

        trades.append({
            "date": str(df["time"].iloc[i]), "pair": pair, "signal": signal,
            "entry": round(entry, 5), "result": result,
            "pnl": round(pnl, 2), "balance": round(balance, 2), "bars_held": exit_bar,
        })

    if not trades:
        return {"pair": pair, "error": "No completed trades"}

    t = pd.DataFrame(trades)
    wins   = t[t["result"] == "WIN"]
    losses = t[t["result"] == "LOSS"]

    gross_profit  = wins["pnl"].sum()   if len(wins)   else 0
    gross_loss    = abs(losses["pnl"].sum()) if len(losses) else 1e-9

    return {
        "pair":          pair,
        "total_trades":  len(t),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(t) * 100, 1),
        "profit_factor": round(gross_profit / gross_loss, 2),
        "expectancy":    round(t["pnl"].mean(), 2),
        "net_pnl":       round(balance - INITIAL_BALANCE, 2),
        "net_pct":       round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 1),
        "max_drawdown":  round(max_dd, 1),
        "final_balance": round(balance, 2),
        "equity_curve":  equity,
        "trades":        trades,
    }


def print_report(r: dict):
    if "error" in r:
        print(f"\n❌ {r['pair']}: {r['error']}")
        return
    grade = "✅ PROFITABLE" if r["net_pnl"] > 0 else "❌ UNPROFITABLE"
    qual  = ("🟢 STRONG"  if r["profit_factor"] >= 1.5 and r["win_rate"] >= 55 else
             "🟡 AVERAGE" if r["profit_factor"] >= 1.0 else "🔴 POOR")
    print(f"""
╔══════════════════════════════════════════╗
  {r['pair']} Backtest Results  {grade}
╚══════════════════════════════════════════╝
  Trades:         {r['total_trades']}  ({r['wins']}W / {r['losses']}L)
  Win Rate:       {r['win_rate']}%
  Profit Factor:  {r['profit_factor']}   {qual}
  Expectancy:     ${r['expectancy']} per trade
  Net P&L:        ${r['net_pnl']} ({r['net_pct']}%)
  Max Drawdown:   {r['max_drawdown']}%
  Final Balance:  ${r['final_balance']}
""")


def save_results(results: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = [f"Backtest Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n", "="*50+"\n"]
    for r in results:
        if "error" in r:
            lines.append(f"{r['pair']}: ERROR — {r['error']}\n")
            continue
        lines.append(
            f"{r['pair']}\n"
            f"  Trades: {r['total_trades']} | Win: {r['win_rate']}% | PF: {r['profit_factor']}\n"
            f"  Net P&L: ${r['net_pnl']} ({r['net_pct']}%) | MaxDD: {r['max_drawdown']}%\n\n"
        )
        pd.DataFrame({"balance": r["equity_curve"]}).to_csv(
            os.path.join(OUTPUT_DIR, f"equity_{r['pair']}.csv"), index=False)
        pd.DataFrame(r["trades"]).to_csv(
            os.path.join(OUTPUT_DIR, f"trades_{r['pair']}.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.writelines(lines)
    print(f"\n💾 Results saved to {OUTPUT_DIR}/")


async def main(pairs: list, months: int):
    print(f"\n📊 Backtesting {len(pairs)} pairs over {months} months...\n")
    results = []
    for i, pair in enumerate(pairs):
        if i > 0:
            print("  ⏳ Waiting 15s for API rate limit...")
            await asyncio.sleep(15)
        print(f"  Fetching {pair}...")
        df = await fetch_daily(pair, months)
        if df.empty:
            results.append({"pair": pair, "error": "No data"}); continue
        df = add_indicators(df).dropna().reset_index(drop=True)
        r  = run_backtest(df, pair)
        print_report(r)
        results.append(r)

    save_results(results)
    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n{'='*45}")
        print(f"  PORTFOLIO AVERAGE")
        print(f"  Win Rate:       {np.mean([r['win_rate'] for r in valid]):.1f}%")
        print(f"  Profit Factor:  {np.mean([r['profit_factor'] for r in valid]):.2f}")
        print(f"  Max Drawdown:   {np.mean([r['max_drawdown'] for r in valid]):.1f}%")
        print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",   type=str, default=None)
    parser.add_argument("--months", type=int, default=12)
    args = parser.parse_args()
    pairs_to_test = [args.pair.upper()] if args.pair else list(PAIRS.keys())
    asyncio.run(main(pairs_to_test, args.months))
