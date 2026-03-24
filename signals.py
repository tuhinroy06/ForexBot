"""
Signal Engine v3 — Twelve Data API
Enhanced with professional trading strategies:
1. Trend Following    — EMA stack + ADX filter
2. Mean Reversion     — BB + RSI divergence
3. Momentum           — MACD + Stochastic confluence
4. Support/Resistance — Pivot points + price action
5. Ichimoku Cloud     — Full cloud analysis
6. Volume Spread      — ATR volatility filter
7. Williams %R        — Overbought/oversold confirmation
8. CCI               — Commodity Channel Index
9. Supertrend         — Dynamic trend line
10. Smart Money       — Session + liquidity analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import pickle
import os
import time as _time
from datetime import datetime, timezone
from typing import Optional, Tuple

TD_SYMBOLS = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD", "EURGBP": "EUR/GBP", "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY", "AUDJPY": "AUD/JPY", "CADJPY": "CAD/JPY",
    "CHFJPY": "CHF/JPY", "EURCHF": "EUR/CHF", "EURAUD": "EUR/AUD",
    "EURCAD": "EUR/CAD", "GBPAUD": "GBP/AUD", "GBPCAD": "GBP/CAD",
    "GBPCHF": "GBP/CHF", "AUDCAD": "AUD/CAD", "AUDCHF": "AUD/CHF",
    "AUDNZD": "AUD/NZD", "NZDJPY": "NZD/JPY",
    "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD",
}

DISPLAY_NAMES = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD", "EURGBP": "EUR/GBP", "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY", "AUDJPY": "AUD/JPY", "CADJPY": "CAD/JPY",
    "CHFJPY": "CHF/JPY", "EURCHF": "EUR/CHF", "EURAUD": "EUR/AUD",
    "EURCAD": "EUR/CAD", "GBPAUD": "GBP/AUD", "GBPCAD": "GBP/CAD",
    "GBPCHF": "GBP/CHF", "AUDCAD": "AUD/CAD", "AUDCHF": "AUD/CHF",
    "AUDNZD": "AUD/NZD", "NZDJPY": "NZD/JPY",
    "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD",
}

PIP_SIZE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCHF": 0.0001,
    "AUDUSD": 0.0001, "USDCAD": 0.0001, "NZDUSD": 0.0001,
    "EURGBP": 0.0001, "EURCHF": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "GBPAUD": 0.0001, "GBPCAD": 0.0001,
    "GBPCHF": 0.0001, "AUDCAD": 0.0001, "AUDCHF": 0.0001,
    "AUDNZD": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    "AUDJPY": 0.01,   "CADJPY": 0.01,   "CHFJPY": 0.01, "NZDJPY": 0.01,
    "XAUUSD": 0.10,   "XAGUSD": 0.01,
}

SL_PIPS = {
    "EURUSD": 20, "GBPUSD": 25, "USDJPY": 20, "USDCHF": 20,
    "AUDUSD": 20, "USDCAD": 22, "NZDUSD": 20,
    "EURGBP": 18, "EURJPY": 25, "GBPJPY": 35, "AUDJPY": 25,
    "CADJPY": 25, "CHFJPY": 25, "EURCHF": 20, "EURAUD": 25,
    "EURCAD": 25, "GBPAUD": 35, "GBPCAD": 35, "GBPCHF": 30,
    "AUDCAD": 22, "AUDCHF": 22, "AUDNZD": 22, "NZDJPY": 25,
    "XAUUSD": 50, "XAGUSD": 30,
}

SESSIONS = {
    "London":   (7, 16),
    "New York": (13, 21),
}

# Cache: {pair_interval: (timestamp, df)}
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


class SignalEngine:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = "https://api.twelvedata.com"
        self.ml_model = self._load_ml_model()

    # ── Data Fetching ─────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, pair: str, interval: str = "1h", outputsize: int = 150) -> Optional[pd.DataFrame]:
        cache_key = f"{pair}_{interval}"
        now_ts    = _time.time()

        if cache_key in _cache:
            cached_ts, cached_df = _cache[cache_key]
            if now_ts - cached_ts < CACHE_TTL:
                return cached_df.copy()

        symbol = TD_SYMBOLS.get(pair)
        if not symbol:
            return None

        params = {
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     self.api_key,
            "format":     "JSON",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/time_series",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
        except Exception:
            return _cache[cache_key][1].copy() if cache_key in _cache else None

        if data.get("status") == "error":
            if cache_key in _cache:
                return _cache[cache_key][1].copy()
            return None

        values = data.get("values")
        if not values:
            return None

        records = []
        for v in values:
            try:
                records.append({
                    "time":  pd.to_datetime(v["datetime"]),
                    "open":  float(v["open"]),
                    "high":  float(v["high"]),
                    "low":   float(v["low"]),
                    "close": float(v["close"]),
                })
            except Exception:
                continue

        if not records:
            return None

        df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
        _cache[cache_key] = (_time.time(), df)
        return df.copy()

    async def _resample_to_h4(self, df_h1: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df_h1 is None or len(df_h1) < 8:
            return None
        df = df_h1.set_index("time")
        df_h4 = df.resample("4h").agg({
            "open": "first", "high": "max",
            "low": "min",    "close": "last",
        }).dropna().reset_index()
        return df_h4

    # ── Core Indicators ───────────────────────────────────────────────────────

    @staticmethod
    def ema(s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()

    @staticmethod
    def sma(s: pd.Series, p: int) -> pd.Series:
        return s.rolling(p).mean()

    @staticmethod
    def rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
        l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
        return 100 - 100 / (1 + g / l)

    @staticmethod
    def macd(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        e12 = s.ewm(span=12, adjust=False).mean()
        e26 = s.ewm(span=26, adjust=False).mean()
        m   = e12 - e26
        sig = m.ewm(span=9, adjust=False).mean()
        return m, sig, m - sig

    @staticmethod
    def bb(s: pd.Series, p: int = 20, k: float = 2.0):
        mid = s.rolling(p).mean()
        std = s.rolling(p).std()
        return mid + k*std, mid, mid - k*std

    @staticmethod
    def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()

    @staticmethod
    def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
        lo    = df["low"].rolling(k).min()
        hi    = df["high"].rolling(k).max()
        pct_k = 100 * (df["close"] - lo) / (hi - lo + 1e-9)
        return pct_k, pct_k.rolling(d).mean()

    @staticmethod
    def adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
        hi, lo = df["high"], df["low"]
        up  = hi.diff(); dn = -lo.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        av  = (hi - lo).ewm(span=p, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / (av + 1e-9)
        ndi = 100 * ndm.ewm(span=p, adjust=False).mean() / (av + 1e-9)
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        return dx.ewm(span=p, adjust=False).mean(), pdi, ndi

    # ── Advanced Indicators ───────────────────────────────────────────────────

    @staticmethod
    def williams_r(df: pd.DataFrame, p: int = 14) -> pd.Series:
        """Williams %R — overbought/oversold oscillator."""
        hi = df["high"].rolling(p).max()
        lo = df["low"].rolling(p).min()
        return -100 * (hi - df["close"]) / (hi - lo + 1e-9)

    @staticmethod
    def cci(df: pd.DataFrame, p: int = 20) -> pd.Series:
        """Commodity Channel Index — trend strength and reversals."""
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-9)

    @staticmethod
    def supertrend(df: pd.DataFrame, p: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Supertrend — dynamic support/resistance trend line."""
        hl2  = (df["high"] + df["low"]) / 2
        atr_ = df["high"] - df["low"]
        atr_ = atr_.ewm(span=p, adjust=False).mean()
        upper = hl2 + mult * atr_
        lower = hl2 - mult * atr_

        supertrend = pd.Series(index=df.index, dtype=float)
        direction  = pd.Series(index=df.index, dtype=float)

        supertrend.iloc[0] = lower.iloc[0]
        direction.iloc[0]  = 1

        for i in range(1, len(df)):
            prev_st  = supertrend.iloc[i-1]
            prev_dir = direction.iloc[i-1]
            close    = df["close"].iloc[i]

            if prev_dir == 1:
                st = max(lower.iloc[i], prev_st) if close > prev_st else upper.iloc[i]
                d  = 1 if close > st else -1
            else:
                st = min(upper.iloc[i], prev_st) if close < prev_st else lower.iloc[i]
                d  = -1 if close < st else 1

            supertrend.iloc[i] = st
            direction.iloc[i]  = d

        return supertrend, direction

    @staticmethod
    def ichimoku(df: pd.DataFrame):
        """Ichimoku Cloud — comprehensive trend/momentum/support system."""
        high = df["high"]; low = df["low"]; close = df["close"]

        # Tenkan-sen (Conversion Line): 9-period
        tenkan  = (high.rolling(9).max() + low.rolling(9).min()) / 2
        # Kijun-sen (Base Line): 26-period
        kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
        # Senkou Span A (Leading Span A): avg of tenkan+kijun, shifted 26
        span_a  = ((tenkan + kijun) / 2).shift(26)
        # Senkou Span B (Leading Span B): 52-period, shifted 26
        span_b  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        # Chikou Span (Lagging Span): close shifted -26
        chikou  = close.shift(-26)

        return tenkan, kijun, span_a, span_b, chikou

    @staticmethod
    def pivot_points(df: pd.DataFrame):
        """Classic Pivot Points — key S/R levels."""
        prev = df.iloc[-2]
        pp   = (prev["high"] + prev["low"] + prev["close"]) / 3
        r1   = 2 * pp - prev["low"]
        r2   = pp + (prev["high"] - prev["low"])
        s1   = 2 * pp - prev["high"]
        s2   = pp - (prev["high"] - prev["low"])
        return pp, r1, r2, s1, s2

    @staticmethod
    def rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 10) -> str:
        """Detect RSI divergence — one of the strongest reversal signals."""
        price_highs = close.rolling(lookback).max()
        price_lows  = close.rolling(lookback).min()
        rsi_highs   = rsi.rolling(lookback).max()
        rsi_lows    = rsi.rolling(lookback).min()

        curr_price = close.iloc[-1]
        prev_price = close.iloc[-lookback]
        curr_rsi   = rsi.iloc[-1]
        prev_rsi   = rsi.iloc[-lookback]

        # Bearish divergence: price makes higher high, RSI makes lower high
        if curr_price > prev_price and curr_rsi < prev_rsi and curr_rsi > 60:
            return "🔴 Bearish divergence"
        # Bullish divergence: price makes lower low, RSI makes higher low
        if curr_price < prev_price and curr_rsi > prev_rsi and curr_rsi < 40:
            return "🟢 Bullish divergence"
        return "🟡 No divergence"

    # ── Session / H4 / ML ─────────────────────────────────────────────────────

    @staticmethod
    def in_trading_session() -> Tuple[bool, str]:
        hour   = datetime.now(timezone.utc).hour
        active = [n for n, (s, e) in SESSIONS.items() if s <= hour < e]
        return (True, " + ".join(active)) if active else (False, "Asian (low liquidity)")

    def h4_trend(self, df_h4: Optional[pd.DataFrame]) -> str:
        if df_h4 is None or len(df_h4) < 55:
            return "NEUTRAL"
        close = df_h4["close"]
        e20   = self.ema(close, 20).iloc[-1]
        e50   = self.ema(close, 50).iloc[-1]
        price = close.iloc[-1]
        if price > e20 > e50:   return "BULL"
        elif price < e20 < e50: return "BEAR"
        return "NEUTRAL"

    def _load_ml_model(self):
        path = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def ml_confidence(self, features: dict) -> Optional[float]:
        if self.ml_model is None:
            return None
        try:
            X    = np.array([[features["rsi"], features["macd_hist"],
                              features["bb_pos"], features["ema_diff"],
                              features["atr_norm"], features["stoch_k"],
                              features["adx"], features["h4_bull"],
                              features["in_session"]]])
            prob = self.ml_model.predict_proba(X)[0]
            return round(max(prob) * 100, 1)
        except Exception:
            return None

    # ── Master Analysis ───────────────────────────────────────────────────────

    def analyze(self, df_h1: pd.DataFrame, df_h4: Optional[pd.DataFrame], pair: str) -> dict:
        close = df_h1["close"]
        high  = df_h1["high"]
        low   = df_h1["low"]
        price = close.iloc[-1]

        # ── Compute all indicators ────────────────────────────────────────────
        rsi_s              = self.rsi(close)
        _, _, macd_hist_s  = self.macd(close)
        bb_up, bb_mid, bb_lo = self.bb(close)
        e9   = self.ema(close, 9);  e21 = self.ema(close, 21)
        e50  = self.ema(close, 50); e200= self.ema(close, 200)
        atr_s              = self.atr(df_h1)
        stk_s, std_s       = self.stochastic(df_h1)
        adx_s, pdi_s, ndi_s= self.adx(df_h1)
        wr_s               = self.williams_r(df_h1)
        cci_s              = self.cci(df_h1)
        st_s, st_dir_s     = self.supertrend(df_h1)
        tenkan, kijun, span_a, span_b, _ = self.ichimoku(df_h1)

        # Current values
        rsi_v      = rsi_s.iloc[-1]
        mhist_v    = macd_hist_s.iloc[-1]
        mhist_prev = macd_hist_s.iloc[-2]
        bb_up_v    = bb_up.iloc[-1]; bb_lo_v = bb_lo.iloc[-1]
        bb_range   = bb_up_v - bb_lo_v
        bb_pos     = (price - bb_lo_v) / bb_range if bb_range > 0 else 0.5
        e9_v  = e9.iloc[-1];   e21_v = e21.iloc[-1]
        e50_v = e50.iloc[-1];  e200_v= e200.iloc[-1]
        e9_p  = e9.iloc[-2];   e21_p = e21.iloc[-2]
        atr_v      = atr_s.iloc[-1]
        stk_v      = stk_s.iloc[-1]; std_v = std_s.iloc[-1]
        adx_v      = adx_s.iloc[-1]
        pdi_v      = pdi_s.iloc[-1]; ndi_v = ndi_s.iloc[-1]
        wr_v       = wr_s.iloc[-1]
        cci_v      = cci_s.iloc[-1]
        st_dir_v   = st_dir_s.iloc[-1]
        st_prev    = st_dir_s.iloc[-2]
        tenkan_v   = tenkan.iloc[-1]; kijun_v = kijun.iloc[-1]
        span_a_v   = span_a.iloc[-1]; span_b_v = span_b.iloc[-1]
        cloud_top  = max(span_a_v, span_b_v) if not (np.isnan(span_a_v) or np.isnan(span_b_v)) else price
        cloud_bot  = min(span_a_v, span_b_v) if not (np.isnan(span_a_v) or np.isnan(span_b_v)) else price

        # Divergence
        div_signal = self.rsi_divergence(close, rsi_s)

        # Pivot points
        try:
            pp, r1, r2, s1, s2 = self.pivot_points(df_h1)
            pivot_signal = ("🟢 Above PP" if price > pp else "🔴 Below PP")
        except Exception:
            pp = price; pivot_signal = "🟡 N/A"

        # H4 trend
        trend_h4              = self.h4_trend(df_h4)
        in_session, sess_name = self.in_trading_session()

        # ── Scoring System ────────────────────────────────────────────────────
        # Each strategy votes with a score. Weights reflect reliability.
        score = 0.0
        votes = []

        def vote(val: float, weight: float, label: str):
            nonlocal score
            score += val * weight
            votes.append((label, val, weight))

        # ── 1. TREND FOLLOWING — EMA Stack ────────────────────────────────
        if e9_v > e21_v > e50_v > e200_v:
            vote(+1, 3, "EMA Stack Bull")
            ema_lbl = "🟢 Full bull stack (9>21>50>200)"
        elif e9_v < e21_v < e50_v < e200_v:
            vote(-1, 3, "EMA Stack Bear")
            ema_lbl = "🔴 Full bear stack (9<21<50<200)"
        elif e9_v > e21_v and e9_p <= e21_p:
            vote(+1, 2, "EMA Golden Cross")
            ema_lbl = "🟢 Golden cross (9/21)"
        elif e9_v < e21_v and e9_p >= e21_p:
            vote(-1, 2, "EMA Death Cross")
            ema_lbl = "🔴 Death cross (9/21)"
        elif e9_v > e21_v:
            vote(+0.5, 2, "EMA Bull")
            ema_lbl = "🟢 Bullish alignment"
        else:
            vote(-0.5, 2, "EMA Bear")
            ema_lbl = "🔴 Bearish alignment"

        # ── 2. MOMENTUM — MACD ────────────────────────────────────────────
        if mhist_v > 0 and mhist_prev <= 0:
            vote(+1, 3, "MACD Bull Cross")
            macd_lbl = "🟢 Bull crossover ⚡"
        elif mhist_v < 0 and mhist_prev >= 0:
            vote(-1, 3, "MACD Bear Cross")
            macd_lbl = "🔴 Bear crossover ⚡"
        elif mhist_v > 0 and mhist_v > macd_hist_s.iloc[-3]:
            vote(+0.7, 3, "MACD Bull Accel")
            macd_lbl = "🟢 Bullish accelerating"
        elif mhist_v < 0 and mhist_v < macd_hist_s.iloc[-3]:
            vote(-0.7, 3, "MACD Bear Accel")
            macd_lbl = "🔴 Bearish accelerating"
        elif mhist_v > 0:
            vote(+0.3, 3, "MACD Bull")
            macd_lbl = "🟢 Bullish"
        else:
            vote(-0.3, 3, "MACD Bear")
            macd_lbl = "🔴 Bearish"

        # ── 3. MEAN REVERSION — RSI ───────────────────────────────────────
        if rsi_v < 25:
            vote(+1, 2.5, "RSI Extreme Oversold")
            rsi_lbl = "🟢 Extreme oversold (<25)"
        elif rsi_v < 35:
            vote(+0.8, 2.5, "RSI Oversold")
            rsi_lbl = "🟢 Oversold"
        elif rsi_v > 75:
            vote(-1, 2.5, "RSI Extreme Overbought")
            rsi_lbl = "🔴 Extreme overbought (>75)"
        elif rsi_v > 65:
            vote(-0.8, 2.5, "RSI Overbought")
            rsi_lbl = "🔴 Overbought"
        elif 45 < rsi_v < 55:
            vote(0, 2.5, "RSI Neutral")
            rsi_lbl = "🟡 Neutral zone"
        elif rsi_v < 50:
            vote(-0.3, 2.5, "RSI Bear Zone")
            rsi_lbl = "🟡 Bearish zone"
        else:
            vote(+0.3, 2.5, "RSI Bull Zone")
            rsi_lbl = "🟡 Bullish zone"

        # ── 4. RSI DIVERGENCE ─────────────────────────────────────────────
        if "Bullish" in div_signal:
            vote(+1, 3, "RSI Bull Divergence")
        elif "Bearish" in div_signal:
            vote(-1, 3, "RSI Bear Divergence")

        # ── 5. BOLLINGER BANDS ────────────────────────────────────────────
        if bb_pos < 0.05:
            vote(+1, 2, "BB Extreme Lower")
            bb_lbl = "🟢 Extreme lower band"
        elif bb_pos < 0.2:
            vote(+0.6, 2, "BB Lower")
            bb_lbl = "🟢 Near lower band"
        elif bb_pos > 0.95:
            vote(-1, 2, "BB Extreme Upper")
            bb_lbl = "🔴 Extreme upper band"
        elif bb_pos > 0.8:
            vote(-0.6, 2, "BB Upper")
            bb_lbl = "🔴 Near upper band"
        else:
            vote(0, 2, "BB Mid")
            bb_lbl = f"🟡 Middle ({bb_pos:.0%})"

        # ── 6. STOCHASTIC ─────────────────────────────────────────────────
        if stk_v < 20 and std_v < 20:
            vote(+1, 2, "Stoch Oversold")
            stoch_lbl = "🟢 Oversold + signal cross"
        elif stk_v > 80 and std_v > 80:
            vote(-1, 2, "Stoch Overbought")
            stoch_lbl = "🔴 Overbought + signal cross"
        elif stk_v < 20:
            vote(+0.5, 2, "Stoch OS")
            stoch_lbl = f"🟢 Oversold ({stk_v:.0f})"
        elif stk_v > 80:
            vote(-0.5, 2, "Stoch OB")
            stoch_lbl = f"🔴 Overbought ({stk_v:.0f})"
        else:
            vote(0, 2, "Stoch Neutral")
            stoch_lbl = f"🟡 Neutral ({stk_v:.0f})"

        # ── 7. WILLIAMS %R ────────────────────────────────────────────────
        if wr_v < -80:
            vote(+0.8, 1.5, "Williams Oversold")
            wr_lbl = f"🟢 Oversold ({wr_v:.0f})"
        elif wr_v > -20:
            vote(-0.8, 1.5, "Williams Overbought")
            wr_lbl = f"🔴 Overbought ({wr_v:.0f})"
        else:
            vote(0, 1.5, "Williams Neutral")
            wr_lbl = f"🟡 Neutral ({wr_v:.0f})"

        # ── 8. CCI ────────────────────────────────────────────────────────
        if cci_v < -100:
            vote(+0.8, 1.5, "CCI Oversold")
            cci_lbl = f"🟢 Oversold ({cci_v:.0f})"
        elif cci_v > 100:
            vote(-0.8, 1.5, "CCI Overbought")
            cci_lbl = f"🔴 Overbought ({cci_v:.0f})"
        elif cci_v > 0:
            vote(+0.3, 1.5, "CCI Bull")
            cci_lbl = f"🟢 Bullish ({cci_v:.0f})"
        else:
            vote(-0.3, 1.5, "CCI Bear")
            cci_lbl = f"🔴 Bearish ({cci_v:.0f})"

        # ── 9. SUPERTREND ─────────────────────────────────────────────────
        if st_dir_v == 1 and st_prev == -1:
            vote(+1, 3, "Supertrend Bull Flip")
            st_lbl = "🟢 Flipped BULLISH ⚡"
        elif st_dir_v == -1 and st_prev == 1:
            vote(-1, 3, "Supertrend Bear Flip")
            st_lbl = "🔴 Flipped BEARISH ⚡"
        elif st_dir_v == 1:
            vote(+0.5, 3, "Supertrend Bull")
            st_lbl = "🟢 Bullish trend"
        else:
            vote(-0.5, 3, "Supertrend Bear")
            st_lbl = "🔴 Bearish trend"

        # ── 10. ICHIMOKU CLOUD ────────────────────────────────────────────
        cloud_valid = not (np.isnan(span_a_v) or np.isnan(span_b_v))
        if cloud_valid:
            if price > cloud_top and tenkan_v > kijun_v:
                vote(+1, 3, "Ichimoku Strong Bull")
                ichi_lbl = "🟢 Above cloud + TK cross"
            elif price < cloud_bot and tenkan_v < kijun_v:
                vote(-1, 3, "Ichimoku Strong Bear")
                ichi_lbl = "🔴 Below cloud + TK cross"
            elif price > cloud_top:
                vote(+0.5, 3, "Ichimoku Bull")
                ichi_lbl = "🟢 Above cloud"
            elif price < cloud_bot:
                vote(-0.5, 3, "Ichimoku Bear")
                ichi_lbl = "🔴 Below cloud"
            else:
                vote(0, 3, "Ichimoku In Cloud")
                ichi_lbl = "🟡 Inside cloud (choppy)"
        else:
            ichi_lbl = "🟡 Insufficient data"

        # ── 11. ADX TREND STRENGTH ────────────────────────────────────────
        adx_lbl = f"{'Strong' if adx_v > 25 else 'Weak'} ({adx_v:.0f})"
        if adx_v > 25:
            # Strong trend — amplify directional score
            if pdi_v > ndi_v:
                vote(+0.5, 2, "ADX Strong Bull")
            else:
                vote(-0.5, 2, "ADX Strong Bear")

        # ── 12. PIVOT POINTS ──────────────────────────────────────────────
        if price > r1:
            vote(-0.5, 1, "Above R1 — Resistance")
            pivot_lbl = f"🔴 Above R1 ({r1:.5f})"
        elif price < s1:
            vote(+0.5, 1, "Below S1 — Support")
            pivot_lbl = f"🟢 Below S1 ({s1:.5f})"
        else:
            pivot_lbl = pivot_signal

        # ── 13. H4 TREND FILTER ───────────────────────────────────────────
        h4_map = {"BULL": (+1,"🟢 Bullish"), "BEAR": (-1,"🔴 Bearish"), "NEUTRAL": (0,"🟡 Neutral")}
        h4_score, h4_lbl = h4_map[trend_h4]
        vote(h4_score, 4, "H4 Trend")  # highest weight

        # ── 14. SESSION MULTIPLIER ────────────────────────────────────────
        score *= 1.15 if in_session else 0.75

        # ── Final Direction ───────────────────────────────────────────────
        direction   = "BUY" if score > 0 else "SELL"
        h4_conflict = (direction == "BUY" and trend_h4 == "BEAR") or \
                      (direction == "SELL" and trend_h4 == "BULL")

        # Count how many strategies agree with direction
        agreeing = sum(1 for _, v, _ in votes if (v > 0 and direction == "BUY") or
                                                  (v < 0 and direction == "SELL"))
        total_votes = len([v for _, v, _ in votes if v != 0])
        agreement_pct = (agreeing / total_votes * 100) if total_votes > 0 else 50

        # Confidence = blend of score magnitude + strategy agreement
        max_possible = sum(abs(w) for _, _, w in votes)
        score_conf   = min((abs(score) / max_possible) * 100, 90) if max_possible > 0 else 30
        final_conf   = round((score_conf * 0.6 + agreement_pct * 0.4), 1)
        final_conf   = max(final_conf, 15)
        if h4_conflict: final_conf *= 0.5

        # ML blend if available
        ml_feats = {
            "rsi": rsi_v, "macd_hist": mhist_v, "bb_pos": bb_pos,
            "ema_diff": (e9_v - e21_v) / price, "atr_norm": atr_v / price,
            "stoch_k": stk_v, "adx": adx_v,
            "h4_bull": h4_score, "in_session": int(in_session),
        }
        ml_conf    = self.ml_confidence(ml_feats)
        final_conf = round((ml_conf * 0.3 + final_conf * 0.7) if ml_conf else final_conf, 1)

        # ── Dynamic SL/TP using ATR ───────────────────────────────────────
        pip     = PIP_SIZE[pair]
        sl_dist = max(SL_PIPS[pair] * pip, atr_v * 1.5)
        rr1, rr2 = 1.5, 2.5

        if direction == "BUY":
            sl  = round(price - sl_dist, 5)
            tp1 = round(price + sl_dist * rr1, 5)
            tp2 = round(price + sl_dist * rr2, 5)
        else:
            sl  = round(price + sl_dist, 5)
            tp1 = round(price - sl_dist * rr1, 5)
            tp2 = round(price - sl_dist * rr2, 5)

        # ── Quality Tag ───────────────────────────────────────────────────
        if final_conf >= 72 and in_session and not h4_conflict and agreeing >= 7:
            quality = "⭐⭐⭐ HIGH"
        elif final_conf >= 55 and not h4_conflict:
            quality = "⭐⭐ MEDIUM"
        elif h4_conflict:
            quality = "⚠️ LOW — H4 conflict"
        else:
            quality = "⭐ LOW"

        # Dominant strategy
        top_vote = sorted(votes, key=lambda x: abs(x[1]*x[2]), reverse=True)
        strategy = top_vote[0][0] if top_vote else "Mixed"

        return {
            "pair":       DISPLAY_NAMES[pair],
            "direction":  direction,
            "entry":      round(price, 5),
            "sl":         sl,
            "tp1":        tp1,
            "tp2":        tp2,
            "confidence": final_conf,
            "quality":    quality,
            "strategy":   strategy,
            "agreement":  f"{agreeing}/{total_votes} strategies",
            "timeframe":  "H1 + H4",
            "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "session":    sess_name,
            "in_session": in_session,
            "h4_trend":   h4_lbl,
            "indicators": {
                "rsi":        rsi_v,
                "rsi_signal": rsi_lbl,
                "divergence": div_signal,
                "macd_signal":macd_lbl,
                "ema_cross":  ema_lbl,
                "bb_position":bb_lbl,
                "stochastic": stoch_lbl,
                "williams_r": wr_lbl,
                "cci":        cci_lbl,
                "supertrend": st_lbl,
                "ichimoku":   ichi_lbl,
                "adx":        adx_lbl,
                "pivot":      pivot_lbl,
                "atr":        round(atr_v, 5),
            },
            "news_sentiment": "Neutral 🟡",
            "ml_used": ml_conf is not None,
        }

    async def get_signal(self, pair: str) -> dict:
        try:
            df_h1 = await self.fetch_ohlcv(pair, "1h", 150)
            await asyncio.sleep(8)
            df_h4 = await self.fetch_ohlcv(pair, "4h", 150)
            if df_h1 is None or len(df_h1) < 60:
                return self._error_signal(pair, "No data from Twelve Data. Check API key.")
            if df_h4 is None or len(df_h4) < 10:
                df_h4 = await self._resample_to_h4(df_h1)
            return self.analyze(df_h1, df_h4, pair)
        except Exception as e:
            return self._error_signal(pair, str(e))

    @staticmethod
    def _error_signal(pair: str, reason: str) -> dict:
        return {
            "pair": DISPLAY_NAMES.get(pair, pair), "direction": "N/A",
            "entry": "N/A", "sl": "N/A", "tp1": "N/A", "tp2": "N/A",
            "confidence": 0, "quality": "❌ Error",
            "strategy": "N/A", "agreement": "0/0",
            "timeframe": "H1+H4",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "session": "Unknown", "in_session": False, "h4_trend": "N/A",
            "indicators": {
                "rsi": 0, "rsi_signal": "❌", "divergence": "❌",
                "macd_signal": "❌", "ema_cross": "❌", "bb_position": "❌",
                "stochastic": "❌", "williams_r": "❌", "cci": "❌",
                "supertrend": "❌", "ichimoku": "❌", "adx": "❌",
                "pivot": "❌", "atr": 0,
            },
            "news_sentiment": f"❌ {reason}",
            "ml_used": False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SCALPING ENGINE — Fast signals on M5/M15 timeframes
# Strategies: Order Flow, Tick Structure, Micro-momentum, VWAP, Spread analysis
# ═══════════════════════════════════════════════════════════════════════════════

class ScalpingEngine:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = "https://api.twelvedata.com"

    async def fetch_scalp_data(self, pair: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """Fetch M5 or M15 candles for scalping analysis."""
        cache_key = f"{pair}_{interval}_scalp"
        now_ts    = _time.time()

        if cache_key in _cache:
            cached_ts, cached_df = _cache[cache_key]
            if now_ts - cached_ts < 60:  # 1 min cache for scalp data
                return cached_df.copy()

        symbol = TD_SYMBOLS.get(pair)
        if not symbol:
            return None

        params = {
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": 100,
            "apikey":     self.api_key,
            "format":     "JSON",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/time_series",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
        except Exception:
            return _cache.get(cache_key, (None, None))[1]

        if data.get("status") == "error" or not data.get("values"):
            return None

        records = []
        for v in data["values"]:
            try:
                records.append({
                    "time":  pd.to_datetime(v["datetime"]),
                    "open":  float(v["open"]),
                    "high":  float(v["high"]),
                    "low":   float(v["low"]),
                    "close": float(v["close"]),
                })
            except Exception:
                continue

        if not records:
            return None

        df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
        _cache[cache_key] = (_time.time(), df)
        return df.copy()

    # ── Scalping Indicators ───────────────────────────────────────────────────

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """VWAP — Volume Weighted Average Price (approximated without volume)."""
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        # Approximate volume with ATR (higher volatility = more activity)
        atr = (df["high"] - df["low"]).rolling(14).mean()
        cum_tp_vol = (tp * atr).cumsum()
        cum_vol    = atr.cumsum()
        return cum_tp_vol / (cum_vol + 1e-9)

    @staticmethod
    def momentum(close: pd.Series, p: int = 10) -> pd.Series:
        """Price momentum — rate of change."""
        return (close - close.shift(p)) / close.shift(p) * 100

    @staticmethod
    def tick_structure(df: pd.DataFrame, lookback: int = 10) -> str:
        """Analyze recent candle structure for order flow direction."""
        recent = df.tail(lookback)
        bull_candles = ((recent["close"] > recent["open"])).sum()
        bear_candles = ((recent["close"] < recent["open"])).sum()
        bull_body    = (recent[recent["close"] > recent["open"]]["close"] -
                        recent[recent["close"] > recent["open"]]["open"]).sum()
        bear_body    = (recent[recent["close"] < recent["open"]]["open"] -
                        recent[recent["close"] < recent["open"]]["close"]).sum()

        if bull_candles >= 7 and bull_body > bear_body * 1.5:
            return "🟢 Strong bullish flow"
        elif bear_candles >= 7 and bear_body > bull_body * 1.5:
            return "🔴 Strong bearish flow"
        elif bull_candles > bear_candles:
            return "🟢 Mild bullish flow"
        elif bear_candles > bull_candles:
            return "🔴 Mild bearish flow"
        return "🟡 Balanced flow"

    @staticmethod
    def spread_volatility(df: pd.DataFrame, p: int = 20) -> dict:
        """Analyze spread/volatility for scalping suitability."""
        spreads = df["high"] - df["low"]
        avg_spread = spreads.rolling(p).mean().iloc[-1]
        curr_spread = spreads.iloc[-1]
        spread_ratio = curr_spread / avg_spread if avg_spread > 0 else 1.0

        if spread_ratio < 0.8:
            condition = "🟢 Tight spread — good for scalping"
        elif spread_ratio > 1.5:
            condition = "🔴 Wide spread — avoid scalping"
        else:
            condition = "🟡 Normal spread"

        return {"avg": round(avg_spread, 5), "current": round(curr_spread, 5),
                "ratio": round(spread_ratio, 2), "condition": condition}

    @staticmethod
    def micro_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
        """Find micro S/R levels from recent swing highs/lows."""
        recent = df.tail(lookback)
        resistance = recent["high"].max()
        support    = recent["low"].min()
        mid        = (resistance + support) / 2
        return {
            "resistance": round(resistance, 5),
            "support":    round(support, 5),
            "mid":        round(mid, 5),
        }

    @staticmethod
    def engulfing(df: pd.DataFrame) -> str:
        """Detect bullish/bearish engulfing candle patterns."""
        if len(df) < 2:
            return "🟡 No pattern"
        prev = df.iloc[-2]; curr = df.iloc[-1]
        prev_bull = prev["close"] > prev["open"]
        curr_bull = curr["close"] > curr["open"]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])

        if not prev_bull and curr_bull and curr_body > prev_body * 1.1:
            return "🟢 Bullish engulfing ⚡"
        if prev_bull and not curr_bull and curr_body > prev_body * 1.1:
            return "🔴 Bearish engulfing ⚡"
        return "🟡 No engulfing"

    @staticmethod
    def pin_bar(df: pd.DataFrame) -> str:
        """Detect pin bars (hammer/shooting star) — reversal signals."""
        c = df.iloc[-1]
        body  = abs(c["close"] - c["open"])
        total = c["high"] - c["low"]
        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]

        if total < 1e-10:
            return "🟡 No pin bar"

        if lower_wick > body * 2 and lower_wick > upper_wick * 2:
            return "🟢 Bullish pin bar (hammer) ⚡"
        if upper_wick > body * 2 and upper_wick > lower_wick * 2:
            return "🔴 Bearish pin bar (shooting star) ⚡"
        return "🟡 No pin bar"

    @staticmethod
    def ema_ribbon(close: pd.Series) -> str:
        """EMA ribbon — multiple EMAs for scalp trend direction."""
        emas = [close.ewm(span=p, adjust=False).mean().iloc[-1]
                for p in [5, 8, 13, 21, 34]]
        if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
            return "🟢 Full bull ribbon"
        if all(emas[i] < emas[i+1] for i in range(len(emas)-1)):
            return "🔴 Full bear ribbon"
        # Check if expanding or contracting
        spread = emas[0] - emas[-1]
        return f"🟢 Bull bias" if spread > 0 else "🔴 Bear bias"

    # ── Scalp Signal Generator ────────────────────────────────────────────────

    async def get_scalp_signal(self, pair: str) -> dict:
        """Generate M5 + M15 scalping signal."""
        try:
            df_m5  = await self.fetch_scalp_data(pair, "5min")
            await asyncio.sleep(8)
            df_m15 = await self.fetch_scalp_data(pair, "15min")

            if df_m5 is None or len(df_m5) < 30:
                return self._error_scalp(pair, "Insufficient M5 data")

            close = df_m5["close"]
            price = close.iloc[-1]

            # Compute indicators
            vwap_v     = self.vwap(df_m5).iloc[-1]
            mom_v      = self.momentum(close).iloc[-1]
            rsi_v      = SignalEngine.rsi(close).iloc[-1]
            tick_flow  = self.tick_structure(df_m5)
            spread     = self.spread_volatility(df_m5)
            micro_sr   = self.micro_support_resistance(df_m5)
            engulf     = self.engulfing(df_m5)
            pin        = self.pin_bar(df_m5)
            ribbon     = self.ema_ribbon(close)
            atr_v      = SignalEngine.atr(df_m5).iloc[-1]

            # M15 trend for confirmation
            m15_trend = "BULL"
            if df_m15 is not None and len(df_m15) >= 21:
                e9_m15  = df_m15["close"].ewm(span=9, adjust=False).mean().iloc[-1]
                e21_m15 = df_m15["close"].ewm(span=21, adjust=False).mean().iloc[-1]
                m15_trend = "BULL" if e9_m15 > e21_m15 else "BEAR"

            # Score
            score = 0

            # VWAP
            vwap_signal = "🟢 Above VWAP" if price > vwap_v else "🔴 Below VWAP"
            score += 1 if price > vwap_v else -1

            # Momentum
            mom_signal = f"🟢 +{mom_v:.3f}%" if mom_v > 0 else f"🔴 {mom_v:.3f}%"
            score += 1 if mom_v > 0.01 else (-1 if mom_v < -0.01 else 0)

            # Tick flow
            score += 2 if "Strong bullish" in tick_flow else \
                     1 if "Mild bullish" in tick_flow else \
                    -2 if "Strong bearish" in tick_flow else \
                    -1 if "Mild bearish" in tick_flow else 0

            # RSI
            score += 1 if rsi_v < 40 else (-1 if rsi_v > 60 else 0)

            # Candle patterns (high weight)
            score += 2 if "Bullish engulfing" in engulf else \
                    -2 if "Bearish engulfing" in engulf else 0
            score += 2 if "Bullish pin" in pin else \
                    -2 if "Bearish pin" in pin else 0

            # EMA ribbon
            score += 1.5 if "bull" in ribbon.lower() else -1.5

            # M15 confirmation
            m15_conf = "🟢 M15 Bullish" if m15_trend == "BULL" else "🔴 M15 Bearish"
            score += 1.5 if m15_trend == "BULL" else -1.5

            direction = "BUY" if score > 0 else "SELL"
            confidence = min(abs(score) / 12 * 100, 90)
            confidence = max(confidence, 15)

            # Tight SL for scalping (0.5x ATR)
            pip     = PIP_SIZE.get(pair, 0.0001)
            sl_dist = max(atr_v * 0.5, pip * 8)
            rr      = 1.2  # tighter RR for scalps

            if direction == "BUY":
                sl  = round(price - sl_dist, 5)
                tp1 = round(price + sl_dist * rr, 5)
                tp2 = round(price + sl_dist * rr * 2, 5)
            else:
                sl  = round(price + sl_dist, 5)
                tp1 = round(price - sl_dist * rr, 5)
                tp2 = round(price - sl_dist * rr * 2, 5)

            # Only scalp if spread is not too wide
            if "Wide" in spread["condition"]:
                confidence *= 0.5
                quality = "⚠️ SKIP — Wide spread"
            elif confidence >= 65:
                quality = "⚡ SCALP HIGH"
            elif confidence >= 45:
                quality = "⚡ SCALP MEDIUM"
            else:
                quality = "⚡ SCALP LOW"

            return {
                "pair":       DISPLAY_NAMES.get(pair, pair),
                "type":       "SCALP",
                "direction":  direction,
                "entry":      round(price, 5),
                "sl":         sl,
                "tp1":        tp1,
                "tp2":        tp2,
                "confidence": round(confidence, 1),
                "quality":    quality,
                "timeframe":  "M5 + M15",
                "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "indicators": {
                    "vwap":       vwap_signal,
                    "momentum":   mom_signal,
                    "tick_flow":  tick_flow,
                    "ribbon":     ribbon,
                    "engulfing":  engulf,
                    "pin_bar":    pin,
                    "m15_trend":  m15_conf,
                    "spread":     spread["condition"],
                    "micro_res":  micro_sr["resistance"],
                    "micro_sup":  micro_sr["support"],
                    "rsi":        round(rsi_v, 1),
                    "atr":        round(atr_v, 5),
                },
            }
        except Exception as e:
            return self._error_scalp(pair, str(e))

    @staticmethod
    def _error_scalp(pair: str, reason: str) -> dict:
        return {
            "pair": DISPLAY_NAMES.get(pair, pair), "type": "SCALP",
            "direction": "N/A", "entry": "N/A", "sl": "N/A",
            "tp1": "N/A", "tp2": "N/A", "confidence": 0,
            "quality": "❌ Error", "timeframe": "M5+M15",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "indicators": {"error": reason},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ARBITRAGE ENGINE — Statistical arbitrage between correlated pairs
# Strategies: Pair correlation, Z-score spread, Cointegration signals
# ═══════════════════════════════════════════════════════════════════════════════

# Highly correlated pair groups for stat arb
ARB_PAIRS = [
    ("EURUSD", "GBPUSD"),   # EUR and GBP highly correlated
    ("AUDUSD", "NZDUSD"),   # AUD and NZD commodity currencies
    ("USDCHF", "EURUSD"),   # Inverse correlation
    ("XAUUSD", "USDCHF"),   # Gold vs CHF safe haven
    ("GBPJPY", "EURJPY"),   # JPY cross pairs
    ("EURUSD", "USDCHF"),   # EUR/CHF relationship
]


class ArbitrageEngine:
    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = "https://api.twelvedata.com"

    async def fetch_close(self, pair: str, outputsize: int = 100) -> Optional[pd.Series]:
        """Fetch closing prices for correlation analysis."""
        cache_key = f"{pair}_1h"
        if cache_key in _cache:
            _, df = _cache[cache_key]
            return df["close"].reset_index(drop=True)

        symbol = TD_SYMBOLS.get(pair)
        if not symbol:
            return None

        params = {
            "symbol": symbol, "interval": "1h",
            "outputsize": outputsize, "apikey": self.api_key, "format": "JSON",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/time_series",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
        except Exception:
            return None

        if data.get("status") == "error" or not data.get("values"):
            return None

        closes = []
        for v in data["values"]:
            try:
                closes.append(float(v["close"]))
            except Exception:
                continue

        if not closes:
            return None

        s = pd.Series(closes[::-1])
        return s

    @staticmethod
    def zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Rolling Z-score of a series."""
        mean = series.rolling(window).mean()
        std  = series.rolling(window).std()
        return (series - mean) / (std + 1e-9)

    @staticmethod
    def correlation(s1: pd.Series, s2: pd.Series, window: int = 30) -> float:
        """Rolling correlation between two series."""
        min_len = min(len(s1), len(s2))
        s1 = s1.iloc[-min_len:].reset_index(drop=True)
        s2 = s2.iloc[-min_len:].reset_index(drop=True)
        return s1.rolling(window).corr(s2).iloc[-1]

    @staticmethod
    def spread_zscore(s1: pd.Series, s2: pd.Series, window: int = 20) -> Tuple[pd.Series, float]:
        """Compute spread between two pairs and its Z-score."""
        min_len = min(len(s1), len(s2))
        s1 = s1.iloc[-min_len:].reset_index(drop=True)
        s2 = s2.iloc[-min_len:].reset_index(drop=True)

        # Normalize both series to same scale
        s1_norm = (s1 - s1.mean()) / (s1.std() + 1e-9)
        s2_norm = (s2 - s2.mean()) / (s2.std() + 1e-9)

        spread = s1_norm - s2_norm
        z      = ArbitrageEngine.zscore(spread, window)
        return spread, z.iloc[-1]

    async def analyze_pair(self, pair_a: str, pair_b: str) -> dict:
        """Analyze statistical arbitrage opportunity between two pairs."""
        s1 = await self.fetch_close(pair_a)
        await asyncio.sleep(8)
        s2 = await self.fetch_close(pair_b)

        if s1 is None or s2 is None or len(s1) < 30 or len(s2) < 30:
            return {"pair_a": pair_a, "pair_b": pair_b, "valid": False,
                    "reason": "Insufficient data"}

        corr = self.correlation(s1, s2)
        spread, zscore_v = self.spread_zscore(s1, s2)

        # Determine signal
        signal      = "NEUTRAL"
        action_a    = "HOLD"
        action_b    = "HOLD"
        confidence  = 0
        description = ""

        abs_corr = abs(corr) if not np.isnan(corr) else 0

        if abs_corr > 0.7:  # Only trade highly correlated pairs
            if zscore_v > 2.0:
                # Spread too wide — expect reversion
                if corr > 0:
                    # Positive correlation: A overperformed B
                    action_a = "SELL"; action_b = "BUY"
                    description = f"{DISPLAY_NAMES[pair_a]} overperformed — expect reversion"
                else:
                    action_a = "BUY"; action_b = "BUY"
                    description = f"Inverse pairs diverged — expect convergence"
                signal     = "TRADE"
                confidence = min((zscore_v - 2.0) * 30 + 50, 85)

            elif zscore_v < -2.0:
                # Spread too narrow — expect reversion
                if corr > 0:
                    action_a = "BUY"; action_b = "SELL"
                    description = f"{DISPLAY_NAMES[pair_b]} overperformed — expect reversion"
                else:
                    action_a = "SELL"; action_b = "SELL"
                    description = f"Inverse pairs converged — expect divergence"
                signal     = "TRADE"
                confidence = min((abs(zscore_v) - 2.0) * 30 + 50, 85)

            elif abs(zscore_v) < 0.5:
                signal      = "NEUTRAL"
                description = "Pairs in equilibrium — no edge"
                confidence  = 0
            else:
                signal      = "WATCH"
                description = f"Spread developing — Z={zscore_v:.2f}"
                confidence  = 30

        else:
            description = f"Low correlation ({corr:.2f}) — skip"
            confidence  = 0

        return {
            "pair_a":      pair_a,
            "pair_b":      pair_b,
            "name_a":      DISPLAY_NAMES.get(pair_a, pair_a),
            "name_b":      DISPLAY_NAMES.get(pair_b, pair_b),
            "correlation": round(corr, 3) if not np.isnan(corr) else 0,
            "zscore":      round(zscore_v, 3),
            "signal":      signal,
            "action_a":    action_a,
            "action_b":    action_b,
            "confidence":  round(confidence, 1),
            "description": description,
            "valid":       True,
            "timestamp":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

    async def scan_all(self) -> list:
        """Scan all arbitrage pairs and return opportunities."""
        results = []
        for pair_a, pair_b in ARB_PAIRS:
            result = await self.analyze_pair(pair_a, pair_b)
            if result.get("valid") and result.get("signal") == "TRADE":
                results.append(result)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
