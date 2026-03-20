"""
Signal Engine v2 — Finnhub API
- 60 req/min free tier (no delays needed)
- Forex candles endpoint: /forex/candle
- Supports H1 + H4 multi-timeframe
- Indicators: RSI, MACD, EMA, BB, ATR, Stochastic, ADX
- ML model scoring (optional)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

# Finnhub forex symbol format: "OANDA:EUR_USD"
FINNHUB_SYMBOLS = {
    "EURUSD": "OANDA:EUR_USD",
    "GBPUSD": "OANDA:GBP_USD",
    "USDJPY": "OANDA:USD_JPY",
    "USDCHF": "OANDA:USD_CHF",
    "AUDUSD": "OANDA:AUD_USD",
    "USDCAD": "OANDA:USD_CAD",
    "NZDUSD": "OANDA:NZD_USD",
    "EURGBP": "OANDA:EUR_GBP",
    "EURJPY": "OANDA:EUR_JPY",
    "GBPJPY": "OANDA:GBP_JPY",
    "AUDJPY": "OANDA:AUD_JPY",
    "CADJPY": "OANDA:CAD_JPY",
    "CHFJPY": "OANDA:CHF_JPY",
    "EURCHF": "OANDA:EUR_CHF",
    "EURAUD": "OANDA:EUR_AUD",
    "EURCAD": "OANDA:EUR_CAD",
    "GBPAUD": "OANDA:GBP_AUD",
    "GBPCAD": "OANDA:GBP_CAD",
    "GBPCHF": "OANDA:GBP_CHF",
    "AUDCAD": "OANDA:AUD_CAD",
    "AUDCHF": "OANDA:AUD_CHF",
    "AUDNZD": "OANDA:AUD_NZD",
    "NZDJPY": "OANDA:NZD_JPY",
    "XAUUSD": "OANDA:XAU_USD",
    "XAGUSD": "OANDA:XAG_USD",
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
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "AUDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01, "NZDJPY": 0.01,
    "XAUUSD": 0.10, "XAGUSD": 0.01,
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


class SignalEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.ml_model = self._load_ml_model()

    # ── Data Fetching ─────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, pair: str, resolution: str = "60") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV from Finnhub forex/candle endpoint.
        resolution: "60" = H1, "240" = H4 (in minutes)
        Returns last 100 candles.
        """
        symbol = FINNHUB_SYMBOLS.get(pair)
        if not symbol:
            return None

        now   = int(time.time())
        # Fetch enough history: 100 candles back
        # H1 = 100 hours, H4 = 400 hours
        hours_back = 100 if resolution == "60" else 400
        from_ts = now - (hours_back * 3600)

        params = {
            "symbol":     symbol,
            "resolution": resolution,
            "from":       from_ts,
            "to":         now,
            "token":      self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/forex/candle",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
        except Exception as e:
            return None

        if data.get("s") != "ok" or not data.get("c"):
            return None

        df = pd.DataFrame({
            "time":  pd.to_datetime(data["t"], unit="s"),
            "open":  data["o"],
            "high":  data["h"],
            "low":   data["l"],
            "close": data["c"],
        })
        return df.sort_values("time").reset_index(drop=True)

    async def _resample_to_h4(self, df_h1: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample H1 dataframe into H4 candles."""
        if df_h1 is None or len(df_h1) < 8:
            return None
        df = df_h1.set_index("time")
        df_h4 = df.resample("4h").agg({
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
        }).dropna().reset_index()
        return df_h4

    # ── Indicators ────────────────────────────────────────────────────────────

    @staticmethod
    def rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
        l = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
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
        return mid + k * std, mid, mid - k * std

    @staticmethod
    def ema(s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
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
        up  = hi.diff()
        dn  = -lo.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        atr_v = (hi - lo).ewm(span=p, adjust=False).mean()
        pdi   = 100 * pdm.ewm(span=p, adjust=False).mean() / (atr_v + 1e-9)
        ndi   = 100 * ndm.ewm(span=p, adjust=False).mean() / (atr_v + 1e-9)
        dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        return dx.ewm(span=p, adjust=False).mean()

    # ── Session Filter ────────────────────────────────────────────────────────

    @staticmethod
    def in_trading_session() -> Tuple[bool, str]:
        hour   = datetime.now(timezone.utc).hour
        active = [n for n, (s, e) in SESSIONS.items() if s <= hour < e]
        return (True, " + ".join(active)) if active else (False, "Asian (low liquidity)")

    # ── H4 Trend ──────────────────────────────────────────────────────────────

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

    # ── ML ────────────────────────────────────────────────────────────────────

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

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze(self, df_h1: pd.DataFrame, df_h4: Optional[pd.DataFrame], pair: str) -> dict:
        close = df_h1["close"]
        price = close.iloc[-1]

        rsi_s             = self.rsi(close)
        _, _, macd_hist_s = self.macd(close)
        bb_up, _, bb_lo   = self.bb(close)
        e9, e21, e50      = self.ema(close, 9), self.ema(close, 21), self.ema(close, 50)
        atr_s             = self.atr(df_h1)
        stk, _            = self.stochastic(df_h1)
        adx_s             = self.adx(df_h1)

        rsi_v       = rsi_s.iloc[-1]
        mhist_v     = macd_hist_s.iloc[-1]
        mhist_prev  = macd_hist_s.iloc[-2]
        bb_range    = bb_up.iloc[-1] - bb_lo.iloc[-1]
        bb_pos      = (price - bb_lo.iloc[-1]) / bb_range if bb_range > 0 else 0.5
        e9_v, e21_v, e50_v = e9.iloc[-1], e21.iloc[-1], e50.iloc[-1]
        e9_p, e21_p        = e9.iloc[-2],  e21.iloc[-2]
        atr_v       = atr_s.iloc[-1]
        stk_v       = stk.iloc[-1]
        adx_v       = adx_s.iloc[-1]

        trend_h4             = self.h4_trend(df_h4)
        in_session, sess_name = self.in_trading_session()

        # ── Score ──────────────────────────────────────────────────────────
        score = 0; max_score = 0

        def add(val, w):
            nonlocal score, max_score
            score += val * w; max_score += w

        # RSI
        rsi_lbl = ("🟢 Oversold" if rsi_v < 30 else "🟡 Below mid" if rsi_v < 45
                   else "🔴 Overbought" if rsi_v > 70 else "🟡 Above mid")
        add(+2 if rsi_v < 30 else +1 if rsi_v < 45 else -2 if rsi_v > 70 else -1 if rsi_v > 55 else 0, 2)

        # MACD
        if mhist_v > 0 and mhist_prev <= 0:   macd_lbl = "🟢 Bull crossover"; add(+3, 3)
        elif mhist_v < 0 and mhist_prev >= 0: macd_lbl = "🔴 Bear crossover"; add(-3, 3)
        elif mhist_v > 0:                      macd_lbl = "🟢 Bullish";        add(+1, 3)
        else:                                  macd_lbl = "🔴 Bearish";        add(-1, 3)

        # EMA
        if   e9_v > e21_v and e9_p <= e21_p:  ema_lbl = "🟢 Golden cross";   add(+2, 2)
        elif e9_v < e21_v and e9_p >= e21_p:  ema_lbl = "🔴 Death cross";    add(-2, 2)
        elif e9_v > e21_v > e50_v:            ema_lbl = "🟢 Bull alignment"; add(+1, 2)
        elif e9_v < e21_v < e50_v:            ema_lbl = "🔴 Bear alignment"; add(-1, 2)
        else:                                  ema_lbl = "🟡 Mixed";           add(0,  2)

        # BB
        if   bb_pos < 0.15: bb_lbl = "🟢 Lower band"; add(+1, 1)
        elif bb_pos > 0.85: bb_lbl = "🔴 Upper band"; add(-1, 1)
        else:               bb_lbl = f"🟡 Mid {bb_pos:.0%}"; add(0, 1)

        # Stoch
        if   stk_v < 20: stoch_lbl = "🟢 Oversold";   add(+1, 1)
        elif stk_v > 80: stoch_lbl = "🔴 Overbought"; add(-1, 1)
        else:            stoch_lbl = f"🟡 {stk_v:.0f}"; add(0, 1)

        # ADX boost
        adx_lbl = f"{'Strong' if adx_v > 25 else 'Weak'} ({adx_v:.0f})"
        if adx_v > 25: score *= 1.2

        # H4 trend (weight 4)
        h4_map = {"BULL": (+1,"🟢 Bullish"), "BEAR": (-1,"🔴 Bearish"), "NEUTRAL": (0,"🟡 Neutral")}
        h4_score, h4_lbl = h4_map[trend_h4]
        add(h4_score * 4, 4)

        # Session multiplier
        score *= 1.15 if in_session else 0.70

        direction   = "BUY" if score > 0 else "SELL"
        h4_conflict = (direction == "BUY" and trend_h4 == "BEAR") or \
                      (direction == "SELL" and trend_h4 == "BULL")

        raw_conf = min(abs(score) / max_score * 100, 95) if max_score > 0 else 50
        raw_conf = max(raw_conf, 40)
        if h4_conflict: raw_conf *= 0.5

        # ML blend
        ml_feats = {
            "rsi": rsi_v, "macd_hist": mhist_v, "bb_pos": bb_pos,
            "ema_diff": (e9_v - e21_v) / price, "atr_norm": atr_v / price,
            "stoch_k": stk_v, "adx": adx_v,
            "h4_bull": h4_score, "in_session": int(in_session),
        }
        ml_conf    = self.ml_confidence(ml_feats)
        final_conf = round((ml_conf * 0.5 + raw_conf * 0.5) if ml_conf else raw_conf, 1)

        # SL / TP
        pip     = PIP_SIZE[pair]
        sl_dist = max(SL_PIPS[pair] * pip, atr_v * 1.2)
        rr1, rr2 = 1.5, 2.5

        if direction == "BUY":
            sl  = round(price - sl_dist, 5)
            tp1 = round(price + sl_dist * rr1, 5)
            tp2 = round(price + sl_dist * rr2, 5)
        else:
            sl  = round(price + sl_dist, 5)
            tp1 = round(price - sl_dist * rr1, 5)
            tp2 = round(price - sl_dist * rr2, 5)

        # Quality
        if final_conf >= 75 and in_session and not h4_conflict: quality = "⭐⭐⭐ HIGH"
        elif final_conf >= 60 and not h4_conflict:              quality = "⭐⭐ MEDIUM"
        elif h4_conflict:                                        quality = "⚠️ LOW — H4 conflict"
        else:                                                    quality = "⭐ LOW"

        return {
            "pair": DISPLAY_NAMES[pair], "direction": direction,
            "entry": round(price, 5), "sl": sl, "tp1": tp1, "tp2": tp2,
            "confidence": final_conf, "quality": quality,
            "timeframe": "H1 + H4",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "session": sess_name, "in_session": in_session, "h4_trend": h4_lbl,
            "indicators": {
                "rsi": rsi_v, "rsi_signal": rsi_lbl,
                "macd_signal": macd_lbl, "ema_cross": ema_lbl,
                "bb_position": bb_lbl, "stochastic": stoch_lbl,
                "adx": adx_lbl, "atr": round(atr_v, 5),
            },
            "news_sentiment": "Neutral 🟡",
            "ml_used": ml_conf is not None,
        }

    async def get_signal(self, pair: str) -> dict:
        try:
            # Fetch H1 and H4 concurrently — Finnhub supports 60 req/min, no delay needed
            df_h1, df_h4_raw = await asyncio.gather(
                self.fetch_ohlcv(pair, "60"),
                self.fetch_ohlcv(pair, "60"),  # fetch extra H1 to resample as H4
            )
            if df_h1 is None or len(df_h1) < 30:
                return self._error_signal(pair, "No data from Finnhub. Check API key.")
            df_h4 = await self._resample_to_h4(df_h4_raw)
            return self.analyze(df_h1, df_h4, pair)
        except Exception as e:
            return self._error_signal(pair, str(e))

    @staticmethod
    def _error_signal(pair: str, reason: str) -> dict:
        return {
            "pair": DISPLAY_NAMES.get(pair, pair), "direction": "N/A",
            "entry": "N/A", "sl": "N/A", "tp1": "N/A", "tp2": "N/A",
            "confidence": 0, "quality": "❌ Error",
            "timeframe": "H1+H4",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "session": "Unknown", "in_session": False, "h4_trend": "N/A",
            "indicators": {
                "rsi": 0, "rsi_signal": "❌", "macd_signal": "❌",
                "ema_cross": "❌", "bb_position": "❌",
                "stochastic": "❌", "adx": "❌", "atr": 0,
            },
            "news_sentiment": f"❌ {reason}",
            "ml_used": False,
        }
