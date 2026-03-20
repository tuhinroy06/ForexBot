"""
ML Trainer — Train a RandomForest classifier on historical Forex signal features.

Usage:
    python ml_trainer.py

What it does:
1. Downloads ~2 years of H1 OHLCV for all pairs via Alpha Vantage (daily batches)
2. Computes the same indicator features used in signals.py
3. Labels each bar: 1 (BUY won) if price rose by RR1×SL in next N bars before hitting SL
4. Trains a RandomForestClassifier
5. Saves ml_model.pkl — loaded automatically by SignalEngine
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from config import Config

config = Config()
API_KEY = config.ALPHA_VANTAGE_API_KEY
BASE_URL = "https://www.alphavantage.co/query"

PAIRS = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
}

PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}
SL_PIPS  = {"EURUSD": 20,     "GBPUSD": 25,       "USDJPY": 20}
RR1      = 1.5
FORWARD_BARS = 20  # bars to check if TP/SL hit


# ── Data Fetching ─────────────────────────────────────────────────────────────

async def fetch_full_history(pair: str, months: int = 24) -> pd.DataFrame:
    """
    AV free tier: FX_DAILY gives up to 20yr of daily data.
    We use FX_DAILY for ML label generation (sufficient signal).
    """
    from_sym, to_sym = PAIRS[pair]
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "outputsize": "full",
        "apikey": API_KEY,
    }
    print(f"  Fetching {pair} daily history...")
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()

    key = "Time Series FX (Daily)"
    if key not in data:
        print(f"  ⚠ No data for {pair}: {list(data.keys())}")
        return pd.DataFrame()

    records = []
    for ts, v in data[key].items():
        records.append({
            "time":  pd.to_datetime(ts),
            "open":  float(v["1. open"]),
            "high":  float(v["2. high"]),
            "low":   float(v["3. low"]),
            "close": float(v["4. close"]),
        })

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=months)
    df = df[df["time"] >= cutoff].reset_index(drop=True)
    print(f"  ✓ {pair}: {len(df)} daily bars")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # RSI
    d = close.diff()
    g = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
    l = (-d.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi = 100 - 100 / (1 + g / l)

    # MACD
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig

    # EMA
    e9  = close.ewm(span=9,  adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    ema_diff = (e9 - e21) / close

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pos = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ATR
    h, lo, cp = df["high"], df["low"], close.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    atr_norm = atr / close

    # Stochastic
    lo14 = df["low"].rolling(14).min()
    hi14 = df["high"].rolling(14).max()
    stoch_k = 100 * (close - lo14) / (hi14 - lo14 + 1e-9)

    # ADX
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr14 = atr
    pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    ndi = 100 * ndm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    adx = dx.ewm(span=14, adjust=False).mean()

    # H4-like trend: use 20/50 EMA on same series
    e20 = close.ewm(span=20, adjust=False).mean()
    h4_bull = ((close > e20) & (e20 > e50)).astype(float) - ((close < e20) & (e20 < e50)).astype(float)

    # Session proxy: not meaningful on daily, set neutral
    in_session = pd.Series(0.5, index=df.index)

    df2 = df.copy()
    df2["rsi"]        = rsi
    df2["macd_hist"]  = macd_hist
    df2["bb_pos"]     = bb_pos
    df2["ema_diff"]   = ema_diff
    df2["atr_norm"]   = atr_norm
    df2["stoch_k"]    = stoch_k
    df2["adx"]        = adx
    df2["h4_bull"]    = h4_bull
    df2["in_session"] = in_session
    return df2


FEATURE_COLS = ["rsi", "macd_hist", "bb_pos", "ema_diff", "atr_norm", "stoch_k", "adx", "h4_bull", "in_session"]


def label_signals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    For each bar: simulate a BUY trade.
    Label = 1 if TP1 hit before SL in next FORWARD_BARS bars.
    Label = 0 if SL hit first or neither.
    """
    pip   = PIP_SIZE.get(pair, 0.0001)
    sl_d  = SL_PIPS.get(pair, 20) * pip
    tp_d  = sl_d * RR1

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    labels = np.full(len(df), -1, dtype=int)

    for i in range(len(df) - FORWARD_BARS):
        entry = closes[i]
        tp    = entry + tp_d
        sl    = entry - sl_d
        result = 0  # default: loss / no hit
        for j in range(1, FORWARD_BARS + 1):
            if highs[i + j] >= tp:
                result = 1
                break
            if lows[i + j] <= sl:
                result = 0
                break
        labels[i] = result

    df2 = df.copy()
    df2["label"] = labels
    return df2[df2["label"] >= 0]


# ── Training ──────────────────────────────────────────────────────────────────

async def train():
    print("=" * 55)
    print("  Forex ML Trainer — RandomForest + GradientBoosting")
    print("=" * 55)

    all_dfs = []
    pairs = list(PAIRS.keys())

    for i, pair in enumerate(pairs):
        if i > 0:
            print(f"  Waiting 15s for API rate limit...")
            await asyncio.sleep(15)
        df = await fetch_full_history(pair, months=24)
        if df.empty:
            continue
        df = compute_features(df)
        df = label_signals(df, pair)
        df["pair"] = pair
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No data fetched. Check your API key.")
        return

    combined = pd.concat(all_dfs, ignore_index=True).dropna(subset=FEATURE_COLS + ["label"])
    X = combined[FEATURE_COLS].values
    y = combined["label"].values

    print(f"\n📊 Dataset: {len(X)} samples | BUY wins: {y.sum()} ({y.mean()*100:.1f}%)")

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n🌲 Training RandomForest...")
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    rf_scores = cross_val_score(rf_pipe, X, y, cv=tscv, scoring="accuracy")
    print(f"   CV Accuracy: {rf_scores.mean()*100:.1f}% ± {rf_scores.std()*100:.1f}%")

    print("\n🚀 Training GradientBoosting...")
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        ))
    ])
    gb_scores = cross_val_score(gb_pipe, X, y, cv=tscv, scoring="accuracy")
    print(f"   CV Accuracy: {gb_scores.mean()*100:.1f}% ± {gb_scores.std()*100:.1f}%")

    # Pick best model
    if gb_scores.mean() >= rf_scores.mean():
        best_pipe = gb_pipe
        best_name = "GradientBoosting"
        best_score = gb_scores.mean()
    else:
        best_pipe = rf_pipe
        best_name = "RandomForest"
        best_score = rf_scores.mean()

    # Fit on full data
    best_pipe.fit(X, y)
    print(f"\n✅ Best model: {best_name} ({best_score*100:.1f}% CV accuracy)")

    # Classification report on last fold
    splits = list(tscv.split(X))
    _, test_idx = splits[-1]
    X_test, y_test = X[test_idx], y[test_idx]
    y_pred = best_pipe.predict(X_test)
    print("\n📋 Last-fold Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["LOSS", "WIN"]))

    # Feature importance (RF only)
    if best_name == "RandomForest":
        importances = best_pipe.named_steps["clf"].feature_importances_
        print("🔍 Feature Importances:")
        for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"   {feat:<12} {bar} {imp:.3f}")

    # Save
    model_path = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\n💾 Model saved → {model_path}")
    print("   Bot will auto-load this on next start.")


if __name__ == "__main__":
    asyncio.run(train())
