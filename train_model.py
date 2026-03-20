"""
ML Trainer — train_model.py
Trains a GradientBoostingClassifier on signal features.

Usage:
    python train_model.py                        # synthetic data
    python train_model.py --csv my_trades.csv    # real trade log
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

FEATURES = ["rsi", "macd_hist", "bb_pos", "ema_diff", "atr_norm", "stoch_k", "adx", "h4_bull", "in_session"]


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        rsi       = rng.uniform(10, 90)
        macd_hist = rng.normal(0, 0.001)
        bb_pos    = rng.uniform(0, 1)
        ema_diff  = rng.normal(0, 0.002)
        atr_norm  = rng.uniform(0.0001, 0.003)
        stoch_k   = rng.uniform(0, 100)
        adx       = rng.uniform(10, 50)
        h4_bull   = rng.choice([-1, 0, 1])
        in_sess   = rng.choice([0, 1], p=[0.35, 0.65])

        win_prob = 0.50
        if rsi < 30:                         win_prob += 0.08
        elif rsi > 70:                       win_prob += 0.06
        else:                                win_prob -= 0.04
        if abs(macd_hist) > 0.0005:          win_prob += 0.05
        if bb_pos < 0.15 or bb_pos > 0.85:  win_prob += 0.05
        if h4_bull != 0:                     win_prob += 0.12
        else:                                win_prob -= 0.08
        if in_sess:                          win_prob += 0.07
        else:                                win_prob -= 0.10
        if adx > 25:                         win_prob += 0.06
        if abs(ema_diff) > 0.001:            win_prob += 0.04

        win_prob = max(0.15, min(0.90, win_prob))
        rows.append({
            "rsi": rsi, "macd_hist": macd_hist, "bb_pos": bb_pos,
            "ema_diff": ema_diff, "atr_norm": atr_norm, "stoch_k": stoch_k,
            "adx": adx, "h4_bull": h4_bull, "in_session": in_sess,
            "result": int(rng.random() < win_prob),
        })
    return pd.DataFrame(rows)


def train(df: pd.DataFrame, output_path: str = "ml_model.pkl"):
    X = df[FEATURES].values
    y = df["result"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42,
        )),
    ])

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nTest Set Report:")
    print(classification_report(y_test, y_pred, target_names=["SL Hit", "TP Hit"]))

    importances = model.named_steps["clf"].feature_importances_
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:<14} {'█' * int(imp * 50)} {imp:.3f}")

    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved to {output_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str, default=None)
    parser.add_argument("--output",  type=str, default="ml_model.pkl")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    if args.csv:
        print(f"Loading real data from {args.csv}...")
        df = pd.read_csv(args.csv)[FEATURES + ["result"]].dropna()
        print(f"Loaded {len(df)} samples.")
    else:
        print(f"Generating {args.samples} synthetic samples...")
        df = generate_synthetic_data(args.samples)
        print(f"Win rate: {df['result'].mean():.1%}")

    train(df, args.output)


if __name__ == "__main__":
    main()
