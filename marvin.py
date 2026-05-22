import argparse

import numpy as np
import pandas as pd
import yfinance as yf


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.005, epochs: int = 3000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -20.0, 20.0)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                linear = x @ self.weights + self.bias
            linear = np.nan_to_num(linear, nan=0.0, posinf=50.0, neginf=-50.0)
            preds = self._sigmoid(linear)

            if not np.isfinite(preds).all():
                break

            dw = (x.T @ (preds - y)) / n_samples
            db = float(np.sum(preds - y) / n_samples)

            dw = np.clip(dw, -1.0, 1.0)
            db = float(np.clip(db, -1.0, 1.0))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            self.weights = np.clip(self.weights, -10.0, 10.0)
            self.bias = float(np.clip(self.bias, -10.0, 10.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained")
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -20.0, 20.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            linear = x @ self.weights + self.bias
        linear = np.nan_to_num(linear, nan=0.0, posinf=50.0, neginf=-50.0)
        return self._sigmoid(linear)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return (probs >= 0.5).astype(int)


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    def metrics_for(cls: int) -> tuple[float, float, float, int]:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1, support

    rows = []
    for cls, name in [(0, "DOWN"), (1, "UP")]:
        p, r, f1, s = metrics_for(cls)
        rows.append(f"{name:>8}  precision={p:0.3f}  recall={r:0.3f}  f1={f1:0.3f}  support={s}")

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    rows.append(f"\naccuracy={accuracy:0.3f}")
    return "\n".join(rows)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["Volatility_10"] = data["Return_1d"].rolling(10).std()

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    return data.dropna()


def run(symbol: str, start: str, end: str) -> None:
    prices = yf.download(symbol, start=start, end=end, auto_adjust=False)
    if prices.empty:
        raise ValueError(f"No data returned for {symbol}")

    dataset = build_features(prices)
    feature_cols = [
        "Return_1d",
        "Return_5d",
        "MA_10",
        "MA_50",
        "Volatility_10",
        "RSI_14",
        "Volume",
    ]

    X = dataset[feature_cols].copy()
    y = dataset["Target"].astype(int).values

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0).replace(0, 1)
    X = (X - x_mean) / x_std
    x_values = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)

    split = int(len(dataset) * 0.8)
    X_train, X_test = x_values[:split], x_values[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegressionGD(lr=0.005, epochs=3000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = float(np.mean(y_test == pred))

    latest_features = x_values[-1:].copy()
    latest_signal = model.predict(latest_features)[0]
    latest_prob = float(model.predict_proba(latest_features)[0])

    signal_text = "UP" if latest_signal == 1 else "DOWN"
    print(f"Symbol: {symbol}")
    print(f"Rows used: {len(dataset)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Directional accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(binary_report(y_test, pred))
    print("Latest next-day prediction:")
    print(f"  Signal: {signal_text}")
    print(f"  Probability(UP): {latest_prob:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock direction predictor")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    run(args.symbol.upper(), args.start, args.end)
