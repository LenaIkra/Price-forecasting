import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.mean(
            np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
        ) * 100
    )


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
    }


def save_metrics(metrics_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)