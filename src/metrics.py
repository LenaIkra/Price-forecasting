import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    Метрика менее чувствительна к малым значениям,
    чем MAPE, поэтому лучше подходит для ритейл-данных.
    """
    denominator = np.sum(np.abs(y_true))

    if denominator < eps:
        return float("nan")

    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Рассчитывает метрики качества регрессионного прогноза.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("Нет корректных значений для расчета метрик.")

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE": wape(y_true, y_pred),
    }


def save_metrics(metrics_dict: dict, path: Path) -> None:
    """
    Сохраняет метрики в JSON-файл.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)