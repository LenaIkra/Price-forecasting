from pathlib import Path
import pandas as pd
import numpy as np

from src.metrics import evaluate_regression, save_metrics


class HybridBoostingM5Trainer:

    def run(
        self,
        boosting_preds_path: Path,
        m5_preds_path: Path,
        metrics_path: Path,
        weight_boosting: float = 0.7,
        weight_m5: float = 0.3,
    ) -> dict:

        boosting_df = pd.read_csv(boosting_preds_path)
        m5_df = pd.read_csv(m5_preds_path)

        # предполагаем одинаковый порядок строк
        y_true = boosting_df["y_true"].values

        boosting_pred = boosting_df["y_pred"].values
        m5_pred = m5_df["prediction"].values

        hybrid_pred = (
            weight_boosting * boosting_pred
            + weight_m5 * m5_pred
        )

        metrics = evaluate_regression(y_true, hybrid_pred)

        save_metrics(metrics, metrics_path)

        print(f"[Hybrid Boosting+M5] metrics: {metrics}")

        return metrics