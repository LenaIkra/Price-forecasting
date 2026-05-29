from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.metrics import evaluate_regression, save_metrics


class HybridBoostingM5Trainer:
    """
    Гибридная модель Boosting + M5-style.

    Модель строит взвешенное среднее прогнозов двух уже обученных моделей:
    - Gradient Boosting;
    - M5-style модель.

    Вес подбирается по сетке от 0 до 1 с шагом 0.01.
    Основная метрика для выбора веса — RMSE.
    """

    @staticmethod
    def _get_prediction_column(df: pd.DataFrame) -> str:
        """
        Определяет название колонки с прогнозом.

        В разных трейнерах прогнозы могли сохраняться под разными именами:
        - y_pred;
        - prediction.
        """
        if "y_pred" in df.columns:
            return "y_pred"

        if "prediction" in df.columns:
            return "prediction"

        raise KeyError(
            "В файле с прогнозами не найдена колонка с прогнозом. "
            "Ожидалась колонка 'y_pred' или 'prediction'."
        )

    def run(
        self,
        boosting_preds_path: Path,
        m5_preds_path: Path,
        metrics_path: Path,
        preds_path: Path | None = None,
        params_path: Path | None = None,
    ) -> tuple[dict, dict]:
        """
        Запускает расчет гибридной модели.

        Parameters
        ----------
        boosting_preds_path:
            Путь к predictions.csv модели Gradient Boosting.

        m5_preds_path:
            Путь к predictions.csv модели M5-style.

        metrics_path:
            Путь для сохранения metrics.json.

        preds_path:
            Путь для сохранения predictions.csv гибридной модели.

        params_path:
            Путь для сохранения best_params.json.

        Returns
        -------
        tuple[dict, dict]
            Метрики гибридной модели и параметры лучшего веса.
        """

        boosting_df = pd.read_csv(boosting_preds_path)
        m5_df = pd.read_csv(m5_preds_path)

        # Приводим даты к одному типу, чтобы merge не падал.
        boosting_df["date"] = pd.to_datetime(boosting_df["date"])
        m5_df["date"] = pd.to_datetime(m5_df["date"])

        boosting_pred_col = self._get_prediction_column(boosting_df)
        m5_pred_col = self._get_prediction_column(m5_df)

        # Склеиваем прогнозы по ключам, а не по порядку строк.
        merge_cols = ["store_id", "item_id", "date"]

        df = boosting_df[
            merge_cols + ["y_true", boosting_pred_col]
        ].rename(
            columns={boosting_pred_col: "boosting_pred"}
        ).merge(
            m5_df[merge_cols + [m5_pred_col]].rename(
                columns={m5_pred_col: "m5_pred"}
            ),
            on=merge_cols,
            how="inner",
        )

        if df.empty:
            raise ValueError(
                "После объединения прогнозов Boosting и M5-style не осталось строк. "
                "Проверь, что predictions.csv построены на одной и той же тестовой выборке."
            )

        y_true = df["y_true"].values
        boosting_pred = df["boosting_pred"].values
        m5_pred = df["m5_pred"].values

        best_metrics = None
        best_params = None
        best_preds = None

        # Подбираем вес Boosting.
        # Вес M5-style считается как 1 - weight_boosting.
        for weight_boosting in np.linspace(0, 1, 101):
            weight_m5 = 1 - weight_boosting

            hybrid_pred = (
                weight_boosting * boosting_pred
                + weight_m5 * m5_pred
            )

            metrics = evaluate_regression(y_true, hybrid_pred)

            if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
                best_metrics = metrics
                best_params = {
                    "weight_boosting": float(weight_boosting),
                    "weight_m5": float(weight_m5),
                    "selection_metric": "RMSE",
                    "grid_step": 0.01,
                    "n_observations": int(len(df)),
                }
                best_preds = hybrid_pred

        # Сохраняем метрики.
        save_metrics(best_metrics, metrics_path)

        # Сохраняем параметры лучшего веса.
        if params_path is not None:
            params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_params, f, ensure_ascii=False, indent=4)

        # Сохраняем прогнозы гибрида.
        if preds_path is not None:
            preds_path.parent.mkdir(parents=True, exist_ok=True)

            preds_df = df[merge_cols + ["y_true"]].copy()
            preds_df["boosting_pred"] = boosting_pred
            preds_df["m5_pred"] = m5_pred
            preds_df["prediction"] = best_preds

            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

        print(f"[Hybrid Boosting+M5] best params: {best_params}")
        print(f"[Hybrid Boosting+M5] metrics: {best_metrics}")

        return best_metrics, best_params