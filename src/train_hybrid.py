from pathlib import Path
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics
from src.models_pkg.boosting_model import PriceBoostingModel, BoostingConfig
from statsmodels.tsa.statespace.sarimax import SARIMAX


class HybridTrainer:
    """
    Гибрид SARIMA + Boosting.

    Логика:
    1. Делим train на inner_train и inner_val
    2. SARIMA прогнозирует на inner_val
    3. Boosting прогнозирует на inner_val
    4. Подбираем лучший alpha на validation
    5. Переобучаем на полном train и прогнозируем test
    """

    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

    def _split_feature_types(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        numeric_cols = [c for c in feature_cols if is_numeric_dtype(df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]
        return numeric_cols, categorical_cols

    def _time_split_train_val(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Делим train на:
        - inner_train
        - inner_val

        Это нужно, чтобы подобрать лучший alpha честно.
        """
        unique_dates = sorted(train_df[self.data_cfg.date_col].dropna().unique())
        split_point = unique_dates[-self.train_cfg.hybrid_val_weeks]

        inner_train = train_df[train_df[self.data_cfg.date_col] < split_point].copy()
        inner_val = train_df[train_df[self.data_cfg.date_col] >= split_point].copy()
        return inner_train, inner_val

    def _fit_boosting_predict(
        self,
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        feature_cols: list[str],
        boosting_params: dict | None = None,
    ) -> np.ndarray:
        """
        Обучает boosting на train_df и прогнозирует pred_df.
        """
        numeric_cols, categorical_cols = self._split_feature_types(train_df, feature_cols)

        if boosting_params is None:
            boosting_params = {
                "max_iter": self.train_cfg.boosting_max_iter,
                "learning_rate": self.train_cfg.boosting_learning_rate,
                "max_depth": self.train_cfg.boosting_max_depth,
            }

        model = PriceBoostingModel(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            cfg=BoostingConfig(
                max_iter=boosting_params["max_iter"],
                learning_rate=boosting_params["learning_rate"],
                max_depth=boosting_params["max_depth"],
                random_state=self.train_cfg.random_state,
            ),
        )

        model.fit(train_df[feature_cols], train_df[self.data_cfg.target_col])
        preds = model.predict(pred_df[feature_cols])
        return preds

    def _sarima_forecast_all_series(
        self,
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ) -> pd.DataFrame:
        """
        Прогноз SARIMA по всем рядам из pred_df.
        """
        pred_rows = []
        group_cols = list(self.data_cfg.series_id_cols)
        target_col = self.data_cfg.target_col
        date_col = self.data_cfg.date_col

        series_keys = (
            pred_df[group_cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        for key in series_keys:
            store_id, item_id = key

            tr = train_df[
                (train_df["store_id"] == store_id) & (train_df["item_id"] == item_id)
            ].sort_values(date_col)

            te = pred_df[
                (pred_df["store_id"] == store_id) & (pred_df["item_id"] == item_id)
            ].sort_values(date_col)

            if len(tr) < 24 or len(te) == 0:
                continue

            y_train = tr[target_col].astype(float).values

            try:
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                forecast = fitted.forecast(steps=len(te))

                forecast = np.asarray(forecast, dtype=float)
                y_true = te[target_col].astype(float).values

                mask = np.isfinite(forecast) & np.isfinite(y_true)
                if mask.sum() == 0:
                    continue

                te = te.iloc[:len(mask)].copy()
                te = te.loc[mask].copy()
                forecast = forecast[mask]

                for dt, yt, yp in zip(te[date_col], te[target_col].values, forecast):
                    pred_rows.append({
                        "store_id": store_id,
                        "item_id": item_id,
                        "date": dt,
                        "y_true": float(yt),
                        "sarima_pred": float(yp),
                    })
            except Exception:
                continue

        return pd.DataFrame(pred_rows)

    def _search_best_hybrid_params(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        sarima_best_params: dict,
    ) -> dict:
        """
        Подбираем:
        - boosting params внутри гибрида
        - alpha
        по validation части.
        """
        inner_train, inner_val = self._time_split_train_val(train_df)

        order = tuple(sarima_best_params["order"])
        seasonal_order = tuple(sarima_best_params["seasonal_order"])

        boosting_grid = [
            {"max_iter": 200, "learning_rate": 0.03, "max_depth": 6},
            {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
            {"max_iter": 500, "learning_rate": 0.03, "max_depth": 10},
        ]

        best_result = None

        for boosting_params in boosting_grid:
            boosting_val_pred = self._fit_boosting_predict(
                inner_train, inner_val, feature_cols, boosting_params=boosting_params
            )

            val_pred_df = inner_val[
                [*self.data_cfg.series_id_cols, self.data_cfg.date_col, self.data_cfg.target_col]
            ].copy()
            val_pred_df["boosting_pred"] = boosting_val_pred

            sarima_val_df = self._sarima_forecast_all_series(
                train_df=inner_train,
                pred_df=inner_val,
                order=order,
                seasonal_order=seasonal_order,
            )

            val_merged = val_pred_df.merge(
                sarima_val_df,
                on=["store_id", "item_id", "date"],
                how="inner",
            )

            if len(val_merged) == 0:
                continue

            for alpha in self.train_cfg.hybrid_alpha_grid:
                final_val_pred = (
                    alpha * val_merged["boosting_pred"].values
                    + (1 - alpha) * val_merged["sarima_pred"].values
                )

                metrics = evaluate_regression(
                    val_merged[self.data_cfg.target_col].values,
                    final_val_pred,
                )

                print(
                    f"[Hybrid] boosting_params={boosting_params} | alpha={alpha} | metrics={metrics}"
                )

                candidate = {
                    "alpha": alpha,
                    "boosting_params": boosting_params,
                    "metrics": metrics,
                }

                if best_result is None or metrics["RMSE"] < best_result["metrics"]["RMSE"]:
                    best_result = candidate

        if best_result is None:
            raise RuntimeError("[Hybrid] Could not find valid hybrid configuration.")

        print(f"[Hybrid] best_result={best_result}")
        return best_result

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tabular_feature_cols: list[str],
        metrics_path: Path,
        sarima_best_params: dict,
        preds_path: Path | None = None,
    ) -> dict:
        """
        Финальный запуск гибрида:
        - подбираем alpha и boosting params на validation
        - считаем test
        """
        order = tuple(sarima_best_params["order"])
        seasonal_order = tuple(sarima_best_params["seasonal_order"])

        # 1. Подбор параметров гибрида
        best_result = self._search_best_hybrid_params(
            train_df=train_df,
            feature_cols=tabular_feature_cols,
            sarima_best_params=sarima_best_params,
        )

        best_alpha = best_result["alpha"]
        best_boosting_params = best_result["boosting_params"]

        # 2. Boosting на полном train
        boosting_test_pred = self._fit_boosting_predict(
            train_df=train_df,
            pred_df=test_df,
            feature_cols=tabular_feature_cols,
            boosting_params=best_boosting_params,
        )

        test_pred_df = test_df[
            [*self.data_cfg.series_id_cols, self.data_cfg.date_col, self.data_cfg.target_col]
        ].copy()
        test_pred_df["boosting_pred"] = boosting_test_pred

        # 3. SARIMA на полном train
        sarima_test_df = self._sarima_forecast_all_series(
            train_df=train_df,
            pred_df=test_df,
            order=order,
            seasonal_order=seasonal_order,
        )

        merged = test_pred_df.merge(
            sarima_test_df,
            on=["store_id", "item_id", "date"],
            how="left",
        )

        # Если SARIMA для какого-то ряда не сработала, используем boosting как fallback
        merged["sarima_pred"] = merged["sarima_pred"].fillna(merged["boosting_pred"])

        merged["final_pred"] = (
            best_alpha * merged["boosting_pred"]
            + (1 - best_alpha) * merged["sarima_pred"]
        )

        # Чистим возможные NaN
        merged = merged.dropna(subset=[self.data_cfg.target_col, "final_pred"]).reset_index(drop=True)

        metrics = evaluate_regression(
            merged[self.data_cfg.target_col].values,
            merged["final_pred"].values,
        )
        save_metrics(metrics, metrics_path)

        if preds_path is not None:
            merged.to_csv(preds_path, index=False, encoding="utf-8-sig")

        params_path = metrics_path.with_name("hybrid_sarima_boosting_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_alpha": best_alpha,
                    "boosting_params": best_boosting_params,
                    "sarima_order": order,
                    "sarima_seasonal_order": seasonal_order,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(f"[Hybrid] final metrics={metrics}")
        return metrics