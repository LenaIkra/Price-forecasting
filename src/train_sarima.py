from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import DataConfig, TrainConfig
from src.metrics import evaluate_regression, save_metrics


class SarimaTrainer:
    """
    SARIMA на всех рядах.

    Логика:
    1. На ограниченной подвыборке рядов подбираем ГЛОБАЛЬНЫЕ параметры SARIMA.
    2. С этими параметрами считаем forecast для всех рядов.
    3. Периодически сохраняем прогресс, чтобы можно было продолжить после прерывания.

    Важная защита:
    - если у отдельного ряда forecast содержит NaN / inf, такой ряд пропускается;
    - перед расчетом итоговых метрик все невалидные строки удаляются.
    """

    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    def _get_series_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = list(self.data_cfg.series_id_cols)

        counts = (
            df.groupby(group_cols)
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
            .reset_index(drop=True)
        )
        return counts

    def _fit_and_forecast_single(
        self,
        train_series: pd.DataFrame,
        test_series: pd.DataFrame,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Обучает SARIMA на одном ряде и возвращает:
        y_true, y_pred

        Если модель не сошлась или прогноз невалиден — возвращает None.
        """
        target_col = self.data_cfg.target_col

        y_train = train_series[target_col].astype(float).values
        y_test = test_series[target_col].astype(float).values

        if len(y_train) < 24 or len(y_test) == 0:
            return None

        try:
            model = SARIMAX(
                y_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=len(y_test))

            y_test = np.asarray(y_test, dtype=float)
            forecast = np.asarray(forecast, dtype=float)

            # Если forecast содержит NaN / inf, такой ряд пропускаем
            mask = np.isfinite(y_test) & np.isfinite(forecast)
            if mask.sum() == 0:
                return None

            y_test = y_test[mask]
            forecast = forecast[mask]

            if len(y_test) == 0:
                return None

            return y_test, forecast

        except Exception:
            return None

    def _split_one_series(
        self,
        df: pd.DataFrame,
        series_key: tuple[str, str],
    ) -> pd.DataFrame:
        store_id, item_id = series_key
        out = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].copy()
        out = out.sort_values(self.data_cfg.date_col)
        return out

    def _search_best_global_params(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """
        Подбор лучших глобальных параметров SARIMA
        на ограниченном наборе рядов.
        """
        warnings.filterwarnings("ignore")

        counts = self._get_series_counts(train_df)
        search_subset = counts.head(self.train_cfg.sarima_search_series_count)

        best_metrics = None
        best_params = None

        print(f"[SARIMA] global search on {len(search_subset)} series")

        for order in self.train_cfg.sarima_orders:
            for seasonal_order in self.train_cfg.sarima_seasonal_orders:
                all_true = []
                all_pred = []

                print(f"[SARIMA] testing order={order}, seasonal_order={seasonal_order}")

                for _, row in search_subset.iterrows():
                    key = (row["store_id"], row["item_id"])

                    train_series = self._split_one_series(train_df, key)
                    test_series = self._split_one_series(test_df, key)

                    result = self._fit_and_forecast_single(
                        train_series=train_series,
                        test_series=test_series,
                        order=order,
                        seasonal_order=seasonal_order,
                    )

                    if result is None:
                        continue

                    y_true, y_pred = result
                    all_true.extend(y_true.tolist())
                    all_pred.extend(y_pred.tolist())

                if len(all_true) == 0:
                    continue

                metrics = evaluate_regression(np.array(all_true), np.array(all_pred))
                print(f"[SARIMA] metrics={metrics}")

                if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
                    best_metrics = metrics
                    best_params = {
                        "order": order,
                        "seasonal_order": seasonal_order,
                    }

        if best_params is None:
            raise RuntimeError("[SARIMA] Failed to find valid global parameters.")

        print(f"[SARIMA] best global params={best_params}")
        return best_params

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metrics_path: Path,
        preds_path: Path,
        do_search: bool = True,
        progress_path: Path | None = None,
    ) -> tuple[dict, dict]:
        warnings.filterwarnings("ignore")

        if progress_path is None:
            progress_path = metrics_path.parent / "sarima_all_series_progress.csv"

        # 1. Подбираем лучшие глобальные параметры
        if do_search:
            best_params = self._search_best_global_params(train_df, test_df)
        else:
            best_params = {
                "order": (1, 1, 1),
                "seasonal_order": (1, 0, 1, 12),
            }

        order = tuple(best_params["order"])
        seasonal_order = tuple(best_params["seasonal_order"])

        counts = self._get_series_counts(train_df)

        # 2. Resume support
        if progress_path.exists():
            done_df = pd.read_csv(progress_path)
            done_keys = set(zip(done_df["store_id"], done_df["item_id"]))
        else:
            done_df = pd.DataFrame(columns=["store_id", "item_id", "status"])
            done_keys = set()

        pred_rows = []
        progress_rows = []

        # Если preds уже есть — подгружаем старые
        if preds_path.exists():
            existing_preds = pd.read_csv(preds_path)
            pred_rows = existing_preds.to_dict("records")

        total_series = len(counts)
        print(f"[SARIMA] running on all series: {total_series}")

        for idx, row in counts.iterrows():
            key = (row["store_id"], row["item_id"])

            if key in done_keys:
                if (idx + 1) % 100 == 0:
                    print(f"[SARIMA] skip progress {idx + 1}/{total_series}")
                continue

            train_series = self._split_one_series(train_df, key)
            test_series = self._split_one_series(test_df, key)

            result = self._fit_and_forecast_single(
                train_series=train_series,
                test_series=test_series,
                order=order,
                seasonal_order=seasonal_order,
            )

            if result is None:
                progress_rows.append({
                    "store_id": key[0],
                    "item_id": key[1],
                    "status": "failed",
                })
            else:
                y_true, y_pred = result

                # Берем только те даты теста, для которых прогноз сохранился после mask
                test_dates = test_series[self.data_cfg.date_col].astype(str).tolist()

                # Если из-за mask длина сократилась, подрежем и даты
                effective_len = min(len(test_dates), len(y_true), len(y_pred))
                test_dates = test_dates[:effective_len]
                y_true = y_true[:effective_len]
                y_pred = y_pred[:effective_len]

                for dt, yt, yp in zip(test_dates, y_true, y_pred):
                    pred_rows.append({
                        "store_id": key[0],
                        "item_id": key[1],
                        "date": dt,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    })

                progress_rows.append({
                    "store_id": key[0],
                    "item_id": key[1],
                    "status": "done",
                })

            # Периодически сохраняем прогресс
            if (idx + 1) % 100 == 0 or (idx + 1) == total_series:
                pd.DataFrame(pred_rows).to_csv(preds_path, index=False, encoding="utf-8-sig")
                pd.DataFrame(progress_rows).to_csv(progress_path, index=False, encoding="utf-8-sig")
                print(f"[SARIMA] progress saved {idx + 1}/{total_series}")

        if len(pred_rows) == 0:
            raise RuntimeError("[SARIMA] No predictions were generated.")

        preds_df = pd.DataFrame(pred_rows)

        # Чистим невалидные строки перед итоговыми метриками
        preds_df["y_true"] = pd.to_numeric(preds_df["y_true"], errors="coerce")
        preds_df["y_pred"] = pd.to_numeric(preds_df["y_pred"], errors="coerce")
        preds_df = preds_df.replace([np.inf, -np.inf], np.nan)
        preds_df = preds_df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)

        if len(preds_df) == 0:
            raise RuntimeError("[SARIMA] All predictions became invalid after cleaning.")

        preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

        print(f"[SARIMA] valid prediction rows after cleaning: {len(preds_df)}")

        metrics = evaluate_regression(preds_df["y_true"].values, preds_df["y_pred"].values)
        save_metrics(metrics, metrics_path)

        params_path = metrics_path.with_name("sarima_best_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=4)

        print(f"[SARIMA] final metrics={metrics}")
        return metrics, best_params