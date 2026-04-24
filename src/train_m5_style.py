from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics
from src.models_pkg.boosting_model import PriceBoostingModel, BoostingConfig


class M5StyleTrainer:
    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

    def _split_feature_types(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        X = df[feature_cols].copy()

        numeric_cols = [c for c in feature_cols if is_numeric_dtype(X[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        print(f"[M5-style] numeric_cols={len(numeric_cols)}, categorical_cols={len(categorical_cols)}")
        return numeric_cols, categorical_cols

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        model_path: Path,
        metrics_path: Path,
        do_search: bool = False,
    ) -> tuple[dict, dict]:
        numeric_cols, categorical_cols = self._split_feature_types(train_df, feature_cols)

        X_train = train_df[feature_cols]
        y_train = np.log1p(train_df[self.data_cfg.target_col].clip(lower=0))

        X_test = test_df[feature_cols]
        y_test = test_df[self.data_cfg.target_col].values

        if do_search:
            search_space = [
                {"max_iter": 300, "learning_rate": 0.03, "max_depth": 8},
                {"max_iter": 500, "learning_rate": 0.03, "max_depth": 10},
                {"max_iter": 700, "learning_rate": 0.02, "max_depth": 12},
            ]
        else:
            search_space = [{"max_iter": 500, "learning_rate": 0.03, "max_depth": 10}]

        best_metrics = None
        best_params = None
        best_model = None

        for params in search_space:
            print(f"[M5-style] testing params={params}")

            model = PriceBoostingModel(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                cfg=BoostingConfig(
                    max_iter=params["max_iter"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    random_state=self.train_cfg.random_state,
                ),
            )

            model.fit(X_train, y_train)
            pred_log = model.predict(X_test)
            preds = np.expm1(pred_log)

            metrics = evaluate_regression(y_test, preds)

            print(f"[M5-style] metrics={metrics}")

            if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
                best_metrics = metrics
                best_params = params
                best_model = model

        best_model.save(model_path)
        save_metrics(best_metrics, metrics_path)

        return best_metrics, best_params