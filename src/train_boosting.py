from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics
from src.models_pkg.boosting_model import PriceBoostingModel, BoostingConfig


class BoostingTrainer:
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

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        model_path: Path,
        metrics_path: Path,
        do_search: bool = False,
        preds_path: Path | None = None,
    ) -> tuple[dict, dict]:
        numeric_cols, categorical_cols = self._split_feature_types(train_df, feature_cols)

        X_train = train_df[feature_cols]
        y_train = train_df[self.data_cfg.target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[self.data_cfg.target_col]

        if do_search:
            search_space = [
                {"max_iter": 200, "learning_rate": 0.03, "max_depth": 6},
                {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
                {"max_iter": 500, "learning_rate": 0.03, "max_depth": 10},
            ]
        else:
            search_space = [{
                "max_iter": self.train_cfg.boosting_max_iter,
                "learning_rate": self.train_cfg.boosting_learning_rate,
                "max_depth": self.train_cfg.boosting_max_depth,
            }]

        best_metrics = None
        best_params = None
        best_model = None
        best_preds = None

        print(f"[Boosting] X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        print(f"[Boosting] numeric_cols={len(numeric_cols)}, categorical_cols={len(categorical_cols)}")

        for params in search_space:
            print(f"[Boosting] testing params={params}")

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
            preds = model.predict(X_test)
            metrics = evaluate_regression(y_test, preds)

            print(f"[Boosting] metrics={metrics}")

            if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
                best_metrics = metrics
                best_params = params
                best_model = model
                best_preds = preds

        best_model.save(model_path)
        save_metrics(best_metrics, metrics_path)

        if preds_path is not None:
            preds_df = test_df[[*self.data_cfg.series_id_cols, self.data_cfg.date_col]].copy()
            preds_df["y_true"] = y_test.values
            preds_df["y_pred"] = best_preds
            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

        return best_metrics, best_params