from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics
from src.models_pkg.boosting_model import PriceBoostingModel, BoostingConfig


try:
    import optuna
except ImportError:
    optuna = None


class BoostingTrainer:
    """
    Тренер модели градиентного бустинга.

    В экспериментальной части ВКР градиентный бустинг используется как
    самостоятельная ML-модель для прогнозирования цены на основе заранее
    сформированного признакового пространства.

    В отличие от M5-style модели, здесь не добавляется отдельная M5-логика
    с логарифмированием целевой переменной и дополнительными иерархическими
    лагами. Это важно для методологического разделения моделей:

    - градиентный бустинг показывает качество классической табличной ML-модели;
    - M5-style модель дополнительно адаптирует практики M5: log-target,
      иерархические признаки и глобальную модель по множеству рядов.

    Для повышения качества и снижения влияния ручного выбора параметров
    в градиентном бустинге используется подбор гиперпараметров. Если Optuna
    не установлена, модель переходит к ограниченному ручному перебору.
    """

    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

    def _split_feature_types(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Делит признаки на числовые и категориальные.

        Такое разделение требуется для PriceBoostingModel: числовые признаки
        используются напрямую, а категориальные признаки кодируются внутри
        модели через существующую логику проекта.
        """
        numeric_cols = [c for c in feature_cols if is_numeric_dtype(df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        return numeric_cols, categorical_cols

    def _train_valid_split(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Формирует validation-выборку по времени.

        Для временных рядов нельзя использовать случайное перемешивание,
        поскольку оно приводит к утечке будущей информации в обучение.
        Поэтому validation-период берется из последних дат обучающей выборки.
        """
        date_col = self.data_cfg.date_col

        sorted_df = train_df.sort_values(date_col).reset_index(drop=True)
        unique_dates = sorted_df[date_col].drop_duplicates().sort_values().to_list()

        if len(unique_dates) < 5:
            raise ValueError(
                "Недостаточно уникальных дат для формирования "
                "time-based validation в BoostingTrainer."
            )

        valid_size = max(1, int(len(unique_dates) * 0.15))
        valid_dates = set(unique_dates[-valid_size:])

        inner_train = sorted_df[~sorted_df[date_col].isin(valid_dates)].copy()
        inner_valid = sorted_df[sorted_df[date_col].isin(valid_dates)].copy()

        return inner_train.reset_index(drop=True), inner_valid.reset_index(drop=True)

    def _get_default_params(self) -> dict:
        """
        Базовая конфигурация градиентного бустинга.

        Эти параметры используются, если подбор гиперпараметров отключен
        или если Optuna недоступна.
        """
        return {
            "max_iter": self.train_cfg.boosting_max_iter,
            "learning_rate": self.train_cfg.boosting_learning_rate,
            "max_depth": self.train_cfg.boosting_max_depth,
        }

    def _manual_search(
        self,
        inner_train_df: pd.DataFrame,
        inner_valid_df: pd.DataFrame,
        feature_cols: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> dict:
        """
        Ограниченный ручной перебор параметров.

        Этот режим оставлен как резервный вариант на случай, если Optuna
        недоступна в среде запуска. Он также делает эксперимент воспроизводимым
        без дополнительных зависимостей.
        """
        search_space = [
            {"max_iter": 200, "learning_rate": 0.03, "max_depth": 6},
            {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
            {"max_iter": 500, "learning_rate": 0.03, "max_depth": 10},
            {"max_iter": 700, "learning_rate": 0.02, "max_depth": 10},
        ]

        X_train = inner_train_df[feature_cols]
        y_train = inner_train_df[self.data_cfg.target_col]

        X_valid = inner_valid_df[feature_cols]
        y_valid = inner_valid_df[self.data_cfg.target_col]

        best_params = None
        best_rmse = np.inf

        for params in search_space:
            print(f"[Boosting] manual search params={params}")

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
            valid_preds = model.predict(X_valid)

            rmse = float(np.sqrt(np.mean((y_valid.values - valid_preds) ** 2)))

            print(f"[Boosting] manual search RMSE={rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

        best_params = dict(best_params)
        best_params["search_method"] = "manual_grid"
        best_params["validation_rmse"] = best_rmse

        return best_params

    def _tune_params_with_optuna(
        self,
        inner_train_df: pd.DataFrame,
        inner_valid_df: pd.DataFrame,
        feature_cols: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> dict:
        """
        Подбирает гиперпараметры градиентного бустинга с помощью Optuna.

        Optuna используется для автоматизации выбора параметров модели.
        Целевая функция минимизирует RMSE на валидационном периоде,
        сформированном по времени.
        """
        if optuna is None:
            warnings.warn(
                "Optuna is not installed. BoostingTrainer will use manual search.",
                RuntimeWarning,
            )
            return self._manual_search(
                inner_train_df=inner_train_df,
                inner_valid_df=inner_valid_df,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )

        X_train = inner_train_df[feature_cols]
        y_train = inner_train_df[self.data_cfg.target_col]

        X_valid = inner_valid_df[feature_cols]
        y_valid = inner_valid_df[self.data_cfg.target_col]

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "max_iter": trial.suggest_int("max_iter", 200, 900),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.08,
                    log=True,
                ),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
            }

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
            valid_preds = model.predict(X_valid)

            rmse = float(np.sqrt(np.mean((y_valid.values - valid_preds) ** 2)))
            return rmse

        n_trials = getattr(self.train_cfg, "optuna_trials_boosting", 25)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = dict(study.best_params)
        best_params["search_method"] = "optuna"
        best_params["optuna_best_value"] = float(study.best_value)
        best_params["optuna_trials"] = n_trials

        print(f"[Boosting] Optuna best value={study.best_value}")
        print(f"[Boosting] Optuna best params={study.best_params}")

        return best_params

    def _fit_model(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
        params: dict,
    ) -> PriceBoostingModel:
        """
        Обучает финальную модель градиентного бустинга на всей train-выборке.

        После подбора параметров на внутренней validation-выборке финальная
        модель обучается на полном train-периоде, чтобы использовать максимум
        доступных исторических данных.
        """
        X_train = train_df[feature_cols]
        y_train = train_df[self.data_cfg.target_col]

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

        return model

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
        """
        Обучает модель градиентного бустинга и сохраняет результаты.

        При do_search=True выполняется подбор гиперпараметров на внутреннем
        validation-периоде. Итоговые метрики считаются только на test_df,
        то есть на отложенном периоде, который не используется при подборе
        параметров.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if preds_path is not None:
            preds_path.parent.mkdir(parents=True, exist_ok=True)

        numeric_cols, categorical_cols = self._split_feature_types(
            df=train_df,
            feature_cols=feature_cols,
        )

        print(f"[Boosting] train shape={train_df.shape}, test shape={test_df.shape}")
        print(
            f"[Boosting] numeric_cols={len(numeric_cols)}, "
            f"categorical_cols={len(categorical_cols)}"
        )

        if do_search:
            inner_train_df, inner_valid_df = self._train_valid_split(train_df)

            best_params = self._tune_params_with_optuna(
                inner_train_df=inner_train_df,
                inner_valid_df=inner_valid_df,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )
        else:
            best_params = self._get_default_params()
            best_params["search_method"] = "TrainConfig"

        final_model = self._fit_model(
            train_df=train_df,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            params=best_params,
        )

        X_test = test_df[feature_cols]
        y_test = test_df[self.data_cfg.target_col]

        preds = final_model.predict(X_test)

        metrics = evaluate_regression(y_test, preds)

        print(f"[Boosting] final metrics={metrics}")

        # Сохраняем модель в прежнем формате проекта, чтобы не ломать
        # существующую инфраструктуру загрузки моделей.
        final_model.save(model_path)

        save_metrics(metrics, metrics_path)

        if preds_path is not None:
            id_cols = [
                col for col in list(self.data_cfg.series_id_cols)
                if col in test_df.columns
            ]

            preds_df = test_df[id_cols + [self.data_cfg.date_col]].copy()
            preds_df["y_true"] = y_test.values

            # Для совместимости с уже существующими гибридными моделями
            # сохраняем оба названия прогноза: y_pred и prediction.
            preds_df["y_pred"] = preds
            preds_df["prediction"] = preds

            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
            print(f"[Boosting] predictions saved to {preds_path}")

        best_params_to_save = {
            **best_params,
            "model_type": "PriceBoostingModel",
            "feature_count": len(feature_cols),
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols),
            "categorical_cols": categorical_cols,
        }

        # Дополнительно сохраняем параметры рядом с метриками.
        # Это облегчает проверку эксперимента и делает запуск воспроизводимым.
        params_path = metrics_path.parent / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params_to_save, f, ensure_ascii=False, indent=4)

        return metrics, best_params_to_save