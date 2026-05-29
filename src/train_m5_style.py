from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics


try:
    import lightgbm as lgb
except ImportError as exc:
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc


try:
    import optuna
except ImportError:
    optuna = None


class M5StyleTrainer:
    """
    Тренер M5-style модели.

    В данной работе название модели остается прежним — M5-style,
    однако сама реализация дорабатывается в соответствии с ключевыми
    принципами решений соревнования M5 Forecasting Accuracy:

    - используется глобальная модель градиентного бустинга LightGBM;
    - целевая переменная прогнозируется в логарифмической шкале;
    - используются календарные, ценовые, лаговые и скользящие признаки;
    - учитывается иерархическая структура M5-данных:
      item_id, dept_id, cat_id, store_id, state_id;
    - гиперпараметры LightGBM подбираются на валидационной выборке;
    - итоговая оценка выполняется на отложенном тестовом периоде.

    Важно: M5-style модель не является полным воспроизведением победившего
    решения M5, так как в работе решается задача прогнозирования цен,
    а не спроса. Поэтому подход адаптирован к целевой переменной price.
    """

    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

    @staticmethod
    def _safe_log1p(values: pd.Series | np.ndarray) -> np.ndarray:
        """
        Логарифмирование целевой переменной используется для снижения влияния
        выбросов и стабилизации обучения бустинговой модели.
        """
        return np.log1p(np.clip(values, a_min=0, a_max=None))

    @staticmethod
    def _safe_expm1(values: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование прогноза из логарифмической шкалы.
        Отрицательные прогнозы после обратного преобразования обрезаются до нуля,
        так как цена не может быть отрицательной.
        """
        preds = np.expm1(values)
        return np.clip(preds, a_min=0, a_max=None)

    def _get_available_hierarchy_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Возвращает доступные уровни иерархии M5.

        В исходных данных M5 товарная структура задается уровнями:
        state_id -> store_id -> cat_id -> dept_id -> item_id.

        Проверка через наличие колонок делает код устойчивым: если часть
        признаков отсутствует в конкретной версии датасета, модель продолжит
        работать на доступных уровнях.
        """
        candidate_cols = [
            "state_id",
            "store_id",
            "cat_id",
            "dept_id",
            "item_id",
        ]

        hierarchy_cols = [col for col in candidate_cols if col in df.columns]

        # Также учитываем идентификаторы ряда из DataConfig, если они есть
        # в датасете и еще не добавлены.
        for col in self.data_cfg.series_id_cols:
            if col in df.columns and col not in hierarchy_cols:
                hierarchy_cols.append(col)

        return hierarchy_cols

    def _add_hierarchy_lag_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Добавляет иерархические лаговые признаки.

        В оригинальном M5 прогнозировались продажи, а верхние уровни иерархии
        формировались суммированием. В данной работе прогнозируется цена,
        поэтому агрегирование по уровням иерархии выполняется через среднее
        значение целевой переменной.

        Для предотвращения прямой утечки целевой переменной используются только
        сдвинутые значения: lag и rolling строятся после shift(1).
        Это означает, что модель получает информацию о предыдущем поведении
        уровня иерархии, но не получает текущее значение target в момент прогноза.
        """

        date_col = self.data_cfg.date_col
        target_col = self.data_cfg.target_col

        hierarchy_cols = self._get_available_hierarchy_cols(train_df)

        if not hierarchy_cols:
            print("[M5-style] hierarchy columns not found; using existing feature set only")
            return train_df.copy(), test_df.copy(), []

        train_part = train_df.copy()
        test_part = test_df.copy()

        train_part["_is_train_part"] = 1
        test_part["_is_train_part"] = 0

        full_df = pd.concat([train_part, test_part], axis=0, ignore_index=True)
        full_df = full_df.sort_values([date_col] + hierarchy_cols).reset_index(drop=True)

        created_features: list[str] = []

        # Для M5-подхода достаточно нескольких устойчивых лагов.
        # Лаг 1 отражает ближайшую динамику, лаги 7 и 28 — недельный
        # и месячный характер изменений.
        lag_values = [1, 7, 28]
        rolling_windows = [7, 28]

        for level_col in hierarchy_cols:
            if level_col not in full_df.columns:
                continue

            level_daily = (
                full_df.groupby([level_col, date_col], as_index=False)[target_col]
                .mean()
                .rename(columns={target_col: f"{level_col}_mean_target"})
            )

            level_daily = level_daily.sort_values([level_col, date_col]).reset_index(drop=True)

            for lag in lag_values:
                feature_name = f"{level_col}_mean_target_lag_{lag}"
                level_daily[feature_name] = (
                    level_daily.groupby(level_col)[f"{level_col}_mean_target"]
                    .shift(lag)
                )
                created_features.append(feature_name)

            for window in rolling_windows:
                shifted = (
                    level_daily.groupby(level_col)[f"{level_col}_mean_target"]
                    .shift(1)
                )

                mean_feature = f"{level_col}_mean_target_roll_mean_{window}"
                std_feature = f"{level_col}_mean_target_roll_std_{window}"

                level_daily[mean_feature] = (
                    shifted.groupby(level_daily[level_col])
                    .rolling(window)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                level_daily[std_feature] = (
                    shifted.groupby(level_daily[level_col])
                    .rolling(window)
                    .std()
                    .reset_index(level=0, drop=True)
                )

                created_features.extend([mean_feature, std_feature])

            level_features = level_daily.drop(columns=[f"{level_col}_mean_target"])

            full_df = full_df.merge(
                level_features,
                on=[level_col, date_col],
                how="left",
            )

        train_new = full_df[full_df["_is_train_part"] == 1].drop(columns=["_is_train_part"])
        test_new = full_df[full_df["_is_train_part"] == 0].drop(columns=["_is_train_part"])

        train_new = train_new.reset_index(drop=True)
        test_new = test_new.reset_index(drop=True)

        created_features = sorted(list(set(created_features)))

        print(f"[M5-style] hierarchy_cols={hierarchy_cols}")
        print(f"[M5-style] added hierarchy features={len(created_features)}")

        return train_new, test_new, created_features

    def _prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
        """
        Формирует итоговый набор признаков для M5-style модели.

        В качестве базы используются признаки, уже построенные в проекте.
        Дополнительно добавляются признаки иерархии и иерархические лаги.
        """

        train_df, test_df, hierarchy_lag_cols = self._add_hierarchy_lag_features(
            train_df=train_df,
            test_df=test_df,
        )

        hierarchy_cols = self._get_available_hierarchy_cols(train_df)

        final_feature_cols = list(feature_cols)

        for col in hierarchy_cols + hierarchy_lag_cols:
            if col in train_df.columns and col not in final_feature_cols:
                final_feature_cols.append(col)

        # Удаляем технические или целевые колонки, если они случайно попали
        # в список признаков.
        forbidden_cols = {
            self.data_cfg.target_col,
            self.data_cfg.date_col,
        }

        final_feature_cols = [
            col for col in final_feature_cols
            if col in train_df.columns and col not in forbidden_cols
        ]

        numeric_cols, categorical_cols = self._split_feature_types(
            df=train_df,
            feature_cols=final_feature_cols,
        )

        return train_df, test_df, final_feature_cols, numeric_cols, categorical_cols

    def _split_feature_types(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Разделяет признаки на числовые и категориальные.

        Для LightGBM категориальные признаки передаются явно. Это позволяет
        не заменять товарные и магазинные идентификаторы произвольными числами,
        а использовать их как категориальные уровни иерархии.
        """
        X = df[feature_cols].copy()

        numeric_cols = [c for c in feature_cols if is_numeric_dtype(X[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        print(
            f"[M5-style] numeric_cols={len(numeric_cols)}, "
            f"categorical_cols={len(categorical_cols)}"
        )

        return numeric_cols, categorical_cols

    def _prepare_lgbm_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Готовит матрицу признаков для LightGBM.

        Категориальные колонки приводятся к dtype='category', чтобы LightGBM
        обрабатывал их как категориальные признаки, а не как обычные строки.
        """
        X = df[feature_cols].copy()

        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")

        return X

    def _train_valid_split(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Делит обучающую выборку на train/validation по времени.

        Для временных рядов нельзя использовать случайное перемешивание,
        потому что это приводит к утечке будущей информации. Поэтому
        валидационная часть берется из последних дат обучающего периода.
        """
        date_col = self.data_cfg.date_col

        sorted_df = train_df.sort_values(date_col).reset_index(drop=True)

        unique_dates = sorted_df[date_col].drop_duplicates().sort_values().to_list()

        if len(unique_dates) < 5:
            raise ValueError(
                "Недостаточно уникальных дат для time-based validation "
                "в M5-style модели."
            )

        valid_size = max(1, int(len(unique_dates) * 0.15))
        valid_dates = set(unique_dates[-valid_size:])

        inner_train = sorted_df[~sorted_df[date_col].isin(valid_dates)].copy()
        inner_valid = sorted_df[sorted_df[date_col].isin(valid_dates)].copy()

        return inner_train.reset_index(drop=True), inner_valid.reset_index(drop=True)

    def _get_default_params(self) -> dict:
        """
        Базовые параметры LightGBM для M5-style модели.

        Эти параметры используются, если Optuna недоступна или если do_search=False.
        """
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 128,
            "max_depth": 10,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": self.train_cfg.random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def _tune_params_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train_log: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        categorical_cols: list[str],
    ) -> dict:
        """
        Подбирает гиперпараметры LightGBM с помощью Optuna.

        Optuna используется здесь потому, что LightGBM имеет широкое
        пространство гиперпараметров, а ручной подбор может привести к
        неоптимальному результату. Целевая функция минимизирует RMSE на
        временной validation-выборке.
        """
        if optuna is None:
            warnings.warn(
                "Optuna is not installed. M5-style model will use default parameters.",
                RuntimeWarning,
            )
            return self._get_default_params()

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 32, 256),
                "max_depth": trial.suggest_int("max_depth", 5, 14),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": self.train_cfg.random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }

            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_valid, self._safe_log1p(y_valid))],
                eval_metric="rmse",
                categorical_feature=categorical_cols,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ],
            )

            valid_pred_log = model.predict(X_valid)
            valid_pred = self._safe_expm1(valid_pred_log)

            rmse = float(np.sqrt(np.mean((y_valid - valid_pred) ** 2)))
            return rmse

        # Число trials ограничено, чтобы подбор был воспроизводимым по времени
        # и не превращал дипломный эксперимент в слишком тяжелый расчет.
        n_trials = getattr(self.train_cfg, "optuna_trials_m5", 30)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = self._get_default_params()
        best_params.update(study.best_params)

        best_params["optuna_best_value"] = float(study.best_value)
        best_params["optuna_trials"] = n_trials

        print(f"[M5-style] Optuna best value={study.best_value}")
        print(f"[M5-style] Optuna best params={study.best_params}")

        return best_params

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
        Обучает M5-style модель и сохраняет метрики, прогнозы и модель.

        Параметр do_search=True включает подбор гиперпараметров. В финальном
        эксперименте этот режим используется для того, чтобы M5-style модель
        была не просто бустингом с логарифмированием target, а доработанной
        моделью с настройкой параметров и признаками иерархии.
        """
        if lgb is None:
            raise ImportError(
                "Для M5-style модели требуется lightgbm. "
                "Установи зависимость: pip install lightgbm"
            ) from LIGHTGBM_IMPORT_ERROR

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if preds_path is not None:
            preds_path.parent.mkdir(parents=True, exist_ok=True)

        train_df, test_df, final_feature_cols, numeric_cols, categorical_cols = (
            self._prepare_features(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
            )
        )

        # После добавления лагов часть строк может содержать NaN.
        # Для бустинговой модели LightGBM это допустимо, однако строки без
        # target использовать нельзя.
        train_df = train_df.dropna(subset=[self.data_cfg.target_col]).reset_index(drop=True)
        test_df = test_df.dropna(subset=[self.data_cfg.target_col]).reset_index(drop=True)

        inner_train_df, inner_valid_df = self._train_valid_split(train_df)

        X_inner_train = self._prepare_lgbm_matrix(
            inner_train_df,
            final_feature_cols,
            categorical_cols,
        )
        y_inner_train_log = self._safe_log1p(inner_train_df[self.data_cfg.target_col].values)

        X_inner_valid = self._prepare_lgbm_matrix(
            inner_valid_df,
            final_feature_cols,
            categorical_cols,
        )
        y_inner_valid = inner_valid_df[self.data_cfg.target_col].values

        if do_search:
            best_params = self._tune_params_with_optuna(
                X_train=X_inner_train,
                y_train_log=y_inner_train_log,
                X_valid=X_inner_valid,
                y_valid=y_inner_valid,
                categorical_cols=categorical_cols,
            )
        else:
            best_params = self._get_default_params()

        # В служебных параметрах могут быть значения, которые не относятся
        # к LightGBM. Перед обучением финальной модели удаляем их.
        model_params = {
            key: value
            for key, value in best_params.items()
            if not key.startswith("optuna_")
        }

        X_train_full = self._prepare_lgbm_matrix(
            train_df,
            final_feature_cols,
            categorical_cols,
        )
        y_train_full_log = self._safe_log1p(train_df[self.data_cfg.target_col].values)

        X_test = self._prepare_lgbm_matrix(
            test_df,
            final_feature_cols,
            categorical_cols,
        )
        y_test = test_df[self.data_cfg.target_col].values

        print(f"[M5-style] final features={len(final_feature_cols)}")
        print(f"[M5-style] training final LightGBM model")

        final_model = lgb.LGBMRegressor(**model_params)

        final_model.fit(
            X_train_full,
            y_train_full_log,
            categorical_feature=categorical_cols,
        )

        pred_log = final_model.predict(X_test)
        preds = self._safe_expm1(pred_log)

        metrics = evaluate_regression(y_test, preds)

        print(f"[M5-style] final metrics={metrics}")

        model_artifact = {
            "model": final_model,
            "feature_cols": final_feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "target_transform": "log1p",
            "inverse_transform": "expm1",
            "model_type": "M5-style LightGBM",
        }

        joblib.dump(model_artifact, model_path)

        save_metrics(metrics, metrics_path)

        if preds_path is not None:
            id_cols = [
                col for col in list(self.data_cfg.series_id_cols)
                if col in test_df.columns
            ]

            base_cols = id_cols + [self.data_cfg.date_col]

            preds_df = test_df[base_cols].copy()
            preds_df["y_true"] = y_test
            preds_df["prediction"] = preds

            # Дополнительно сохраняем ключевые уровни иерархии, если они есть.
            # Это нужно для последующего анализа ошибки по товарным группам.
            for col in ["state_id", "store_id", "cat_id", "dept_id", "item_id"]:
                if col in test_df.columns and col not in preds_df.columns:
                    preds_df[col] = test_df[col].values

            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
            print(f"[M5-style] predictions saved to {preds_path}")

        best_params_to_save = {
            **best_params,
            "model_type": "LightGBM",
            "target_transform": "log1p",
            "feature_count": len(final_feature_cols),
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols),
            "categorical_cols": categorical_cols,
            "hierarchy_cols_used": self._get_available_hierarchy_cols(train_df),
        }

        # Дополнительно сохраняем параметры рядом с моделью. Это удобно для
        # проверки эксперимента без открытия итоговой таблицы сравнения.
        params_path = metrics_path.parent / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params_to_save, f, ensure_ascii=False, indent=4)

        return metrics, best_params_to_save