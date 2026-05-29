from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter

from src.train_sarima import SarimaTrainer
from src.train_boosting import BoostingTrainer
from src.train_m5_style import M5StyleTrainer
from src.train_lstm import LSTMTrainer
from src.train_hybrid_boosting_m5 import HybridBoostingM5Trainer

from src.final_results_manager import FinalResultsManager


def log_step(message: str) -> None:
    """
    Вывод этапов выполнения пайплайна.

    Такой лог нужен для воспроизводимости эксперимента: по нему можно
    отследить, какие этапы были выполнены, а какие были загружены из
    ранее сохраненных результатов.
    """
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


class FinalPipeline:
    """
    Итоговый пайплайн экспериментальной части ВКР.

    Пайплайн запускает все модели на единой выборке:
    - SARIMA;
    - градиентный бустинг;
    - M5-style модель;
    - LSTM с embedding-признаками;
    - гибридную модель Boosting + M5.

    Единая выборка нужна для корректного сравнения моделей: различия в
    итоговых метриках должны объясняться моделирующим подходом, а не разным
    составом обучающих и тестовых данных.

    В пайплайне предусмотрено повторное использование ранее сохраненных
    результатов. Это важно, поскольку SARIMA, LSTM и M5-style модель могут
    обучаться долго. При повторном запуске пайплайн может не пересчитывать
    уже готовые модели, если сохранены все ключевые артефакты:
    - metrics.json;
    - predictions.csv;
    - файл обученной модели;
    - best_params.json, если параметры сохраняются.

    Для принудительного пересчета можно использовать флаги:
    - force_rebuild_dataset=True — заново построить аналитическую панель;
    - force_retrain_models=True — заново обучить все модели.
    """

    def __init__(
        self,
        paths: PathsConfig,
        data_cfg: DataConfig,
        feat_cfg: FeatureConfig,
        train_cfg: TrainConfig,
        force_rebuild_dataset: bool = False,
        force_retrain_models: bool = False,
    ) -> None:
        self.paths = paths
        self.data_cfg = data_cfg
        self.feat_cfg = feat_cfg
        self.train_cfg = train_cfg

        # Для текущей доработанной версии актуальные результаты пишутся в final_run.
        # baseline_run не удаляется и остается как резерв предыдущего успешного запуска.
        self.final_reports_dir = self.paths.reports_dir / "final_run"
        self.final_models_dir = self.paths.models_dir / "final_run"

        self.force_rebuild_dataset = force_rebuild_dataset
        self.force_retrain_models = force_retrain_models

    def _prepare_dirs(self) -> None:
        """
        Создает директории, необходимые для сохранения датасета, моделей,
        метрик и итоговых сравнений.
        """
        for path in [
            self.final_reports_dir,
            self.final_models_dir,
            self.paths.processed_dir,
            self.paths.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_exists(path: Path) -> bool:
        """
        Проверяет, что файл существует и не является пустым.
        """
        return path.exists() and path.is_file() and path.stat().st_size > 0

    def _model_artifacts_exist(
        self,
        metrics_path: Path,
        preds_path: Path,
        model_path: Path | None = None,
        params_path: Path | None = None,
    ) -> bool:
        """
        Проверяет наличие сохраненных результатов модели.

        Проверять только metrics.json недостаточно: метрики могут остаться
        от старого запуска, а прогнозы или модель при этом отсутствовать.
        Поэтому для повторного использования результата проверяются все
        необходимые артефакты.
        """
        required = [
            self._file_exists(metrics_path),
            self._file_exists(preds_path),
        ]

        if model_path is not None:
            required.append(self._file_exists(model_path))

        if params_path is not None:
            required.append(self._file_exists(params_path))

        return all(required)

    @staticmethod
    def _load_json(path: Path) -> dict:
        """
        Загружает JSON-файл с метриками или параметрами модели.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(data: dict, path: Path) -> None:
        """
        Сохраняет словарь в JSON-файл в читаемом виде.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _subset_top_series_for_models(self, train_df, test_df):
        """
        Оставляет top-N наиболее полных временных рядов.

        Ограничение числа рядов используется для того, чтобы тяжелые модели
        временных рядов и нейросетевые модели могли быть обучены в разумное
        время. При этом одна и та же подвыборка используется для всех моделей,
        что сохраняет корректность итогового сравнения.
        """
        max_series = 5000
        group_cols = list(self.data_cfg.series_id_cols)

        counts = (
            train_df.groupby(group_cols)
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
            .head(max_series)
        )

        train_sub = train_df.merge(counts[group_cols], on=group_cols, how="inner")
        test_sub = test_df.merge(counts[group_cols], on=group_cols, how="inner")

        return train_sub.reset_index(drop=True), test_sub.reset_index(drop=True), max_series

    def _build_or_load_dataset(self):
        """
        Строит или загружает аналитическую панель.

        Если файл price_panel_final.parquet уже существует и не включен
        force_rebuild_dataset, пайплайн использует сохраненный датасет.
        Это позволяет повторно запускать обучение моделей без пересборки
        исходной панели.

        Важно: даже при загрузке сохраненного parquet вызывается builder.build(),
        чтобы получить список feature_columns из существующей логики проекта.
        Если позднее в PriceDatasetBuilder будет добавлен отдельный метод для
        восстановления списка признаков из parquet, этот блок можно упростить.
        """
        processed_path = self.final_reports_dir / "price_panel_final.parquet"

        log_step("Loading raw data...")
        loader = DataLoader(self.paths, self.data_cfg)
        bundle = loader.load()

        log_step(
            f"Loaded: sales={bundle.sales.shape}, "
            f"calendar={bundle.calendar.shape}, "
            f"prices={bundle.prices.shape}"
        )

        builder = PriceDatasetBuilder(self.data_cfg, self.feat_cfg)

        if self._file_exists(processed_path) and not self.force_rebuild_dataset:
            log_step(f"Loading existing analytical dataset: {processed_path}")

            # Загружаем уже построенную аналитическую панель.
            panel = pd.read_parquet(processed_path)

            # В текущей архитектуре проекта список признаков хранится в artifacts.
            # Поэтому builder.build() вызывается для восстановления feature_columns.
            # Сам panel при этом берется из parquet, чтобы не зависеть от новой сборки.
            artifacts = builder.build(bundle)
            all_feature_cols = artifacts.feature_columns

            log_step(f"Loaded saved panel shape: {panel.shape}")
            log_step(f"Feature columns count: {len(all_feature_cols)}")

            return panel, all_feature_cols

        log_step("Building analytical dataset...")

        artifacts = builder.build(bundle)
        panel = artifacts.panel
        all_feature_cols = artifacts.feature_columns

        panel.to_parquet(processed_path, index=False)

        log_step(f"Saved dataset to {processed_path}")
        log_step(f"Panel shape: {panel.shape}")
        log_step(f"Feature columns count: {len(all_feature_cols)}")

        return panel, all_feature_cols

    def run(self) -> None:
        """
        Запускает полный итоговый эксперимент.

        По умолчанию пайплайн использует сохраненные результаты, если они
        существуют. Чтобы пересчитать модели после изменения их кода, нужно
        создать FinalPipeline(..., force_retrain_models=True).
        """
        self._prepare_dirs()

        manager = FinalResultsManager(self.final_reports_dir)

        # ==========================================================
        # 1. Датасет
        # ==========================================================

        panel, all_feature_cols = self._build_or_load_dataset()

        # ==========================================================
        # 2. Train / test split
        # ==========================================================

        log_step("Splitting into train/test...")

        splitter = TimeSeriesSplitter(
            date_col=self.data_cfg.date_col,
            test_size_weeks=self.train_cfg.test_size_weeks,
        )

        train_df, test_df = splitter.split(panel)

        log_step(f"Train shape: {train_df.shape}")
        log_step(f"Test shape: {test_df.shape}")

        # ==========================================================
        # 3. Единая подвыборка top-5000 для всех моделей
        # ==========================================================

        train_model_df, test_model_df, used_series_count = self._subset_top_series_for_models(
            train_df,
            test_df,
        )

        log_step(f"Model train shape: {train_model_df.shape}")
        log_step(f"Model test shape: {test_model_df.shape}")
        log_step(f"Top series used for final comparison: {used_series_count}")

        # ==========================================================
        # 4. SARIMA
        # ==========================================================

        log_step("Final run: SARIMA")

        sarima_dir = self.final_reports_dir / "sarima"
        sarima_dir.mkdir(parents=True, exist_ok=True)

        sarima_metrics_path = sarima_dir / "metrics.json"
        sarima_preds_path = sarima_dir / "predictions.csv"
        sarima_params_path = sarima_dir / "best_params.json"

        if (
            not self.force_retrain_models
            and self._file_exists(sarima_metrics_path)
            and self._file_exists(sarima_preds_path)
            and self._file_exists(sarima_params_path)
        ):
            log_step("SARIMA artifacts found. Loading saved result...")

            sarima_metrics = self._load_json(sarima_metrics_path)
            sarima_params = self._load_json(sarima_params_path)

        else:
            # SARIMA остается классической статистической моделью.
            # Optuna здесь не используется: параметры подбираются внутри
            # SarimaTrainer через do_search=True.
            sarima_metrics, sarima_params = SarimaTrainer(
                self.data_cfg,
                self.train_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                metrics_path=sarima_metrics_path,
                preds_path=sarima_preds_path,
                do_search=True,
                progress_path=sarima_dir / "progress.csv",
            )

            self._save_json(sarima_params, sarima_params_path)

        manager.add_result(
            model_name="sarima",
            feature_set_name="top_5000_final",
            metrics=sarima_metrics,
            best_params=sarima_params,
            notes="final run; SARIMA without Optuna",
        )

        log_step(f"SARIMA done: {sarima_metrics}")

        # ==========================================================
        # 5. Градиентный бустинг
        # ==========================================================

        log_step("Final run: Boosting")

        boosting_dir = self.final_reports_dir / "boosting"
        boosting_dir.mkdir(parents=True, exist_ok=True)

        boosting_metrics_path = boosting_dir / "metrics.json"
        boosting_preds_path = boosting_dir / "predictions.csv"
        boosting_params_path = boosting_dir / "best_params.json"
        boosting_model_path = self.final_models_dir / "boosting_model.pkl"

        if (
            not self.force_retrain_models
            and self._model_artifacts_exist(
                metrics_path=boosting_metrics_path,
                preds_path=boosting_preds_path,
                model_path=boosting_model_path,
                params_path=boosting_params_path,
            )
        ):
            log_step("Boosting artifacts found. Loading saved result...")

            boosting_metrics = self._load_json(boosting_metrics_path)
            boosting_params = self._load_json(boosting_params_path)

        else:
            # Внутри BoostingTrainer будет выполняться подбор гиперпараметров.
            # В итоговой версии для этой модели допустимо использовать Optuna,
            # так как у градиентного бустинга широкое пространство параметров.
            boosting_metrics, boosting_params = BoostingTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                feature_cols=all_feature_cols,
                model_path=boosting_model_path,
                metrics_path=boosting_metrics_path,
                do_search=True,
                preds_path=boosting_preds_path,
            )

            self._save_json(boosting_params, boosting_params_path)

        manager.add_result(
            model_name="boosting",
            feature_set_name="top_5000_final",
            metrics=boosting_metrics,
            best_params=boosting_params,
            notes="final run; tuned gradient boosting",
        )

        log_step(f"Boosting done: {boosting_metrics}")

        # ==========================================================
        # 6. M5-style модель
        # ==========================================================

        log_step("Final run: M5-style")

        m5_dir = self.final_reports_dir / "m5_style"
        m5_dir.mkdir(parents=True, exist_ok=True)

        m5_metrics_path = m5_dir / "metrics.json"
        m5_preds_path = m5_dir / "predictions.csv"
        m5_params_path = m5_dir / "best_params.json"
        m5_model_path = self.final_models_dir / "m5_style_model.pkl"

        if (
            not self.force_retrain_models
            and self._model_artifacts_exist(
                metrics_path=m5_metrics_path,
                preds_path=m5_preds_path,
                model_path=m5_model_path,
                params_path=m5_params_path,
            )
        ):
            log_step("M5-style artifacts found. Loading saved result...")

            m5_metrics = self._load_json(m5_metrics_path)
            m5_params = self._load_json(m5_params_path)

        else:
            # M5-style модель является ключевой доработкой эксперимента.
            # Здесь сохраняется название модели, но усиливается содержание:
            # LightGBM, логарифмирование целевой переменной, лаговые признаки,
            # rolling-признаки, календарные признаки и признаки иерархии M5.
            m5_metrics, m5_params = M5StyleTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                feature_cols=all_feature_cols,
                model_path=m5_model_path,
                metrics_path=m5_metrics_path,
                do_search=True,
                preds_path=m5_preds_path,
            )

            self._save_json(m5_params, m5_params_path)

        manager.add_result(
            model_name="m5_style",
            feature_set_name="top_5000_final",
            metrics=m5_metrics,
            best_params=m5_params,
            notes="final run; M5-style model with hierarchy-aware feature space",
        )

        log_step(f"M5-style done: {m5_metrics}")

        # ==========================================================
        # 7. LSTM with embeddings
        # ==========================================================

        log_step("Final run: LSTM with embeddings")

        lstm_dir = self.final_reports_dir / "lstm_embeddings"
        lstm_dir.mkdir(parents=True, exist_ok=True)

        lstm_metrics_path = lstm_dir / "metrics.json"
        lstm_preds_path = lstm_dir / "predictions.csv"
        lstm_params_path = lstm_dir / "best_params.json"
        lstm_model_path = self.final_models_dir / "lstm_embeddings_model.pt"

        lstm_params = {
            "model_type": "LSTMEmbeddingRegressor",
            "seq_len": self.train_cfg.seq_len,
            "hidden_size": self.train_cfg.lstm_hidden_size,
            "num_layers": self.train_cfg.lstm_num_layers,
            "dropout": self.train_cfg.lstm_dropout,
            "epochs": self.train_cfg.lstm_epochs,
            "batch_size": self.train_cfg.lstm_batch_size,
            "learning_rate": self.train_cfg.lstm_lr,
            "optimizer": "Adam",
            "loss_function": "MSELoss",
            "embedding_dim_item_id": self.train_cfg.embedding_dim_item_id,
            "embedding_dim_store_id": self.train_cfg.embedding_dim_store_id,
            "embedding_dim_dept_id": self.train_cfg.embedding_dim_dept_id,
            "embedding_dim_cat_id": self.train_cfg.embedding_dim_cat_id,
            "embedding_dim_state_id": self.train_cfg.embedding_dim_state_id,
            "event_name_1_embedding_dim": 8,
            "event_type_1_embedding_dim": 4,
            "target_transform": "none",
            "source": "TrainConfig; no Optuna tuning",
        }

        if (
            not self.force_retrain_models
            and self._file_exists(lstm_metrics_path)
            and self._file_exists(lstm_preds_path)
        ):
            log_step("LSTM embeddings metrics and predictions found. Loading saved result...")

            lstm_metrics = self._load_json(lstm_metrics_path)

            if self._file_exists(lstm_params_path):
                lstm_params = self._load_json(lstm_params_path)
            else:
                self._save_json(lstm_params, lstm_params_path)
                log_step(f"LSTM params file created: {lstm_params_path}")

        else:
            lstm_feature_cols = all_feature_cols

            lstm_metrics, _ = LSTMTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                seq_feature_cols=lstm_feature_cols,
                model_path=lstm_model_path,
                metrics_path=lstm_metrics_path,
                preds_path=lstm_preds_path,
            )

            self._save_json(lstm_params, lstm_params_path)

        manager.add_result(
            model_name="lstm_embeddings",
            feature_set_name="top_5000_final",
            metrics=lstm_metrics,
            best_params=lstm_params,
            notes="final run; LSTM with trainable embeddings",
        )

        log_step(f"LSTM embeddings done: {lstm_metrics}")

        # ==========================================================
        # 8. Гибридная модель Boosting + M5-style
        # ==========================================================

        log_step("Final run: Hybrid Boosting + M5-style")

        hybrid_dir = self.final_reports_dir / "hybrid_boosting_m5"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        hybrid_metrics_path = hybrid_dir / "metrics.json"
        hybrid_preds_path = hybrid_dir / "predictions.csv"
        hybrid_params_path = hybrid_dir / "best_params.json"

        if (
            not self.force_retrain_models
            and self._file_exists(hybrid_metrics_path)
            and self._file_exists(hybrid_preds_path)
            and self._file_exists(hybrid_params_path)
        ):
            log_step("Hybrid Boosting + M5 artifacts found. Loading saved result...")

            hybrid_metrics = self._load_json(hybrid_metrics_path)
            hybrid_params = self._load_json(hybrid_params_path)

        else:
            hybrid_metrics, hybrid_params = HybridBoostingM5Trainer().run(
                boosting_preds_path=boosting_preds_path,
                m5_preds_path=m5_preds_path,
                metrics_path=hybrid_metrics_path,
                preds_path=hybrid_preds_path,
                params_path=hybrid_params_path,
            )

        manager.add_result(
            model_name="hybrid_boosting_m5",
            feature_set_name="top_5000_final",
            metrics=hybrid_metrics,
            best_params=hybrid_params,
            notes="final run; weighted ensemble of Boosting and M5-style",
        )

        log_step(f"Hybrid Boosting + M5 done: {hybrid_metrics}")

        # ==========================================================
        # 9. Финальное сравнение одиночных моделей
        # ==========================================================

        log_step("Final comparison of individual models finished.")

        print(f"Saved comparison to: {manager.final_path}")
        print(f"Saved sorted comparison to: {manager.sorted_path}")