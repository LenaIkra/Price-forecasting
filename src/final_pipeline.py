from datetime import datetime
import json
from pathlib import Path

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
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


class FinalPipeline:
    """
    Финальный baseline-прогон моделей.

    Все модели считаются:
    - на одинаковом train/test split;
    - на одинаковой подвыборке top-N временных рядов.

    Это необходимо для корректного сравнительного анализа моделей.
    """

    def __init__(
        self,
        paths: PathsConfig,
        data_cfg: DataConfig,
        feat_cfg: FeatureConfig,
        train_cfg: TrainConfig,
    ) -> None:
        self.paths = paths
        self.data_cfg = data_cfg
        self.feat_cfg = feat_cfg
        self.train_cfg = train_cfg

        self.final_reports_dir = self.paths.reports_dir / "baseline_run"
        self.final_models_dir = self.paths.models_dir / "baseline_run"

    def _prepare_dirs(self) -> None:
        for path in [
            self.final_reports_dir,
            self.final_models_dir,
            self.paths.processed_dir,
            self.paths.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _metrics_exist(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def _subset_top_series_for_models(self, train_df, test_df):
        """
        Оставляем top-N наиболее полных рядов
        для корректного и сопоставимого сравнения моделей.
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

    def run(self) -> None:
        self._prepare_dirs()

        manager = FinalResultsManager(self.final_reports_dir)

        # 1. Загрузка данных
        log_step("Loading raw data...")
        loader = DataLoader(self.paths, self.data_cfg)
        bundle = loader.load()

        log_step(
            f"Loaded: sales={bundle.sales.shape}, "
            f"calendar={bundle.calendar.shape}, "
            f"prices={bundle.prices.shape}"
        )

        # 2. Построение аналитической панели
        log_step("Building analytical dataset...")

        builder = PriceDatasetBuilder(self.data_cfg, self.feat_cfg)
        artifacts = builder.build(bundle)

        processed_path = self.final_reports_dir / "price_panel_final.parquet"
        artifacts.panel.to_parquet(processed_path, index=False)

        log_step(f"Saved dataset to {processed_path}")
        log_step(f"Panel shape: {artifacts.panel.shape}")

        all_feature_cols = artifacts.feature_columns

        # 3. Train / test split
        log_step("Splitting into train/test...")

        splitter = TimeSeriesSplitter(
            date_col=self.data_cfg.date_col,
            test_size_weeks=self.train_cfg.test_size_weeks,
        )

        train_df, test_df = splitter.split(artifacts.panel)

        log_step(f"Train shape: {train_df.shape}")
        log_step(f"Test shape: {test_df.shape}")

        # 4. ЕДИНАЯ подвыборка top-5000 для ВСЕХ моделей
        train_model_df, test_model_df, used_series_count = self._subset_top_series_for_models(
            train_df,
            test_df,
        )

        log_step(f"Model train shape: {train_model_df.shape}")
        log_step(f"Model test shape: {test_model_df.shape}")
        log_step(f"Top series used for baseline comparison: {used_series_count}")

        # ==========================================================
        # 5. SARIMA
        # ==========================================================

        log_step("Baseline run: SARIMA")

        sarima_dir = self.final_reports_dir / "sarima"
        sarima_dir.mkdir(parents=True, exist_ok=True)

        sarima_metrics_path = sarima_dir / "metrics.json"

        if self._metrics_exist(sarima_metrics_path):
            log_step("SARIMA already calculated. Skipping...")

            with open(sarima_metrics_path, "r", encoding="utf-8") as f:
                sarima_metrics = json.load(f)

            manager.add_result(
                model_name="sarima",
                feature_set_name="top_5000_baseline",
                metrics=sarima_metrics,
                best_params=None,
                notes="loaded from cache",
            )

        else:
            sarima_metrics, sarima_params = SarimaTrainer(
                self.data_cfg,
                self.train_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                metrics_path=sarima_metrics_path,
                preds_path=sarima_dir / "predictions.csv",
                do_search=True,
                progress_path=sarima_dir / "progress.csv",
            )

            with open(sarima_dir / "best_params.json", "w", encoding="utf-8") as f:
                json.dump(sarima_params, f, ensure_ascii=False, indent=4)

            manager.add_result(
                model_name="sarima",
                feature_set_name="top_5000_baseline",
                metrics=sarima_metrics,
                best_params=sarima_params,
                notes="top-5000 series baseline",
            )

            log_step(f"SARIMA done: {sarima_metrics}")
        
        # ==========================================================
        # 6. BOOSTING
        # ==========================================================

        log_step("Baseline run: Boosting")

        boosting_dir = self.final_reports_dir / "boosting"
        boosting_dir.mkdir(parents=True, exist_ok=True)

        boosting_metrics_path = boosting_dir / "metrics.json"

        if self._metrics_exist(boosting_metrics_path):
            log_step("Boosting already calculated. Skipping...")

            with open(boosting_metrics_path, "r", encoding="utf-8") as f:
                boosting_metrics = json.load(f)

            manager.add_result(
                model_name="boosting",
                feature_set_name="top_5000_baseline",
                metrics=boosting_metrics,
                best_params=None,
                notes="loaded from cache",
            )

        else:
            boosting_metrics, boosting_params = BoostingTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                feature_cols=all_feature_cols,
                model_path=self.final_models_dir / "boosting_model.pkl",
                metrics_path=boosting_metrics_path,
                do_search=True,
                preds_path=boosting_dir / "predictions.csv",
            )

            manager.add_result(
                model_name="boosting",
                feature_set_name="top_5000_baseline",
                metrics=boosting_metrics,
                best_params=boosting_params,
                notes="top-5000 series baseline",
            )

            log_step(f"Boosting done: {boosting_metrics}")

        # ==========================================================
        # 7. M5 STYLE
        # ==========================================================

        log_step("Baseline run: M5-style")

        m5_dir = self.final_reports_dir / "m5_style"
        m5_dir.mkdir(parents=True, exist_ok=True)

        m5_metrics_path = m5_dir / "metrics.json"

        if self._metrics_exist(m5_metrics_path):
            log_step("M5-style already calculated. Skipping...")

            with open(m5_metrics_path, "r", encoding="utf-8") as f:
                m5_metrics = json.load(f)

            manager.add_result(
                model_name="m5_style",
                feature_set_name="top_5000_baseline",
                metrics=m5_metrics,
                best_params=None,
                notes="loaded from cache",
            )

        else:
            m5_metrics, m5_params = M5StyleTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                feature_cols=all_feature_cols,
                model_path=self.final_models_dir / "m5_style_model.pkl",
                metrics_path=m5_metrics_path,
                do_search=True,
                preds_path=m5_dir / "predictions.csv",
            )

            manager.add_result(
                model_name="m5_style",
                feature_set_name="top_5000_baseline",
                metrics=m5_metrics,
                best_params=m5_params,
                notes="top-5000 series baseline",
            )

            log_step(f"M5-style done: {m5_metrics}")

        # ==========================================================
        # 8. LSTM WITH EMBEDDINGS
        # ==========================================================

        log_step("Baseline run: LSTM with embeddings")

        lstm_dir = self.final_reports_dir / "lstm_embeddings"
        lstm_dir.mkdir(parents=True, exist_ok=True)

        lstm_metrics_path = lstm_dir / "metrics.json"

        if self._metrics_exist(lstm_metrics_path):
            log_step("LSTM embeddings already calculated. Skipping...")

            with open(lstm_metrics_path, "r", encoding="utf-8") as f:
                lstm_metrics = json.load(f)

            manager.add_result(
                model_name="lstm_embeddings",
                feature_set_name="top_5000_baseline",
                metrics=lstm_metrics,
                best_params=None,
                notes="loaded from cache",
            )

        else:
            lstm_feature_cols = all_feature_cols

            lstm_metrics, _ = LSTMTrainer(
                self.train_cfg,
                self.data_cfg,
            ).run(
                train_df=train_model_df,
                test_df=test_model_df,
                seq_feature_cols=lstm_feature_cols,
                model_path=self.final_models_dir / "lstm_embeddings_model.pt",
                metrics_path=lstm_metrics_path,
                preds_path=lstm_dir / "predictions.csv",
            )

            manager.add_result(
                model_name="lstm_embeddings",
                feature_set_name="top_5000_baseline",
                metrics=lstm_metrics,
                best_params={
                    "seq_len": self.train_cfg.seq_len,
                    "hidden_size": self.train_cfg.lstm_hidden_size,
                    "num_layers": self.train_cfg.lstm_num_layers,
                    "dropout": self.train_cfg.lstm_dropout,
                    "lr": self.train_cfg.lstm_lr,
                    "epochs": self.train_cfg.lstm_epochs,
                },
                notes="top-5000 series baseline",
            )

            log_step(f"LSTM embeddings done: {lstm_metrics}")

        # ==========================================================
        # 9. HYBRID BOOSTING + M5
        # ==========================================================

        log_step("Hybrid run: Boosting + M5")

        hybrid_dir = self.final_reports_dir / "hybrid_boosting_m5"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        hybrid_metrics_path = hybrid_dir / "metrics.json"

        if self._metrics_exist(hybrid_metrics_path):
            log_step("Hybrid Boosting + M5 already calculated. Skipping...")

            with open(hybrid_metrics_path, "r", encoding="utf-8") as f:
                hybrid_metrics = json.load(f)

            manager.add_result(
                model_name="hybrid_boosting_m5",
                feature_set_name="top_5000_baseline",
                metrics=hybrid_metrics,
                best_params=None,
                notes="loaded from cache",
            )

        else:
            hybrid_metrics = HybridBoostingM5Trainer().run(
                boosting_preds_path=boosting_dir / "predictions.csv",
                m5_preds_path=m5_dir / "predictions.csv",
                metrics_path=hybrid_metrics_path,
            )

            manager.add_result(
                model_name="hybrid_boosting_m5",
                feature_set_name="top_5000_baseline",
                metrics=hybrid_metrics,
                best_params={
                    "boosting_weight": 0.7,
                    "m5_weight": 0.3,
                },
                notes="weighted ensemble",
            )

            log_step(f"Hybrid done: {hybrid_metrics}")


        # ==========================================================
        # 10. ФИНАЛ
        # ==========================================================

        log_step("Baseline comparison finished.")

        print(f"Saved comparison to: {manager.final_path}")
        print(f"Saved sorted comparison to: {manager.sorted_path}")