from datetime import datetime
import json

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter

from src.train_sarima import SarimaTrainer
from src.train_boosting import BoostingTrainer
from src.train_m5_style import M5StyleTrainer
from src.train_hybrid import HybridTrainer

from src.final_results_manager import FinalResultsManager


def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


class FinalPipeline:
    """
    Финальный прогон.

    Считаем:
    1. SARIMA — на всех рядах
    2. Boosting — на top-N рядах (по памяти)
    3. M5-style — на top-N рядах
    4. Hybrid SARIMA + Boosting — на top-N рядах

    Все результаты складываются отдельно в reports/final_run.
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

        self.final_reports_dir = self.paths.reports_dir / "final_run"
        self.final_models_dir = self.paths.models_dir / "final_run"

    def _prepare_dirs(self) -> None:
        for path in [
            self.final_reports_dir,
            self.final_models_dir,
            self.paths.processed_dir,
            self.paths.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _subset_top_series_for_tabular(self, train_df, test_df):
        """
        Оставляем top-N рядов для табличных моделей,
        чтобы boosting / one-hot не убили процесс по памяти.
        """
        max_series = self.train_cfg.final_tabular_max_series
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

        return train_sub.reset_index(drop=True), test_sub.reset_index(drop=True)

    def run(self) -> None:
        self._prepare_dirs()

        manager = FinalResultsManager(self.final_reports_dir)

        # 1. Загрузка данных
        log_step("Loading raw data...")
        loader = DataLoader(self.paths, self.data_cfg)
        bundle = loader.load()
        log_step(
            f"Loaded: sales={bundle.sales.shape}, calendar={bundle.calendar.shape}, prices={bundle.prices.shape}"
        )

        # 2. Построение аналитической панели
        log_step("Building analytical dataset...")
        builder = PriceDatasetBuilder(self.data_cfg, self.feat_cfg)
        artifacts = builder.build(bundle)

        processed_path = self.final_reports_dir / "price_panel_final.parquet"
        artifacts.panel.to_parquet(processed_path, index=False)

        log_step(f"Saved processed final dataset to {processed_path}")
        log_step(f"Panel shape: {artifacts.panel.shape}")
        log_step(f"Feature count: {len(artifacts.feature_columns)}")

        # Лучший набор признаков для ML-моделей
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
        log_step(
            f"Train date range: {train_df[self.data_cfg.date_col].min()} -> {train_df[self.data_cfg.date_col].max()}"
        )
        log_step(
            f"Test date range: {test_df[self.data_cfg.date_col].min()} -> {test_df[self.data_cfg.date_col].max()}"
        )

        # 4. Подвыборка для табличных моделей
        train_tabular_df, test_tabular_df = self._subset_top_series_for_tabular(train_df, test_df)
        log_step(f"Tabular train shape: {train_tabular_df.shape}")
        log_step(f"Tabular test shape: {test_tabular_df.shape}")
        log_step(f"Tabular max series: {self.train_cfg.final_tabular_max_series}")

        # 5. SARIMA — на всех рядах
        log_step("Final run: SARIMA")
        sarima_dir = self.final_reports_dir / "sarima"
        sarima_dir.mkdir(parents=True, exist_ok=True)

        sarima_metrics, sarima_params = SarimaTrainer(self.data_cfg, self.train_cfg).run(
            train_df=train_df,
            test_df=test_df,
            metrics_path=sarima_dir / "metrics.json",
            preds_path=sarima_dir / "predictions.csv",
            do_search=True,
            progress_path=sarima_dir / "progress.csv",
        )

        with open(sarima_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(sarima_params, f, ensure_ascii=False, indent=4)

        manager.add_result(
            model_name="sarima",
            feature_set_name="univariate_target_only",
            metrics=sarima_metrics,
            best_params=sarima_params,
            notes="final run on all series",
        )
        log_step(f"SARIMA done: {sarima_metrics}")

        # 6. Boosting — на tabular subset
        log_step("Final run: Boosting")
        boosting_dir = self.final_reports_dir / "boosting"
        boosting_dir.mkdir(parents=True, exist_ok=True)

        boosting_metrics, boosting_params = BoostingTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_tabular_df,
            test_df=test_tabular_df,
            feature_cols=all_feature_cols,
            model_path=self.final_models_dir / "boosting_model.pkl",
            metrics_path=boosting_dir / "metrics.json",
            do_search=True,
            preds_path=boosting_dir / "predictions.csv",
        )

        manager.add_result(
            model_name="boosting",
            feature_set_name="all_features",
            metrics=boosting_metrics,
            best_params=boosting_params,
            notes=f"final run on top-{self.train_cfg.final_tabular_max_series} series",
        )
        log_step(f"Boosting done: {boosting_metrics}")

        # 7. M5-style — на tabular subset
        log_step("Final run: M5-style")
        m5_dir = self.final_reports_dir / "m5_style"
        m5_dir.mkdir(parents=True, exist_ok=True)

        m5_metrics, m5_params = M5StyleTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_tabular_df,
            test_df=test_tabular_df,
            feature_cols=all_feature_cols,
            model_path=self.final_models_dir / "m5_style_model.pkl",
            metrics_path=m5_dir / "metrics.json",
            do_search=True,
        )

        manager.add_result(
            model_name="m5_style",
            feature_set_name="all_features",
            metrics=m5_metrics,
            best_params=m5_params,
            notes=f"final run on top-{self.train_cfg.final_tabular_max_series} series",
        )
        log_step(f"M5-style done: {m5_metrics}")

        # 8. Hybrid SARIMA + Boosting — на tabular subset
        log_step("Final run: Hybrid SARIMA + Boosting")
        hybrid_dir = self.final_reports_dir / "hybrid_sarima_boosting"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        hybrid_metrics = HybridTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_tabular_df,
            test_df=test_tabular_df,
            tabular_feature_cols=all_feature_cols,
            metrics_path=hybrid_dir / "metrics.json",
            sarima_best_params=sarima_params,
            preds_path=hybrid_dir / "predictions.csv",
        )

        manager.add_result(
            model_name="hybrid_sarima_boosting",
            feature_set_name="all_features",
            metrics=hybrid_metrics,
            best_params=None,
            notes=f"final run on top-{self.train_cfg.final_tabular_max_series} series",
        )
        log_step(f"Hybrid done: {hybrid_metrics}")

        # 9. Финал
        log_step("Final run finished.")
        print(f"Final comparison saved to: {manager.final_path}")
        print(f"Sorted comparison saved to: {manager.sorted_path}")