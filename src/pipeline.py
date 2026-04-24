from datetime import datetime

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter
from src.train_linear import LinearTrainer
from src.train_sarima import SarimaTrainer
from src.train_boosting import BoostingTrainer
from src.train_m5_style import M5StyleTrainer
from src.train_lstm import LSTMTrainer
from src.train_hybrid import HybridTrainer
from src.results_manager import ResultsManager


def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


class FullPipeline:
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

    def run(self) -> None:
        self._prepare_dirs()

        comparison_path = self.paths.metrics_dir / "model_comparison.csv"

        # Жестко удаляем старый файл перед запуском
        if comparison_path.exists():
            print("Удаляю старый model_comparison.csv")
            comparison_path.unlink()

        results_manager = ResultsManager(
            self.paths.metrics_dir / "model_comparison.csv"
        )

        log_step("Loading raw data...")
        loader = DataLoader(self.paths, self.data_cfg)
        bundle = loader.load()
        log_step(
            f"Loaded: sales={bundle.sales.shape}, calendar={bundle.calendar.shape}, prices={bundle.prices.shape}"
        )

        log_step("Building analytical dataset...")
        builder = PriceDatasetBuilder(self.data_cfg, self.feat_cfg)
        artifacts = builder.build(bundle)

        processed_path = self.paths.processed_dir / "price_panel.parquet"
        artifacts.panel.to_parquet(processed_path, index=False)
        log_step(f"Saved processed dataset to {processed_path}")
        log_step(f"Panel shape: {artifacts.panel.shape}")
        log_step(f"Number of features: {len(artifacts.feature_columns)}")

        log_step("Splitting into train/test...")
        splitter = TimeSeriesSplitter(
            date_col=self.data_cfg.date_col,
            test_size_weeks=self.train_cfg.test_size_weeks,
        )
        train_df, test_df = splitter.split(artifacts.panel)

        log_step(f"Train shape: {train_df.shape}")
        log_step(f"Test shape: {test_df.shape}")
        log_step(f"Train date range: {train_df[self.data_cfg.date_col].min()} -> {train_df[self.data_cfg.date_col].max()}")
        log_step(f"Test date range: {test_df[self.data_cfg.date_col].min()} -> {test_df[self.data_cfg.date_col].max()}")

        all_feature_cols = artifacts.feature_columns

        seq_feature_cols = [
            c for c in all_feature_cols
            if c.startswith("price_lag_")
            or c.startswith("sales_lag_")
            or c.startswith("price_roll_mean_")
            or c.startswith("sales_roll_mean_")
        ]

       

        log_step("Training SARIMA baseline...")
        sarima_metrics, sarima_params = SarimaTrainer(self.data_cfg, max_series=20).run(
            train_df=train_df,
            test_df=test_df,
            metrics_path=self.paths.metrics_dir / "sarima_metrics.json",
            preds_path=self.paths.metrics_dir / "sarima_predictions.json",
            do_search=True,
        )
        results_manager.append_result("sarima", sarima_metrics, sarima_params, notes="top-20 series")
        log_step(f"SARIMA done: {sarima_metrics}")

        log_step("Training Boosting...")
        boosting_metrics, boosting_params = BoostingTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_df,
            test_df=test_df,
            feature_cols=all_feature_cols,
            model_path=self.paths.models_dir / "boosting_model.pkl",
            metrics_path=self.paths.metrics_dir / "boosting_metrics.json",
            do_search=True,
        )
        results_manager.append_result("boosting", boosting_metrics, boosting_params)
        log_step(f"Boosting done: {boosting_metrics}")

        log_step("Training M5-style model...")
        m5_metrics, m5_params = M5StyleTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_df,
            test_df=test_df,
            feature_cols=all_feature_cols,
            model_path=self.paths.models_dir / "m5_style_model.pkl",
            metrics_path=self.paths.metrics_dir / "m5_style_metrics.json",
            do_search=True,
        )
        results_manager.append_result("m5_style", m5_metrics, m5_params, notes="adapted from M5 approach")
        log_step(f"M5-style done: {m5_metrics}")

        log_step("Training LSTM...")
        lstm_metrics, _ = LSTMTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_df,
            test_df=test_df,
            seq_feature_cols=seq_feature_cols,
            model_path=self.paths.models_dir / "lstm_model.pt",
            metrics_path=self.paths.metrics_dir / "lstm_metrics.json",
        )
        results_manager.append_result("lstm", lstm_metrics, {"seq_len": self.train_cfg.seq_len})
        log_step(f"LSTM done: {lstm_metrics}")

        log_step("Training Hybrid LSTM + Boosting...")
        hybrid_metrics = HybridTrainer(self.train_cfg, self.data_cfg).run(
            train_df=train_df,
            test_df=test_df,
            tabular_feature_cols=all_feature_cols,
            seq_feature_cols=seq_feature_cols,
            lstm_model_path=self.paths.models_dir / "lstm_model.pt",
            metrics_path=self.paths.metrics_dir / "hybrid_metrics.json",
        )
        results_manager.append_result(
            "hybrid_lstm_boosting",
            hybrid_metrics,
            None,
            notes="residual hybrid"
        )
        log_step(f"Hybrid done: {hybrid_metrics}")

        log_step("Final comparison table:")
        print(results_manager.load())

    def _prepare_dirs(self) -> None:
        for path in [
            self.paths.processed_dir,
            self.paths.models_dir,
            self.paths.reports_dir,
            self.paths.figures_dir,
            self.paths.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)