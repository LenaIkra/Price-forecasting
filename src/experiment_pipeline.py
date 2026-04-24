from datetime import datetime

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter
from src.feature_sets import FeatureSetBuilder
from src.experiment_tracker import ExperimentTracker

from src.train_sarima import SarimaTrainer
from src.train_boosting import BoostingTrainer
from src.train_m5_style import M5StyleTrainer
from src.train_lstm import LSTMTrainer
from src.train_hybrid import HybridTrainer


def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


class ExperimentPipeline:
    """
    Массовый запуск экспериментов:
    - строит единый датасет
    - создает feature sets
    - запускает все модели
    - сохраняет таблицы результатов
    - умеет продолжать с места остановки
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

    def run(self) -> None:
        self._prepare_dirs()

        tracker = ExperimentTracker(self.paths.reports_dir / "experiments")

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
        log_step(f"Common features count: {len(artifacts.feature_columns)}")

        splitter = TimeSeriesSplitter(
            date_col=self.data_cfg.date_col,
            test_size_weeks=self.train_cfg.test_size_weeks,
        )
        train_df, test_df = splitter.split(artifacts.panel)

        log_step(f"Train shape: {train_df.shape}")
        log_step(f"Test shape: {test_df.shape}")

        feature_sets = FeatureSetBuilder.build(artifacts.feature_columns)

        print("\nFeature sets:")
        for name, cols in feature_sets.items():
            print(f"{name}: {len(cols)} features")

        # --- SARIMA on all series ---
        sarima_params = None
        sarima_params_path = self.paths.metrics_dir / "sarima_best_params.json"

        if not tracker.already_done("sarima", "all_series_univariate"):
            log_step("Running SARIMA on all series...")

            sarima_metrics, sarima_params = SarimaTrainer(self.data_cfg, self.train_cfg).run(
                train_df=train_df,
                test_df=test_df,
                metrics_path=self.paths.metrics_dir / "sarima_metrics.json",
                preds_path=self.paths.metrics_dir / "sarima_test_predictions.csv",
                do_search=True,
                progress_path=self.paths.metrics_dir / "sarima_progress.csv",
            )

            tracker.add_result(
                model_name="sarima",
                feature_set_name="all_series_univariate",
                metrics=sarima_metrics,
                best_params=sarima_params,
                notes="all-series global-parameter SARIMA",
            )
        else:
            print("[SKIP] sarima + all_series_univariate")

        if sarima_params_path.exists():
            import json
            with open(sarima_params_path, "r", encoding="utf-8") as f:
                sarima_params = json.load(f)

        if sarima_params is None:
            raise RuntimeError("SARIMA params were not found.")

        # --- feature-set dependent models ---
        for fs_name, fs_cols in feature_sets.items():
            log_step(f"Running feature set: {fs_name}")

            # --- Boosting ---
            if not tracker.already_done("boosting", fs_name):
                log_step(f"[{fs_name}] Boosting...")

                boosting_metrics, boosting_params = BoostingTrainer(self.train_cfg, self.data_cfg).run(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=fs_cols,
                    model_path=self.paths.models_dir / f"boosting_{fs_name}.pkl",
                    metrics_path=self.paths.metrics_dir / f"boosting_{fs_name}_metrics.json",
                    do_search=True,
                    preds_path=self.paths.metrics_dir / f"boosting_{fs_name}_preds.csv",
                )

                tracker.add_result(
                    model_name="boosting",
                    feature_set_name=fs_name,
                    metrics=boosting_metrics,
                    best_params=boosting_params,
                )
            else:
                print(f"[SKIP] boosting + {fs_name}")

            # --- M5-style ---
            if not tracker.already_done("m5_style", fs_name):
                log_step(f"[{fs_name}] M5-style...")

                m5_metrics, m5_params = M5StyleTrainer(self.train_cfg, self.data_cfg).run(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=fs_cols,
                    model_path=self.paths.models_dir / f"m5_style_{fs_name}.pkl",
                    metrics_path=self.paths.metrics_dir / f"m5_style_{fs_name}_metrics.json",
                    do_search=True,
                )

                tracker.add_result(
                    model_name="m5_style",
                    feature_set_name=fs_name,
                    metrics=m5_metrics,
                    best_params=m5_params,
                    notes="adapted from M5 approach",
                )
            else:
                print(f"[SKIP] m5_style + {fs_name}")

            # --- LSTM with embeddings ---
            if not tracker.already_done("lstm_embeddings", fs_name):
                log_step(f"[{fs_name}] LSTM with embeddings...")

                lstm_metrics, _ = LSTMTrainer(self.train_cfg, self.data_cfg).run(
                    train_df=train_df,
                    test_df=test_df,
                    seq_feature_cols=fs_cols,
                    model_path=self.paths.models_dir / f"lstm_embeddings_{fs_name}.pt",
                    metrics_path=self.paths.metrics_dir / f"lstm_embeddings_{fs_name}_metrics.json",
                )

                tracker.add_result(
                    model_name="lstm_embeddings",
                    feature_set_name=fs_name,
                    metrics=lstm_metrics,
                    best_params={"seq_len": self.train_cfg.seq_len},
                )
            else:
                print(f"[SKIP] lstm_embeddings + {fs_name}")

            # --- Hybrid SARIMA + Boosting ---
           

    def _prepare_dirs(self) -> None:
        for path in [
            self.paths.processed_dir,
            self.paths.models_dir,
            self.paths.reports_dir,
            self.paths.figures_dir,
            self.paths.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)