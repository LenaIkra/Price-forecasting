from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    project_dir: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    metrics_dir: Path = Path("reports/metrics")


@dataclass(frozen=True)
class DataConfig:
    sales_file: str = "sales_train_evaluation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"

    target_col: str = "sell_price"
    series_id_cols: tuple[str, str] = ("store_id", "item_id")
    cat_cols: tuple[str, ...] = ("store_id", "item_id", "dept_id", "cat_id", "state_id")
    date_col: str = "date"
    week_col: str = "wm_yr_wk"

    # Берем все доступные ряды
    max_series: int | None = None
    min_history_weeks: int = 40
    forecast_horizon_weeks: int = 4


@dataclass(frozen=True)
class FeatureConfig:
    price_lags: tuple[int, ...] = (1, 2, 3, 4, 8, 12)
    sales_lags: tuple[int, ...] = (1, 2, 3, 4, 8, 12)
    rolling_windows: tuple[int, ...] = (4, 8, 12)


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size_weeks: int = 12
    # Ограничение числа рядов для табличных моделей в финальном прогоне
    final_tabular_max_series: int = 3000

    # Boosting
    boosting_max_iter: int = 300
    boosting_learning_rate: float = 0.05
    boosting_max_depth: int = 8

    # LSTM with embeddings
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 5
    lstm_batch_size: int = 256
    lstm_lr: float = 1e-3
    seq_len: int = 12

    # Embedding dimensions
    embedding_dim_item_id: int = 16
    embedding_dim_store_id: int = 4
    embedding_dim_dept_id: int = 4
    embedding_dim_cat_id: int = 3
    embedding_dim_state_id: int = 3

    # SARIMA global search
    sarima_search_series_count: int = 200
    sarima_orders: tuple[tuple[int, int, int], ...] = (
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
    )
    sarima_seasonal_orders: tuple[tuple[int, int, int, int], ...] = (
        (1, 0, 1, 12),
        (1, 1, 1, 12),
    )

    # Hybrid validation
    hybrid_val_weeks: int = 8
    hybrid_alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)