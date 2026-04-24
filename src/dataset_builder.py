from dataclasses import dataclass
import pandas as pd

from src.config import DataConfig, FeatureConfig
from src.data_loader import RawDataBundle


@dataclass
class DatasetArtifacts:
    panel: pd.DataFrame
    feature_columns: list[str]
    target_column: str


class PriceDatasetBuilder:
    """
    Строит единую аналитическую панель:
    - weekly sales
    - prices
    - calendar
    - lag / rolling features

    ВАЖНО:
    category/id-колонки остаются в feature_columns,
    потому что:
    - boosting умеет их использовать через preprocessing,
    - LSTM будет использовать их через embeddings.
    """

    def __init__(self, data_cfg: DataConfig, feat_cfg: FeatureConfig) -> None:
        self.data_cfg = data_cfg
        self.feat_cfg = feat_cfg

    def build(self, bundle: RawDataBundle) -> DatasetArtifacts:
        sales = bundle.sales.copy()
        calendar = bundle.calendar.copy()
        prices = bundle.prices.copy()

        series_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_cols = [c for c in sales.columns if c.startswith("d_")]

        sales_long = sales.melt(
            id_vars=series_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="sales_qty",
        )

        sales_long = sales_long.merge(
            calendar[
                [
                    "d",
                    "date",
                    "wm_yr_wk",
                    "month",
                    "year",
                    "wday",
                    "event_name_1",
                    "event_type_1",
                    "snap_CA",
                    "snap_TX",
                    "snap_WI",
                ]
            ],
            on="d",
            how="left",
        )

        weekly_sales = (
            sales_long.groupby(
                ["store_id", "item_id", "dept_id", "cat_id", "state_id", "wm_yr_wk"],
                as_index=False
            )["sales_qty"]
            .sum()
        )

        week_calendar = (
            calendar.groupby("wm_yr_wk", as_index=False)
            .agg(
                date=("date", "min"),
                month=("month", "first"),
                year=("year", "first"),
                wday=("wday", "first"),
                event_name_1=("event_name_1", "first"),
                event_type_1=("event_type_1", "first"),
                snap_CA=("snap_CA", "max"),
                snap_TX=("snap_TX", "max"),
                snap_WI=("snap_WI", "max"),
            )
        )

        panel = prices.merge(
            weekly_sales,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        panel = panel.merge(week_calendar, on="wm_yr_wk", how="left")
        panel["sales_qty"] = panel["sales_qty"].fillna(0)

        panel = panel.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        panel = self._filter_top_series(panel)
        panel = self._create_features(panel)
        panel = panel.dropna().reset_index(drop=True)

        # Исключаем только target/date/недельный ключ
        excluded_cols = {
            self.data_cfg.target_col,
            self.data_cfg.date_col,
            self.data_cfg.week_col,
            "id",
            "d",
        }

        feature_columns = [c for c in panel.columns if c not in excluded_cols]

        return DatasetArtifacts(
            panel=panel,
            feature_columns=feature_columns,
            target_column=self.data_cfg.target_col,
        )

    def _filter_top_series(self, panel: pd.DataFrame) -> pd.DataFrame:
        counts = (
            panel.groupby(["store_id", "item_id"])
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
        )

        if self.data_cfg.max_series is not None:
            top_series = counts.head(self.data_cfg.max_series)[["store_id", "item_id"]]
            panel = panel.merge(top_series, on=["store_id", "item_id"], how="inner")

        enough_hist = (
            panel.groupby(["store_id", "item_id"])
            .size()
            .reset_index(name="n_obs")
            .query("n_obs >= @self.data_cfg.min_history_weeks")
        )

        panel = panel.merge(
            enough_hist[["store_id", "item_id"]],
            on=["store_id", "item_id"],
            how="inner",
        )
        return panel

    def _create_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["store_id", "item_id"]

        # Лаги цены
        for lag in self.feat_cfg.price_lags:
            panel[f"price_lag_{lag}"] = (
                panel.groupby(group_cols)[self.data_cfg.target_col].shift(lag)
            )

        # Лаги продаж
        for lag in self.feat_cfg.sales_lags:
            panel[f"sales_lag_{lag}"] = (
                panel.groupby(group_cols)["sales_qty"].shift(lag)
            )

        # Скользящие средние
        for window in self.feat_cfg.rolling_windows:
            panel[f"price_roll_mean_{window}"] = (
                panel.groupby(group_cols)[self.data_cfg.target_col]
                .transform(lambda s: s.shift(1).rolling(window).mean())
            )

            panel[f"sales_roll_mean_{window}"] = (
                panel.groupby(group_cols)["sales_qty"]
                .transform(lambda s: s.shift(1).rolling(window).mean())
            )

        panel["price_diff_1"] = panel[self.data_cfg.target_col] - panel["price_lag_1"]

        # Календарные числовые признаки
        panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
        panel["quarter"] = panel["date"].dt.quarter
        panel["is_month_start"] = panel["date"].dt.is_month_start.astype(int)
        panel["is_month_end"] = panel["date"].dt.is_month_end.astype(int)

        # События как строки + бинарные derived features
        panel["event_name_1"] = panel["event_name_1"].fillna("NO_EVENT")
        panel["event_type_1"] = panel["event_type_1"].fillna("NO_EVENT")

        panel["has_event_1"] = (panel["event_name_1"] != "NO_EVENT").astype(int)
        panel["is_religious_event"] = (panel["event_type_1"] == "Religious").astype(int)
        panel["is_cultural_event"] = (panel["event_type_1"] == "Cultural").astype(int)
        panel["is_sporting_event"] = (panel["event_type_1"] == "Sporting").astype(int)
        panel["is_national_event"] = (panel["event_type_1"] == "National").astype(int)

        return panel