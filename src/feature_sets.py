from typing import Dict, List


class FeatureSetBuilder:
    """
    Формирует разные наборы признаков для массовых экспериментов.
    """

    @staticmethod
    def build(common_feature_cols: list[str]) -> Dict[str, List[str]]:
        common_feature_cols = sorted(common_feature_cols)

        price_lag_cols = [c for c in common_feature_cols if c.startswith("price_lag_")]
        price_roll_cols = [c for c in common_feature_cols if c.startswith("price_roll_mean_")]
        sales_lag_cols = [c for c in common_feature_cols if c.startswith("sales_lag_")]
        sales_roll_cols = [c for c in common_feature_cols if c.startswith("sales_roll_mean_")]

        calendar_cols = [
            c for c in common_feature_cols
            if c in {
                "month",
                "year",
                "wday",
                "week_of_year",
                "quarter",
                "is_month_start",
                "is_month_end",
                "snap_CA",
                "snap_TX",
                "snap_WI",
                "has_event_1",
                "is_religious_event",
                "is_cultural_event",
                "is_sporting_event",
                "is_national_event",
            }
        ]

        price_diff_cols = [c for c in common_feature_cols if c == "price_diff_1"]

        feature_sets = {
            "all_features": common_feature_cols,
            "no_price_history": sorted(
                [c for c in common_feature_cols if c not in set(price_lag_cols + price_roll_cols + price_diff_cols)]
            ),
            "price_only_history": sorted(
                list(set(price_lag_cols + price_roll_cols + price_diff_cols + calendar_cols))
            ),
            "sales_calendar_only": sorted(
                list(set(sales_lag_cols + sales_roll_cols + calendar_cols))
            ),
        }

        return feature_sets