import pandas as pd


class TimeSeriesSplitter:
    def __init__(self, date_col: str, test_size_weeks: int) -> None:
        self.date_col = date_col
        self.test_size_weeks = test_size_weeks

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(df[self.date_col].dropna().unique())
        split_point = unique_dates[-self.test_size_weeks]

        train_df = df[df[self.date_col] < split_point].copy()
        test_df = df[df[self.date_col] >= split_point].copy()

        return train_df, test_df