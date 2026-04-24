import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PriceSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: list[str],
        target_col: str,
        group_cols: list[str],
        date_col: str = "date",
        min_target_date=None,
    ) -> None:
        self.X_seq = []
        self.y = []
        self.target_dates = []

        df = df.sort_values(group_cols + [date_col]).copy()

        for _, group in df.groupby(group_cols):
            values = group[feature_cols + [target_col, date_col]].reset_index(drop=True)

            if len(values) <= seq_len:
                continue

            for i in range(seq_len, len(values)):
                target_date = values.iloc[i][date_col]

                if min_target_date is not None and target_date < min_target_date:
                    continue

                x_window = values.iloc[i - seq_len:i][feature_cols].to_numpy(dtype=np.float32)
                y_value = np.float32(values.iloc[i][target_col])

                self.X_seq.append(x_window)
                self.y.append(y_value)
                self.target_dates.append(target_date)

        self.X_seq = np.asarray(self.X_seq, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X_seq[idx]), torch.tensor(self.y[idx])