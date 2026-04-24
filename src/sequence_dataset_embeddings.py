import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PriceSequenceEmbeddingDataset(Dataset):
    """
    Dataset для LSTM с embeddings.

    На каждом таймшаге есть:
    - numeric features
    - categorical features (как integer indices)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        numeric_feature_cols: list[str],
        categorical_feature_cols: list[str],
        categorical_vocab_maps: dict[str, dict[str, int]],
        target_col: str,
        group_cols: list[str],
        date_col: str = "date",
        min_target_date=None,
    ) -> None:
        self.X_num = []
        self.X_cat = []
        self.y = []

        df = df.sort_values(group_cols + [date_col]).copy()

        # Преобразуем категории в индексы
        for col in categorical_feature_cols:
            vocab = categorical_vocab_maps[col]
            df[col] = df[col].astype(str).map(lambda x: vocab.get(x, 0)).astype(int)

        for _, group in df.groupby(group_cols):
            needed_cols = numeric_feature_cols + categorical_feature_cols + [target_col, date_col]
            values = group[needed_cols].reset_index(drop=True)

            if len(values) <= seq_len:
                continue

            for i in range(seq_len, len(values)):
                target_date = values.iloc[i][date_col]

                if min_target_date is not None and target_date < min_target_date:
                    continue

                x_num = values.iloc[i - seq_len:i][numeric_feature_cols].to_numpy(dtype=np.float32)
                x_cat = values.iloc[i - seq_len:i][categorical_feature_cols].to_numpy(dtype=np.int64)
                y_value = np.float32(values.iloc[i][target_col])

                self.X_num.append(x_num)
                self.X_cat.append(x_cat)
                self.y.append(y_value)

        self.X_num = np.asarray(self.X_num, dtype=np.float32)
        self.X_cat = np.asarray(self.X_cat, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X_num[idx]),
            torch.tensor(self.X_cat[idx]),
            torch.tensor(self.y[idx]),
        )