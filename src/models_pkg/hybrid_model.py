import numpy as np
import pandas as pd

from src.models_pkg.boosting_model import PriceBoostingModel, BoostingConfig


class ResidualHybridModel:
    """
    Гибридная модель:
    1. LSTM предсказывает базовый сигнал.
    2. Boosting обучается на остатках.
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], cfg: BoostingConfig):
        self.residual_model = PriceBoostingModel(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            cfg=cfg,
        )

    def fit(self, X_train: pd.DataFrame, residual_train: np.ndarray) -> None:
        self.residual_model.fit(X_train, residual_train)

    def predict(self, X_test: pd.DataFrame, lstm_pred: np.ndarray) -> np.ndarray:
        residual_pred = self.residual_model.predict(X_test)
        return lstm_pred + residual_pred