from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models_pkg.base_model import BaseForecastModel


@dataclass
class BoostingConfig:
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 8
    random_state: int = 42


class PriceBoostingModel(BaseForecastModel):
    """
    Boosting model:
    - numeric features -> median imputer
    - categorical features -> one-hot encoder
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], cfg: BoostingConfig):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cfg = cfg
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_cols),
                ("cat", categorical_pipe, self.categorical_cols),
            ],
            remainder="drop",
        )

        model = HistGradientBoostingRegressor(
            max_iter=self.cfg.max_iter,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.random_state,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def fit(self, X: pd.DataFrame, y) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)