from abc import ABC, abstractmethod
from pathlib import Path
import joblib


class BaseForecastModel(ABC):
    @abstractmethod
    def fit(self, X, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        return joblib.load(path)