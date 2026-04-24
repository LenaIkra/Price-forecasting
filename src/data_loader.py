from dataclasses import dataclass
import pandas as pd

from src.config import PathsConfig, DataConfig


@dataclass
class RawDataBundle:
    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


class DataLoader:
    def __init__(self, paths: PathsConfig, data_cfg: DataConfig) -> None:
        self.paths = paths
        self.data_cfg = data_cfg

    def load(self) -> RawDataBundle:
        sales = pd.read_csv(self.paths.raw_dir / self.data_cfg.sales_file)
        calendar = pd.read_csv(self.paths.raw_dir / self.data_cfg.calendar_file)
        prices = pd.read_csv(self.paths.raw_dir / self.data_cfg.prices_file)

        calendar["date"] = pd.to_datetime(calendar["date"])

        return RawDataBundle(sales=sales, calendar=calendar, prices=prices)