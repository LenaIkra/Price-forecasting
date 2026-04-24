from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.final_pipeline import FinalPipeline


def main() -> None:
    paths = PathsConfig()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    train_cfg = TrainConfig()

    pipeline = FinalPipeline(paths, data_cfg, feat_cfg, train_cfg)
    pipeline.run()


if __name__ == "__main__":
    main()