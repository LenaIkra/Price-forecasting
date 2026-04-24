from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.experiment_pipeline import ExperimentPipeline


def main() -> None:
    paths = PathsConfig()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    train_cfg = TrainConfig()

    pipeline = ExperimentPipeline(paths, data_cfg, feat_cfg, train_cfg)
    pipeline.run()


if __name__ == "__main__":
    main()