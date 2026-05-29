from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.final_pipeline import FinalPipeline


def main() -> None:
    """
    Точка входа для итогового запуска экспериментальной части ВКР.

    В этом файле создаются конфигурации проекта и запускается FinalPipeline.
    Сам пайплайн отвечает за:
    - построение или загрузку аналитического датасета;
    - обучение SARIMA, градиентного бустинга, M5-style модели и LSTM;
    - построение гибридной модели;
    - сохранение метрик, прогнозов и итогового сравнения.

    Параметры force_rebuild_dataset и force_retrain_models позволяют
    управлять повторным запуском эксперимента:

    force_rebuild_dataset=False:
        если аналитический датасет уже сохранен, он будет загружен из parquet.

    force_rebuild_dataset=True:
        датасет будет построен заново из исходных файлов M5.

    force_retrain_models=False:
        если для модели уже сохранены метрики, прогнозы и файл модели,
        повторное обучение выполняться не будет.

    force_retrain_models=True:
        все модели будут обучены заново. Этот режим используется после
        изменения кода моделей или состава признаков.
    """

    paths = PathsConfig()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    train_cfg = TrainConfig()

    pipeline = FinalPipeline(
        paths=paths,
        data_cfg=data_cfg,
        feat_cfg=feat_cfg,
        train_cfg=train_cfg,

        # Для обычного повторного запуска оставляем False.
        # Для полной пересборки датасета после изменения dataset_builder.py
        # нужно временно поставить True.
        force_rebuild_dataset=False,

        # Для запуска после доработки моделей нужно временно поставить True.
        # После успешного финального прогона можно вернуть False, чтобы
        # использовать сохраненные модели и прогнозы.
        force_retrain_models=False,
    )

    pipeline.run()


if __name__ == "__main__":
    main()