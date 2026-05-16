from pathlib import Path
import pandas as pd


class ExperimentTracker:
    """
    Класс для хранения результатов массовых экспериментов.

    Что делает:
    1. Хранит детальную таблицу со всеми прогонами.
    2. Делает summary-таблицу, отсортированную по качеству.
    3. Делает таблицу с лучшим результатом для каждой модели.
    4. Позволяет понять, какой эксперимент уже был посчитан,
       чтобы не запускать его повторно.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Файл со всеми запусками
        self.detailed_path = self.output_dir / "experiments_detailed.csv"

        # Та же таблица, но отсортированная по метрикам
        self.summary_path = self.output_dir / "experiments_summary.csv"

        # Лучшая конфигурация для каждой модели
        self.best_by_model_path = self.output_dir / "best_by_model.csv"

        # Не удаляем старые результаты — это важно для resume
        if self.detailed_path.exists():
            self.rows = pd.read_csv(self.detailed_path).to_dict("records")
        else:
            self.rows = []

        # Если уже есть данные — сразу пересобираем summary-файлы
        self._flush()

    def already_done(self, model_name: str, feature_set_name: str) -> bool:
        """
        Проверяет, был ли уже выполнен эксперимент
        для данной модели и данного набора признаков.
        """
        return any(
            row["model_name"] == model_name and row["feature_set"] == feature_set_name
            for row in self.rows
        )

    def add_result(
        self,
        model_name: str,
        feature_set_name: str,
        metrics: dict,
        best_params: dict | None = None,
        notes: str | None = None,
    ) -> None:
        """
        Добавляет новый результат эксперимента
        и обновляет все итоговые таблицы.
        """

        row = {
            "model_name": model_name,
            "feature_set": feature_set_name,
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "WAPE": metrics.get("WAPE"),
            "best_params": str(best_params) if best_params else "",
            "notes": notes or "",
        }

        # Если такая запись уже была, удаляем старую и заменяем новой
        self.rows = [
            existing_row
            for existing_row in self.rows
            if not (
                existing_row["model_name"] == model_name
                and existing_row["feature_set"] == feature_set_name
            )
        ]

        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        """
        Пересохраняет все итоговые CSV.
        """
        if not self.rows:
            return

        df = pd.DataFrame(self.rows)

        # Полная таблица экспериментов
        detailed = df.copy()
        detailed.to_csv(self.detailed_path, index=False, encoding="utf-8-sig")

        # Summary по качеству
        summary = df.sort_values(["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
        summary.to_csv(self.summary_path, index=False, encoding="utf-8-sig")

        # Лучшая строка на каждую модель
        best_by_model = (
            df.sort_values(["model_name", "RMSE", "MAE"], ascending=[True, True, True])
            .groupby("model_name", as_index=False)
            .first()
            .sort_values("RMSE", ascending=True)
            .reset_index(drop=True)
        )
        best_by_model.to_csv(self.best_by_model_path, index=False, encoding="utf-8-sig")