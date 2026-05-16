from pathlib import Path
import pandas as pd


class FinalResultsManager:
    """
    Хранит результаты финального прогона.

    Делает 2 таблицы:
    1. final_comparison.csv — как считали
    2. final_comparison_sorted.csv — отсортировано по RMSE
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.final_path = self.output_dir / "final_comparison.csv"
        self.sorted_path = self.output_dir / "final_comparison_sorted.csv"

        self.rows: list[dict] = []

    def add_result(
        self,
        model_name: str,
        feature_set_name: str,
        metrics: dict,
        best_params: dict | None = None,
        notes: str | None = None,
    ) -> None:
        row = {
            "model_name": model_name,
            "feature_set": feature_set_name,
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "WAPE": metrics.get("WAPE"),
            "best_params": str(best_params) if best_params else "",
            "notes": notes or "",
        }
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return

        df = pd.DataFrame(self.rows)
        df.to_csv(self.final_path, index=False, encoding="utf-8-sig")

        sorted_df = df.sort_values(["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
        sorted_df.to_csv(self.sorted_path, index=False, encoding="utf-8-sig")