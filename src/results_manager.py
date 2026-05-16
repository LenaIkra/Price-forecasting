from pathlib import Path
import json
import pandas as pd


class ResultsManager:
    def __init__(self, results_csv_path: Path) -> None:
        self.results_csv_path = results_csv_path

    def append_result(
        self,
        model_name: str,
        metrics: dict,
        best_params: dict | None = None,
        notes: str | None = None,
    ) -> None:
        self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "model_name": model_name,
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "WAPE": metrics.get("WAPE"),
            "best_params": json.dumps(best_params, ensure_ascii=False) if best_params else "",
            "notes": notes or "",
        }

        if self.results_csv_path.exists():
            df = pd.read_csv(self.results_csv_path)
            df = df[df["model_name"] != model_name]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        df.to_csv(self.results_csv_path, index=False, encoding="utf-8-sig")

    def load(self) -> pd.DataFrame:
        if self.results_csv_path.exists():
            return pd.read_csv(self.results_csv_path)
        return pd.DataFrame(columns=["model_name", "MAE", "RMSE", "WAPE", "best_params", "notes"])