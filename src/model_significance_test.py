from pathlib import Path
import json

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_BOOTSTRAP = 1000


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))


def find_prediction_column(df: pd.DataFrame) -> str:
    """
    В разных файлах прогнозы могли называться по-разному:
    prediction или y_pred.
    Эта функция сама находит нужную колонку.
    """
    if "prediction" in df.columns:
        return "prediction"
    if "y_pred" in df.columns:
        return "y_pred"

    raise ValueError(f"Не найдена колонка с прогнозом. Колонки файла: {df.columns.tolist()}")


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    pred_col = find_prediction_column(df)

    result = df.copy()
    result = result.rename(columns={pred_col: f"pred_{model_name}"})

    needed_cols = ["y_true", f"pred_{model_name}"]

    # Если есть ключевые колонки — оставляем их для корректного merge
    key_cols = [col for col in ["store_id", "item_id", "date"] if col in result.columns]

    return result[key_cols + needed_cols]


def bootstrap_metric_diff(
    y_true,
    pred_a,
    pred_b,
    metric_func,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    """
    Считает bootstrap-интервал для разницы метрик.

    diff = metric(model_a) - metric(model_b)

    Если diff > 0, значит model_a хуже model_b,
    потому что ошибка у model_a больше.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        metric_a = metric_func(y_true[idx], pred_a[idx])
        metric_b = metric_func(y_true[idx], pred_b[idx])

        diffs.append(metric_a - metric_b)

    diffs = np.array(diffs)

    return {
        "diff_mean": float(np.mean(diffs)),
        "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
    }


def compare_models(df, model_a, model_b):
    """
    model_a сравнивается с model_b.

    diff = error(model_a) - error(model_b)

    Если diff > 0 — model_a хуже.
    Если diff < 0 — model_a лучше.
    """
    y_true = df["y_true"].to_numpy(dtype=float)
    pred_a = df[f"pred_{model_a}"].to_numpy(dtype=float)
    pred_b = df[f"pred_{model_b}"].to_numpy(dtype=float)

    rows = []

    for metric_name, metric_func in [
        ("RMSE", rmse),
        ("MAE", mae),
        ("WAPE", wape),
    ]:
        result = bootstrap_metric_diff(
            y_true=y_true,
            pred_a=pred_a,
            pred_b=pred_b,
            metric_func=metric_func,
        )

        metric_a_value = metric_func(y_true, pred_a)
        metric_b_value = metric_func(y_true, pred_b)

        if result["ci_95_low"] > 0:
            conclusion = f"{model_b} statistically better"
        elif result["ci_95_high"] < 0:
            conclusion = f"{model_a} statistically better"
        else:
            conclusion = "difference is not statistically significant"

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                f"{model_a}_value": metric_a_value,
                f"{model_b}_value": metric_b_value,
                "diff_a_minus_b": result["diff_mean"],
                "ci_95_low": result["ci_95_low"],
                "ci_95_high": result["ci_95_high"],
                "conclusion": conclusion,
            }
        )

    return rows


def main():
    reports_dir = Path("reports/final_run")
    output_dir = reports_dir / "significance_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    boosting_path = reports_dir / "boosting" / "predictions.csv"
    m5_path = reports_dir / "m5_style" / "predictions.csv"
    hybrid_path = reports_dir / "hybrid_boosting_m5" / "predictions.csv"

    boosting = load_predictions(boosting_path, "boosting")
    m5 = load_predictions(m5_path, "m5_style")
    hybrid = load_predictions(hybrid_path, "hybrid")

    # Если во всех файлах есть store_id, item_id, date — объединяем по ним.
    # Если в каком-то файле этих колонок нет, объединяем по индексу.
    if all(col in boosting.columns for col in ["store_id", "item_id", "date"]) and \
       all(col in m5.columns for col in ["store_id", "item_id", "date"]) and \
       all(col in hybrid.columns for col in ["store_id", "item_id", "date"]):

        df = boosting.merge(
            m5.drop(columns=["y_true"]),
            on=["store_id", "item_id", "date"],
            how="inner",
        ).merge(
            hybrid.drop(columns=["y_true"]),
            on=["store_id", "item_id", "date"],
            how="inner",
        )

    else:
        min_len = min(len(boosting), len(m5), len(hybrid))

        df = pd.DataFrame(
            {
                "y_true": m5["y_true"].iloc[:min_len].to_numpy(),
                "pred_boosting": boosting["pred_boosting"].iloc[:min_len].to_numpy(),
                "pred_m5_style": m5["pred_m5_style"].iloc[:min_len].to_numpy(),
                "pred_hybrid": hybrid["pred_hybrid"].iloc[:min_len].to_numpy(),
            }
        )

    print(f"Comparison dataset shape: {df.shape}")

    all_rows = []

    # Главное сравнение: M5-style против гибрида
    all_rows.extend(compare_models(df, "hybrid", "m5_style"))

    # Дополнительно: M5-style против обычного boosting
    all_rows.extend(compare_models(df, "boosting", "m5_style"))

    # Дополнительно: hybrid против обычного boosting
    all_rows.extend(compare_models(df, "boosting", "hybrid"))

    result_df = pd.DataFrame(all_rows)

    csv_path = output_dir / "model_significance_tests.csv"
    json_path = output_dir / "model_significance_tests.json"

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=4)

    print(result_df)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()