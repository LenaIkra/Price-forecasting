import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter


def calc_metrics(y_true, y_pred):
    err = y_true - y_pred

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    }


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rows.append(calc_metrics(y_true[idx], y_pred[idx]))

    boot_df = pd.DataFrame(rows)

    ci = {}
    for metric in ["MAE", "RMSE", "WAPE"]:
        ci[f"{metric}_ci_lower"] = boot_df[metric].quantile(0.025)
        ci[f"{metric}_ci_upper"] = boot_df[metric].quantile(0.975)

    return ci


def get_top_series(train_df, data_cfg, n_series):
    group_cols = list(data_cfg.series_id_cols)

    return (
        train_df.groupby(group_cols)
        .size()
        .reset_index(name="n_obs")
        .sort_values("n_obs", ascending=False)
        .head(n_series)
    )


def make_subset(train_df, test_df, series_df, data_cfg):
    group_cols = list(data_cfg.series_id_cols)

    train_sub = train_df.merge(series_df[group_cols], on=group_cols, how="inner")
    test_sub = test_df.merge(series_df[group_cols], on=group_cols, how="inner")

    return train_sub.reset_index(drop=True), test_sub.reset_index(drop=True)


def evaluate_subset(
    subset_name,
    test_df,
    model,
    feature_cols,
    data_cfg,
    n_series,
):
    y_true = test_df[data_cfg.target_col].values
    y_pred = model.predict(test_df[feature_cols])

    metrics = calc_metrics(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred)

    result = {
        "sample": subset_name,
        "series_count": n_series,
        **metrics,
        **ci,
    }

    return result


def main():
    paths = PathsConfig()
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    train_cfg = TrainConfig()

    reports_dir = paths.reports_dir / "baseline_run"
    model_path = paths.models_dir / "baseline_run" / "boosting_model.pkl"

    output_dir = reports_dir / "independent_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    loader = DataLoader(paths, data_cfg)
    bundle = loader.load()

    print("Building analytical dataset...")
    builder = PriceDatasetBuilder(data_cfg, feat_cfg)
    artifacts = builder.build(bundle)

    feature_cols = artifacts.feature_columns

    print("Splitting train/test...")
    splitter = TimeSeriesSplitter(
        date_col=data_cfg.date_col,
        test_size_weeks=train_cfg.test_size_weeks,
    )

    train_df, test_df = splitter.split(artifacts.panel)

    print("Loading trained boosting model...")
    model = joblib.load(model_path)

    # Основная выборка top-5000
    top_5000 = get_top_series(train_df, data_cfg, 5000)

    _, test_top_5000 = make_subset(
        train_df=train_df,
        test_df=test_df,
        series_df=top_5000,
        data_cfg=data_cfg,
    )

    # Независимая подвыборка: следующие 1000 рядов после top-5000
    all_ranked_series = get_top_series(train_df, data_cfg, 6000)

    group_cols = list(data_cfg.series_id_cols)

    independent_series = all_ranked_series.merge(
        top_5000[group_cols],
        on=group_cols,
        how="left",
        indicator=True,
    )

    independent_series = (
        independent_series[independent_series["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .head(1000)
    )

    _, test_independent = make_subset(
        train_df=train_df,
        test_df=test_df,
        series_df=independent_series,
        data_cfg=data_cfg,
    )

    print("Evaluating top-5000 sample...")
    result_main = evaluate_subset(
        subset_name="Основная тестовая выборка top-5000",
        test_df=test_top_5000,
        model=model,
        feature_cols=feature_cols,
        data_cfg=data_cfg,
        n_series=5000,
    )

    print("Evaluating independent sample...")
    result_independent = evaluate_subset(
        subset_name="Независимая подвыборка",
        test_df=test_independent,
        model=model,
        feature_cols=feature_cols,
        data_cfg=data_cfg,
        n_series=1000,
    )

    results_df = pd.DataFrame([result_main, result_independent])

    output_csv = output_dir / "boosting_metrics_with_ci.csv"
    output_json = output_dir / "boosting_metrics_with_ci.json"

    results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

    print(results_df)
    print(f"Saved to: {output_csv}")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()