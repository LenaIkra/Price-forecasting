import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PathsConfig, DataConfig, FeatureConfig, TrainConfig
from src.data_loader import DataLoader
from src.dataset_builder import PriceDatasetBuilder
from src.splits import TimeSeriesSplitter
from src.train_m5_style import M5StyleTrainer


def calc_metrics(y_true, y_pred):
    err = y_true - y_pred

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": float(wape),
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
        ci[f"{metric}_ci_lower"] = float(boot_df[metric].quantile(0.025))
        ci[f"{metric}_ci_upper"] = float(boot_df[metric].quantile(0.975))

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


def extract_model(loaded_object):
    """
    На случай, если модель сохранена не напрямую, а внутри словаря.
    Например:
    {
        "model": trained_model,
        "params": ...
    }
    """
    if hasattr(loaded_object, "predict"):
        return loaded_object

    if isinstance(loaded_object, dict):
        for key in ["model", "best_model", "final_model", "lgbm_model"]:
            if key in loaded_object and hasattr(loaded_object[key], "predict"):
                return loaded_object[key]

    raise TypeError(
        "Не удалось извлечь модель: загруженный объект не имеет метода predict()."
    )


def align_lightgbm_categories(model, df, categorical_cols):
    """
    Приводит категориальные признаки к тем же категориям,
    которые были у LightGBM при обучении.

    Это нужно, чтобы избежать ошибки:
    ValueError: train and valid dataset categorical_feature do not match.
    """

    df = df.copy()

    pandas_categorical = None

    if hasattr(model, "booster_"):
        pandas_categorical = getattr(model.booster_, "pandas_categorical", None)

    if pandas_categorical is None and hasattr(model, "_Booster"):
        pandas_categorical = getattr(model._Booster, "pandas_categorical", None)

    if pandas_categorical is not None:
        for col, categories in zip(categorical_cols, pandas_categorical):
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str), categories=categories)
    else:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

    return df


def evaluate_subset(
    subset_name,
    train_df,
    test_df,
    model,
    trainer,
    feature_cols,
    data_cfg,
    n_series,
):
    """
    Оценивает M5-style модель на заданной подвыборке.

    Важно:
    - M5-style модель использует дополнительные иерархические признаки;
    - признаки готовятся тем же способом, что и при обучении;
    - модель обучалась на log1p(sell_price), поэтому прогноз возвращается
      в исходную шкалу через expm1.
    """

    prepared_train_df, prepared_test_df, final_feature_cols, numeric_cols, categorical_cols = (
        trainer._prepare_features(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )
    )

    y_true = prepared_test_df[data_cfg.target_col].values

    X_test = prepared_test_df[final_feature_cols].copy()
    X_test = align_lightgbm_categories(
        model=model,
        df=X_test,
        categorical_cols=categorical_cols,
    )

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

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

    reports_dir = paths.reports_dir / "final_run"
    model_path = paths.models_dir / "final_run" / "m5_style_model.pkl"

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

    print("Loading trained M5-style model...")
    loaded_object = joblib.load(model_path)
    model = extract_model(loaded_object)

    trainer = M5StyleTrainer(
        train_cfg=train_cfg,
        data_cfg=data_cfg,
    )

    # Основная выборка top-5000
    top_5000 = get_top_series(train_df, data_cfg, 5000)

    train_top_5000, test_top_5000 = make_subset(
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

    train_independent, test_independent = make_subset(
        train_df=train_df,
        test_df=test_df,
        series_df=independent_series,
        data_cfg=data_cfg,
    )

    print("Evaluating top-5000 sample...")
    result_main = evaluate_subset(
        subset_name="Основная тестовая выборка top-5000",
        train_df=train_top_5000,
        test_df=test_top_5000,
        model=model,
        trainer=trainer,
        feature_cols=feature_cols,
        data_cfg=data_cfg,
        n_series=5000,
    )

    print("Evaluating independent sample...")
    result_independent = evaluate_subset(
        subset_name="Независимая подвыборка",
        train_df=train_independent,
        test_df=test_independent,
        model=model,
        trainer=trainer,
        feature_cols=feature_cols,
        data_cfg=data_cfg,
        n_series=1000,
    )

    results_df = pd.DataFrame([result_main, result_independent])

    output_csv = output_dir / "m5_style_metrics_with_ci.csv"
    output_json = output_dir / "m5_style_metrics_with_ci.json"

    results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

    print(results_df)
    print(f"Saved to: {output_csv}")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()