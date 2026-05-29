from pathlib import Path

import pandas as pd


def main():
    """
    Расчет потенциального экономического эффекта для лучшей модели M5-style Boosting.

    Скрипт:
    1. Загружает прогнозы модели M5-style.
    2. Загружает аналитическую панель, чтобы взять фактические продажи.
    3. Соединяет прогнозы с продажами по store_id, item_id и date.
    4. Считает отклонение прогнозируемой цены от фактической.
    5. Оценивает потенциальное отклонение выручки с учетом объема продаж.
    6. Сохраняет итоговые таблицы для ВКР.
    """

    reports_dir = Path("reports/final_run")

    preds_path = reports_dir / "m5_style" / "predictions.csv"
    panel_path = reports_dir / "price_panel_final.parquet"

    output_dir = reports_dir / "economic_effect"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загружаем прогнозы лучшей модели
    preds = pd.read_csv(preds_path)

    # 2. Приводим названия колонок к единому виду
    # В разных версиях pipeline прогноз мог называться prediction или y_pred.
    if "prediction" in preds.columns and "y_pred" not in preds.columns:
        preds = preds.rename(columns={"prediction": "y_pred"})

    if "target" in preds.columns and "y_true" not in preds.columns:
        preds = preds.rename(columns={"target": "y_true"})

    required_pred_cols = {"store_id", "item_id", "date", "y_true", "y_pred"}
    missing_pred_cols = required_pred_cols - set(preds.columns)

    if missing_pred_cols:
        raise ValueError(
            f"В файле прогнозов не хватает колонок: {missing_pred_cols}. "
            f"Фактические колонки: {preds.columns.tolist()}"
        )

    # 3. Загружаем аналитический набор, чтобы взять продажи
    panel = pd.read_parquet(
        panel_path,
        columns=["store_id", "item_id", "date", "sales_qty"],
    )

    # 4. Приводим даты к одному типу
    preds["date"] = pd.to_datetime(preds["date"])
    panel["date"] = pd.to_datetime(panel["date"])

    # 5. Присоединяем продажи к прогнозам
    df = preds.merge(
        panel,
        on=["store_id", "item_id", "date"],
        how="left",
    )

    df = df.rename(columns={"sales_qty": "sales"})

    # Если по каким-то строкам продажи не подтянулись, заменяем на 0
    df["sales"] = df["sales"].fillna(0)

    # 6. Считаем отклонение прогноза цены
    df["price_deviation"] = df["y_true"] - df["y_pred"]

    # 7. Считаем абсолютное отклонение цены
    df["abs_price_deviation"] = df["price_deviation"].abs()

    # 8. Оцениваем потенциальное влияние ошибки цены на выручку
    df["revenue_deviation"] = df["sales"] * df["price_deviation"]
    df["abs_revenue_deviation"] = df["sales"] * df["abs_price_deviation"]

    # 9. Сводные показатели
    summary = pd.DataFrame(
        {
            "metric": [
                "Количество наблюдений",
                "Суммарные продажи",
                "Среднее абсолютное отклонение цены",
                "Суммарное абсолютное отклонение выручки",
                "Среднее абсолютное отклонение выручки на наблюдение",
            ],
            "value": [
                len(df),
                df["sales"].sum(),
                df["abs_price_deviation"].mean(),
                df["abs_revenue_deviation"].sum(),
                df["abs_revenue_deviation"].mean(),
            ],
        }
    )

    # 10. Разделяем сценарии недооценки и переоценки цены
    underpricing = df[df["y_pred"] < df["y_true"]]
    overpricing = df[df["y_pred"] > df["y_true"]]

    scenario_summary = pd.DataFrame(
        {
            "scenario": [
                "Недооценка цены моделью",
                "Переоценка цены моделью",
            ],
            "observations": [
                len(underpricing),
                len(overpricing),
            ],
            "sales_sum": [
                underpricing["sales"].sum(),
                overpricing["sales"].sum(),
            ],
            "abs_revenue_deviation": [
                underpricing["abs_revenue_deviation"].sum(),
                overpricing["abs_revenue_deviation"].sum(),
            ],
        }
    )

    # 11. Сохраняем результаты
    df.to_csv(
        output_dir / "economic_effect_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output_dir / "economic_effect_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scenario_summary.to_csv(
        output_dir / "economic_effect_by_scenario.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSummary:")
    print(summary)

    print("\nScenario summary:")
    print(scenario_summary)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()