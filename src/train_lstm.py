from pathlib import Path
import json
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig, DataConfig
from src.metrics import evaluate_regression, save_metrics
from src.sequence_dataset_embeddings import PriceSequenceEmbeddingDataset
from src.models_pkg.lstm_embedding_model import LSTMEmbeddingRegressor


class LSTMTrainer:
    """
    Тренер нейросетевой модели LSTM с embedding-представлениями категориальных признаков.

    В экспериментальной части ВКР данная модель используется как нейросетевой
    подход к прогнозированию цены. В отличие от градиентного бустинга, LSTM
    получает на вход не отдельную строку таблицы, а последовательность наблюдений
    фиксированной длины. Это позволяет модели учитывать локальную временную
    динамику ценового ряда.

    Числовые признаки передаются в модель как последовательность float-векторов.
    Категориальные признаки, например item_id, store_id, dept_id, cat_id,
    state_id, преобразуются в embedding-представления. Такой подход позволяет
    модели использовать информацию о товарной иерархии и принадлежности ряда
    к конкретному магазину или товарной группе.

    Для контроля переобучения используется validation-период, выделенный по
    времени из конца обучающей выборки. Случайное перемешивание при разбиении
    не используется, так как для временных рядов оно приводит к утечке будущей
    информации в обучение.
    """

    def __init__(self, train_cfg: TrainConfig, data_cfg: DataConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Разделяет признаки на числовые и категориальные.

        Числовые признаки используются напрямую, а категориальные признаки
        передаются в модель через embedding-слои.
        """
        numeric_cols = [
            c for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        categorical_cols = [
            c for c in feature_cols
            if c in df.columns and c not in numeric_cols
        ]

        return numeric_cols, categorical_cols

    def _time_validation_split(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Формирует внутренние train/validation выборки по времени.

        Validation-период берется из последних дат обучающей выборки.
        Это имитирует реальную ситуацию прогнозирования, когда модель
        обучается на прошлом и проверяется на более позднем периоде.
        """
        date_col = self.data_cfg.date_col

        sorted_df = train_df.sort_values(date_col).reset_index(drop=True)
        unique_dates = sorted_df[date_col].drop_duplicates().sort_values().to_list()

        if len(unique_dates) < 5:
            raise ValueError(
                "[LSTM-Emb] Недостаточно уникальных дат для time-based validation."
            )

        valid_size = max(1, int(len(unique_dates) * 0.15))
        valid_dates = set(unique_dates[-valid_size:])

        inner_train_df = sorted_df[~sorted_df[date_col].isin(valid_dates)].copy()
        inner_valid_df = sorted_df[sorted_df[date_col].isin(valid_dates)].copy()

        return (
            inner_train_df.reset_index(drop=True),
            inner_valid_df.reset_index(drop=True),
        )

    def _build_vocab_maps(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
    ) -> dict[str, dict[str, int]]:
        """
        Для каждой категориальной колонки строит отображение:
        category_value -> integer index.

        Индекс 0 резервируется под unknown-значения. Это нужно для устойчивой
        работы на тестовом периоде: если в тесте появится категория, которой
        не было в обучении, модель получит индекс 0, а не упадет с ошибкой.

        Словарь строится только на обучающей части, чтобы не использовать
        информацию о составе тестового периода при обучении модели.
        """
        vocab_maps: dict[str, dict[str, int]] = {}

        for col in categorical_cols:
            unique_values = sorted(df[col].astype(str).dropna().unique().tolist())
            vocab_maps[col] = {
                value: idx + 1
                for idx, value in enumerate(unique_values)
            }

        return vocab_maps

    def _embedding_dims(self, categorical_cols: list[str]) -> dict[str, int]:
        """
        Задает размерность embedding-слоя для категориальных признаков.

        Для основных M5-идентификаторов используются параметры из TrainConfig.
        Для остальных категориальных признаков применяется небольшая размерность
        по умолчанию, чтобы не раздувать число параметров модели.
        """
        mapping = {
            "item_id": self.train_cfg.embedding_dim_item_id,
            "store_id": self.train_cfg.embedding_dim_store_id,
            "dept_id": self.train_cfg.embedding_dim_dept_id,
            "cat_id": self.train_cfg.embedding_dim_cat_id,
            "state_id": self.train_cfg.embedding_dim_state_id,
            "event_name_1": 8,
            "event_type_1": 4,
        }

        return {col: mapping.get(col, 4) for col in categorical_cols}

    def _build_train_dataset(
        self,
        train_df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        vocab_maps: dict[str, dict[str, int]],
    ) -> PriceSequenceEmbeddingDataset:
        """
        Создает dataset для обучающей части.
        """
        return PriceSequenceEmbeddingDataset(
            df=train_df,
            seq_len=self.train_cfg.seq_len,
            numeric_feature_cols=numeric_cols,
            categorical_feature_cols=categorical_cols,
            categorical_vocab_maps=vocab_maps,
            target_col=self.data_cfg.target_col,
            group_cols=list(self.data_cfg.series_id_cols),
            date_col=self.data_cfg.date_col,
            min_target_date=None,
        )

    def _build_future_dataset(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        vocab_maps: dict[str, dict[str, int]],
    ) -> PriceSequenceEmbeddingDataset:
        """
        Создает dataset для validation или test.

        Для построения последовательностей используется история до целевого
        периода. Это важно для LSTM: чтобы спрогнозировать дату из validation
        или test, модели нужна предыдущая история ряда.
        """
        full_df = pd.concat([history_df, target_df], ignore_index=True)
        min_target_date = target_df[self.data_cfg.date_col].min()

        return PriceSequenceEmbeddingDataset(
            df=full_df,
            seq_len=self.train_cfg.seq_len,
            numeric_feature_cols=numeric_cols,
            categorical_feature_cols=categorical_cols,
            categorical_vocab_maps=vocab_maps,
            target_col=self.data_cfg.target_col,
            group_cols=list(self.data_cfg.series_id_cols),
            date_col=self.data_cfg.date_col,
            min_target_date=min_target_date,
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """
        Создает DataLoader для обучения или оценки.
        """
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.lstm_batch_size,
            shuffle=shuffle,
        )

    def _build_model(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        vocab_maps: dict[str, dict[str, int]],
    ) -> tuple[LSTMEmbeddingRegressor, dict, dict]:
        """
        Инициализирует LSTM-модель с embedding-слоями.
        """
        categorical_cardinalities = {
            col: max(vocab_maps[col].values(), default=0) + 1
            for col in categorical_cols
        }

        embedding_dims = self._embedding_dims(categorical_cols)

        model = LSTMEmbeddingRegressor(
            num_numeric_features=len(numeric_cols),
            categorical_cardinalities=categorical_cardinalities,
            embedding_dims=embedding_dims,
            hidden_size=self.train_cfg.lstm_hidden_size,
            num_layers=self.train_cfg.lstm_num_layers,
            dropout=self.train_cfg.lstm_dropout,
        ).to(self.device)

        return model, categorical_cardinalities, embedding_dims

    def _evaluate_loader(
        self,
        model: LSTMEmbeddingRegressor,
        loader: DataLoader,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Считает loss, прогнозы и истинные значения на переданном DataLoader.
        """
        criterion = torch.nn.MSELoss()

        model.eval()
        losses = []
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x_num, x_cat, y_batch in loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                y_batch_device = y_batch.to(self.device)

                preds = model(x_num, x_cat)
                loss = criterion(preds, y_batch_device)

                losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(y_batch.numpy().tolist())

        mean_loss = float(np.mean(losses)) if losses else np.inf

        return mean_loss, np.array(all_preds), np.array(all_true)

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_feature_cols: list[str],
        model_path: Path,
        metrics_path: Path,
        preds_path: Path | None = None,
    ) -> tuple[dict, dict]:
        """
        Обучает LSTM-модель, сохраняет лучшую модель, прогнозы и метрики.

        В отличие от предыдущей версии, сохраняется не последняя эпоха,
        а лучший чекпоинт по validation loss. Это снижает риск переобучения
        и делает нейросетевой эксперимент более корректным.
        """
        print(f"[LSTM-Emb] device: {self.device}")
        print("[LSTM-Emb] splitting numeric/categorical features...")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if preds_path is not None:
            preds_path.parent.mkdir(parents=True, exist_ok=True)

        numeric_cols, categorical_cols = self._split_features(
            train_df,
            seq_feature_cols,
        )

        print(
            f"[LSTM-Emb] numeric_features={len(numeric_cols)} | "
            f"categorical_features={len(categorical_cols)}"
        )

        inner_train_df, inner_valid_df = self._time_validation_split(train_df)

        vocab_maps = self._build_vocab_maps(inner_train_df, categorical_cols)

        train_dataset = self._build_train_dataset(
            inner_train_df,
            numeric_cols,
            categorical_cols,
            vocab_maps,
        )

        valid_dataset = self._build_future_dataset(
            history_df=inner_train_df,
            target_df=inner_valid_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            vocab_maps=vocab_maps,
        )

        test_dataset = self._build_future_dataset(
            history_df=train_df,
            target_df=test_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            vocab_maps=vocab_maps,
        )

        print(
            f"[LSTM-Emb] train sequences={len(train_dataset)} | "
            f"valid sequences={len(valid_dataset)} | "
            f"test sequences={len(test_dataset)} | "
            f"seq_len={self.train_cfg.seq_len}"
        )

        if len(train_dataset) == 0:
            raise ValueError("[LSTM-Emb] Training dataset is empty.")

        if len(valid_dataset) == 0:
            raise ValueError("[LSTM-Emb] Validation dataset is empty.")

        if len(test_dataset) == 0:
            raise ValueError("[LSTM-Emb] Test dataset is empty.")

        train_loader = self._make_loader(train_dataset, shuffle=True)
        valid_loader = self._make_loader(valid_dataset, shuffle=False)
        test_loader = self._make_loader(test_dataset, shuffle=False)

        model, categorical_cardinalities, embedding_dims = self._build_model(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            vocab_maps=vocab_maps,
        )

        print(
            f"[LSTM-Emb] model initialized | "
            f"num_numeric={len(numeric_cols)} | "
            f"num_categorical={len(categorical_cols)}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_cfg.lstm_lr,
        )

        criterion = torch.nn.MSELoss()

        max_epochs = self.train_cfg.lstm_epochs

        # patience задает, сколько эпох модель может не улучшать validation loss.
        # Если отдельного параметра нет в TrainConfig, используется безопасное
        # значение по умолчанию.
        patience = getattr(self.train_cfg, "lstm_patience", 5)

        best_valid_loss = np.inf
        best_epoch = 0
        best_state_dict = None
        epochs_without_improvement = 0

        print(
            f"[LSTM-Emb] training started | epochs={max_epochs} | "
            f"batch_size={self.train_cfg.lstm_batch_size} | "
            f"lr={self.train_cfg.lstm_lr} | patience={patience}"
        )

        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []

            for batch_idx, (x_num, x_cat, y_batch) in enumerate(train_loader, start=1):
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                preds = model(x_num, x_cat)
                loss = criterion(preds, y_batch)

                loss.backward()

                # Ограничение нормы градиента повышает устойчивость обучения LSTM.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_losses.append(loss.item())

                if batch_idx % 100 == 0:
                    print(
                        f"[LSTM-Emb] epoch {epoch + 1}/{max_epochs} | "
                        f"batch {batch_idx}/{len(train_loader)} | "
                        f"batch_loss={loss.item():.6f}"
                    )

            mean_train_loss = float(np.mean(epoch_losses))

            valid_loss, _, _ = self._evaluate_loader(model, valid_loader)

            print(
                f"[LSTM-Emb] epoch {epoch + 1}/{max_epochs} completed | "
                f"train_loss={mean_train_loss:.6f} | "
                f"valid_loss={valid_loss:.6f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

                torch.save(best_state_dict, model_path)

                print(
                    f"[LSTM-Emb] best checkpoint updated | "
                    f"epoch={best_epoch} | valid_loss={best_valid_loss:.6f} | "
                    f"path={model_path}"
                )

            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"[LSTM-Emb] early stopping triggered | "
                        f"best_epoch={best_epoch} | "
                        f"best_valid_loss={best_valid_loss:.6f}"
                    )
                    break

        if best_state_dict is None:
            raise RuntimeError("[LSTM-Emb] No model checkpoint was saved.")

        model.load_state_dict(best_state_dict)

        print("[LSTM-Emb] evaluating best checkpoint on test...")

        _, all_preds, all_true = self._evaluate_loader(model, test_loader)

        metrics = evaluate_regression(all_true, all_preds)
        save_metrics(metrics, metrics_path)

        params = {
            "model_type": "LSTMEmbeddingRegressor",
            "seq_len": self.train_cfg.seq_len,
            "hidden_size": self.train_cfg.lstm_hidden_size,
            "num_layers": self.train_cfg.lstm_num_layers,
            "dropout": self.train_cfg.lstm_dropout,
            "lr": self.train_cfg.lstm_lr,
            "batch_size": self.train_cfg.lstm_batch_size,
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "best_valid_loss": float(best_valid_loss),
            "patience": patience,
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "categorical_cardinalities": categorical_cardinalities,
            "embedding_dims": embedding_dims,
        }

        meta_path = model_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        params_path = metrics_path.parent / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        if preds_path is not None:
            preds_df = pd.DataFrame({
                "y_true": all_true,
                "prediction": all_preds,
                "y_pred": all_preds,
            })

            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
            print(f"[LSTM-Emb] predictions saved to {preds_path}")

        print(f"[LSTM-Emb] metrics: {metrics}")
        return metrics, params