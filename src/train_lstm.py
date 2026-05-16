from pathlib import Path
import json
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
    LSTMTrainer:
    - numeric features идут как float
    - categorical features идут через embeddings
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
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]
        return numeric_cols, categorical_cols

    def _build_vocab_maps(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
    ) -> dict[str, dict[str, int]]:
        """
        Для каждой категориальной колонки строим отображение:
        category_value -> integer index
        0 резервируем под unknown.
        """
        vocab_maps: dict[str, dict[str, int]] = {}

        for col in categorical_cols:
            unique_values = sorted(df[col].astype(str).dropna().unique().tolist())
            vocab_maps[col] = {value: idx + 1 for idx, value in enumerate(unique_values)}

        return vocab_maps

    def _embedding_dims(self, categorical_cols: list[str]) -> dict[str, int]:
        """
        Ручная настройка embedding dims.
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

    def _build_test_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        vocab_maps: dict[str, dict[str, int]],
    ) -> PriceSequenceEmbeddingDataset:
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        min_test_date = test_df[self.data_cfg.date_col].min()

        return PriceSequenceEmbeddingDataset(
            df=full_df,
            seq_len=self.train_cfg.seq_len,
            numeric_feature_cols=numeric_cols,
            categorical_feature_cols=categorical_cols,
            categorical_vocab_maps=vocab_maps,
            target_col=self.data_cfg.target_col,
            group_cols=list(self.data_cfg.series_id_cols),
            date_col=self.data_cfg.date_col,
            min_target_date=min_test_date,
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.train_cfg.lstm_batch_size, shuffle=shuffle)

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_feature_cols: list[str],
        model_path: Path,
        metrics_path: Path,
        preds_path: Path | None = None,
    ) -> tuple[dict, np.ndarray]:
        print(f"[LSTM-Emb] device: {self.device}")
        print("[LSTM-Emb] splitting numeric/categorical features...")

        numeric_cols, categorical_cols = self._split_features(train_df, seq_feature_cols)

        print(
            f"[LSTM-Emb] numeric_features={len(numeric_cols)} | "
            f"categorical_features={len(categorical_cols)}"
        )

        # Строим vocab по train+test, потому что это словарь категорий, а не target-информация
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        vocab_maps = self._build_vocab_maps(full_df, categorical_cols)

        train_dataset = self._build_train_dataset(train_df, numeric_cols, categorical_cols, vocab_maps)
        test_dataset = self._build_test_dataset(train_df, test_df, numeric_cols, categorical_cols, vocab_maps)

        print(
            f"[LSTM-Emb] train sequences={len(train_dataset)} | "
            f"test sequences={len(test_dataset)} | "
            f"seq_len={self.train_cfg.seq_len}"
        )

        if len(train_dataset) == 0:
            raise ValueError("[LSTM-Emb] Training dataset is empty.")

        if len(test_dataset) == 0:
            raise ValueError("[LSTM-Emb] Test dataset is empty.")

        train_loader = self._make_loader(train_dataset, shuffle=True)
        test_loader = self._make_loader(test_dataset, shuffle=False)

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

        print(
            f"[LSTM-Emb] model initialized | "
            f"num_numeric={len(numeric_cols)} | num_categorical={len(categorical_cols)}"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg.lstm_lr)
        criterion = torch.nn.MSELoss()

        print(
            f"[LSTM-Emb] training started | epochs={self.train_cfg.lstm_epochs} | "
            f"batch_size={self.train_cfg.lstm_batch_size} | lr={self.train_cfg.lstm_lr}"
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем вспомогательные метаданные модели
        meta_path = model_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "categorical_cardinalities": categorical_cardinalities,
                    "embedding_dims": embedding_dims,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        for epoch in range(self.train_cfg.lstm_epochs):
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
                optimizer.step()
                epoch_losses.append(loss.item())

                if batch_idx % 100 == 0:
                    print(
                        f"[LSTM-Emb] epoch {epoch + 1}/{self.train_cfg.lstm_epochs} | "
                        f"batch {batch_idx}/{len(train_loader)} | batch_loss={loss.item():.6f}"
                    )

            mean_epoch_loss = float(np.mean(epoch_losses))
            print(
                f"[LSTM-Emb] epoch {epoch + 1}/{self.train_cfg.lstm_epochs} completed | "
                f"mean_loss={mean_epoch_loss:.6f}"
            )

            # Чекпоинт после каждой эпохи
            torch.save(model.state_dict(), model_path)
            print(f"[LSTM-Emb] checkpoint saved after epoch {epoch + 1} -> {model_path}")

        print("[LSTM-Emb] evaluating on test...")
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x_num, x_cat, y_batch in test_loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                preds = model(x_num, x_cat).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_true.extend(y_batch.numpy().tolist())

        metrics = evaluate_regression(np.array(all_true), np.array(all_preds))
        save_metrics(metrics, metrics_path)

        if preds_path is not None:
            preds_path.parent.mkdir(parents=True, exist_ok=True)

            preds_df = pd.DataFrame({
                "y_true": all_true,
                "prediction": all_preds,
            })

            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
            print(f"[LSTM-Emb] predictions saved to {preds_path}")

        print(f"[LSTM-Emb] metrics: {metrics}")
        return metrics, np.array(all_preds)