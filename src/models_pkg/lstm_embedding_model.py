import torch
import torch.nn as nn


class LSTMEmbeddingRegressor(nn.Module):
    """
    LSTM, которая:
    - получает numeric features
    - получает categorical features через embeddings
    """

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: dict[str, int],
        embedding_dims: dict[str, int],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.categorical_feature_names = list(categorical_cardinalities.keys())

        self.embedding_layers = nn.ModuleDict({
            feature_name: nn.Embedding(
                num_embeddings=categorical_cardinalities[feature_name],
                embedding_dim=embedding_dims[feature_name],
            )
            for feature_name in self.categorical_feature_names
        })

        total_embedding_dim = sum(
            embedding_dims[feature_name] for feature_name in self.categorical_feature_names
        )

        lstm_input_size = num_numeric_features + total_embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: [batch, seq_len, num_numeric_features]
        x_cat: [batch, seq_len, num_categorical_features]
        """
        embedded_parts = []

        for idx, feature_name in enumerate(self.categorical_feature_names):
            feature_indices = x_cat[:, :, idx]
            embedded = self.embedding_layers[feature_name](feature_indices)
            embedded_parts.append(embedded)

        if embedded_parts:
            x_emb = torch.cat(embedded_parts, dim=-1)
            x = torch.cat([x_num, x_emb], dim=-1)
        else:
            x = x_num

        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)