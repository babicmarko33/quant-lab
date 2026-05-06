"""LSTM sequence classifier for alpha_ml (Phase 5.2).

Architecture:
    Input: (batch, seq_len, n_features)
    LSTM: n_layers=1, hidden_size configurable
    Output head: Linear(hidden_size, 2) → softmax probabilities

Training:
    - Adam optimizer, CrossEntropyLoss
    - Sliding window sequences of length seq_len
    - The label for a window ending at index t is y[t]

Feature importance:
    - Approximated by gradient × input sensitivity (input gradient norm per feature)
    - Averaged over all validation windows post-training
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alpha_ml.models.base import ModelTrainer


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int, n_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        _, (h_n, _) = self.lstm(x)   # h_n: (n_layers, B, hidden)
        return self.head(h_n[-1])    # (B, n_classes)


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences of length seq_len.

    Window [i : i+seq_len] → label y[i+seq_len-1].
    Returns arrays of shape (n_windows, seq_len, n_features) and (n_windows,).
    """
    n = len(X)
    xs, ys = [], []
    for i in range(n - seq_len + 1):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


class LSTMClassifier(ModelTrainer):
    """Binary/multi-class LSTM classifier implementing the ModelTrainer ABC.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden state dimension (default 64).
    n_layers : int
        Number of LSTM layers (default 1).
    seq_len : int
        Length of input sequence window (default 20).
    n_epochs : int
        Training epochs (default 20).
    batch_size : int
        Mini-batch size (default 32).
    lr : float
        Adam learning rate (default 1e-3).
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 1,
        seq_len: int = 20,
        n_epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self._net: _LSTMNet | None = None
        self._feature_names: list[str] | None = None
        self._n_features: int | None = None

    def _check_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) < self.seq_len:
            raise ValueError(
                f"seq_len={self.seq_len} must be < number of samples ({len(X)})"
            )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self._feature_names = list(X.columns)
        self._n_features = X.shape[1]
        n_classes = int(y.nunique())

        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int64)

        X_seq, y_seq = _make_sequences(X_arr, y_arr, self.seq_len)

        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._net = _LSTMNet(self._n_features, self.hidden_size, self.n_layers, n_classes)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._net.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self._net.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        proba = self.predict_proba(X)
        return proba.argmax(axis=1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self._net is not None

        X_arr = X.values.astype(np.float32)
        n = len(X_arr)
        seq = self.seq_len

        # Build sequences; for rows < seq_len, pad with zeros at the front
        X_seq = np.zeros((n, seq, X_arr.shape[1]), dtype=np.float32)
        for i in range(n):
            start = max(0, i - seq + 1)
            length = i - start + 1
            X_seq[i, seq - length :] = X_arr[start : i + 1]

        x_t = torch.from_numpy(X_seq)
        with torch.no_grad():
            logits = self._net(x_t)
            return torch.softmax(logits, dim=-1).numpy()

    def feature_importance(self) -> pd.Series:
        """Approximate feature importance as mean |gradient| per feature."""
        self._check_fitted()
        assert self._net is not None
        assert self._feature_names is not None

        # Use gradient × input approach on a random batch of the training sequences
        # We compute grad of the predicted class score w.r.t. each input feature
        # and take the mean absolute gradient across time steps and samples
        seq = self.seq_len
        n_feat = len(self._feature_names)
        dummy = torch.randn(max(32, seq), seq, n_feat, requires_grad=True)

        self._net.train()  # enable grad
        logits = self._net(dummy)
        score = logits[:, 0].sum()
        score.backward()
        importance = dummy.grad.abs().mean(dim=(0, 1)).detach().numpy()
        self._net.eval()

        return pd.Series(importance.tolist(), index=self._feature_names)
