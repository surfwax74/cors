"""DeepGraph_GNN: GraphConv autoencoder over a correlation graph.

A graph is built from the training correlation matrix (edges where
|corr| >= threshold). A small 2-layer GraphConv autoencoder is trained to
reconstruct per-node time windows; reconstruction error is the anomaly score.

PyTorch / torch_geometric are imported lazily so the rest of the project keeps
working in environments where they are not installed. If they are missing,
constructing/fitting this model raises a clear ImportError and the engine
records the model as "skipped".
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

try:  # optional heavy dependencies
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import networkx as nx
    from torch_geometric.nn import GraphConv

    _TG_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    _TG_AVAILABLE = False
    _IMPORT_ERROR = exc


if _TG_AVAILABLE:

    class _GDN(nn.Module):
        """GraphConv reconstruction net: [n_nodes, window] -> [n_nodes, window]."""

        def __init__(self, edge_index, window, hidden_dim=8):
            super().__init__()
            self.edge_index = edge_index
            self.conv1 = GraphConv(window, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, window)

        def forward(self, x):
            h = F.relu(self.conv1(x, self.edge_index))
            return self.conv2(h, self.edge_index)

    def _build_graph_from_corr(X_train, threshold):
        corr = np.corrcoef(X_train, rowvar=False)
        n = corr.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    G.add_edge(i, j)
        if G.number_of_edges() == 0:  # avoid an empty edge set
            for i in range(n - 1):
                G.add_edge(i, i + 1)
        return torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

    def _extract_window(X_t, i, window):
        """Return a (window, n_features) block, left-padding the start."""
        start, end = i - window, i
        if start < 0:
            block = X_t[0:end]
            pad = block[0].repeat(-start, 1)
            block = torch.cat([pad, block], dim=0)
        else:
            block = X_t[start:end]
        return block


class DeepGraphModel(BaseModel):
    name = "DeepGraph_GNN"

    def __init__(
        self,
        window: int = 12,
        epochs: int = 2,
        corr_threshold: float = 0.3,
        hidden_dim: int = 8,
        train_stride: int = 10,
        num_train_windows: int = 10,
        use_cuda: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__(
            window=window,
            epochs=epochs,
            corr_threshold=corr_threshold,
            hidden_dim=hidden_dim,
            train_stride=train_stride,
            num_train_windows=num_train_windows,
            use_cuda=use_cuda,
            lr=lr,
        )
        self.window = window
        self.epochs = epochs
        self.corr_threshold = corr_threshold
        self.hidden_dim = hidden_dim
        self.train_stride = train_stride
        self.num_train_windows = num_train_windows
        self.use_cuda = use_cuda
        self.lr = lr

        self.model = None
        self.device = None

    @staticmethod
    def is_available() -> bool:
        return _TG_AVAILABLE

    def _require_torch(self):
        if not _TG_AVAILABLE:
            raise ImportError(
                "DeepGraph_GNN requires torch and torch_geometric. "
                f"Original import error: {_IMPORT_ERROR!r}"
            )

    def fit(self, X_train: np.ndarray) -> "DeepGraphModel":
        self._require_torch()
        self.device = torch.device(
            "cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu"
        )

        edge_index = _build_graph_from_corr(X_train, self.corr_threshold).to(self.device)
        self.model = _GDN(edge_index, self.window, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)

        # Pick an evenly spaced subset of strided training windows.
        all_indices = list(range(self.window, len(X_train)))
        stride_indices = all_indices[:: max(1, self.train_stride)]
        if len(stride_indices) > self.num_train_windows:
            idx = np.linspace(0, len(stride_indices) - 1, self.num_train_windows).astype(int)
            train_indices = [stride_indices[j] for j in idx]
        else:
            train_indices = stride_indices

        train_windows = [
            _extract_window(X_train_t, i, self.window).T for i in train_indices
        ]

        for _ in range(self.epochs):
            for x_win in train_windows:
                optimizer.zero_grad()
                recon = self.model(x_win)
                loss = F.mse_loss(recon, x_win)
                loss.backward()
                optimizer.step()

        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._require_torch()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        scores = []
        for i in range(len(X)):
            if i < self.window:
                scores.append(0.0)
                continue
            x_win = _extract_window(X_t, i, self.window).T
            with torch.no_grad():
                recon = self.model(x_win)
                scores.append(F.mse_loss(recon, x_win).item())
        return np.array(scores)
