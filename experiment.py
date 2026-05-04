import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd

from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, IncrementalPCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GraphConv
from tqdm import tqdm

# ============================================================
# CONFIG (EDITABLE)
# ============================================================

TRAIN_DAYS = 30
TEST_DAYS = 5
BIN_MINUTES = 10

ANOMALY_WINDOW_MINUTES = 60
ANOMALY_WINDOW_BINS = ANOMALY_WINDOW_MINUTES // BIN_MINUTES

CACHE_DIR = "cache_results"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---- MahalanobisCorr parameters ----
MAHALANOBIS_WINDOW = 12
MAHALANOBIS_K_EIGEN = 50        # <--- number of top eigenvalues to keep
MAHALANOBIS_PCA_DIM = 20        # PCA dimension after eigenvalues
MAHALANOBIS_STRIDE = 10         # stride over training windows




# ---- DeepGraph / GNN parameters ----
DEEPGRAPH_WINDOW = 12              # time window length (bins)
DEEPGRAPH_EPOCHS = 2               # very small, graph is static
DEEPGRAPH_TRAIN_STRIDE = 10        # stride over train windows
DEEPGRAPH_NUM_TRAIN_WINDOWS = 10   # max number of train windows
DEEPGRAPH_CORR_THRESHOLD = 0.3
DEEPGRAPH_HIDDEN_DIM = 8           # small hidden size
DEEPGRAPH_HEADS = 1                # kept for config symmetry (GraphConv ignores)
USE_CUDA_FOR_DEEPGRAPH = False     # flip to True under WSL+GPU

# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(f"[INFO] {msg}", flush=True)

# ============================================================
# CACHING
# ============================================================

def df_hash(df):
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(data_bytes).hexdigest()

def cached(method_name, df, compute_fn):
    h = df_hash(df)
    fname = os.path.join(CACHE_DIR, f"{method_name}_{h}.pkl")

    if os.path.exists(fname):
        log(f"  → Loading cached result for {method_name}")
        with open(fname, "rb") as f:
            return pickle.load(f)

    log(f"  → Computing {method_name} (no cache found)")
    result = compute_fn()

    with open(fname, "wb") as f:
        pickle.dump(result, f)

    return result

# ============================================================
# LOAD DATA
# ============================================================

log("Loading signals_10min_features.csv...")
df = pd.read_csv("signals_10min_features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bins_per_day = int(24 * 60 / BIN_MINUTES)
train_end = TRAIN_DAYS * bins_per_day

df_train = df.iloc[:train_end].copy()
df_test  = df.iloc[train_end:].copy()

feature_cols = [c for c in df.columns if c != "timestamp"]

X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values

log(f"Data loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================================
# GROUND TRUTH LABELS
# ============================================================

y_test = np.zeros(len(df_test), dtype=int)
for d in range(TEST_DAYS):
    start = d * bins_per_day
    end   = start + ANOMALY_WINDOW_BINS
    y_test[start:end] = 1

log("Ground truth anomaly labels constructed.")

# ============================================================
# CLASSICAL CORS METHODS
# ============================================================

def rolling_corr_shift(X, window=12):
    base_corr = np.corrcoef(X_train, rowvar=False)
    scores = []
    for i in range(len(X)):
        if i < window:
            scores.append(0)
            continue
        block = X[i-window:i]
        corr = np.corrcoef(block, rowvar=False)
        diff = corr - base_corr
        scores.append(np.linalg.norm(diff, ord="fro"))
    return np.array(scores)


def mahalanobis_corr_score_fast(
    X,
    window=MAHALANOBIS_WINDOW,
    k=MAHALANOBIS_K_EIGEN,
    pca_dim=MAHALANOBIS_PCA_DIM,
    stride=MAHALANOBIS_STRIDE,
):
    """
    MahalanobisCorr with:
      - top-k eigenvalues of corr matrix
      - PCA reduction to pca_dim
      - stride over training windows
    """

    # ---- Base correlation eigenvalues ----
    base_corr = np.corrcoef(X_train, rowvar=False)
    base_vals = np.linalg.eigvalsh(base_corr)[-k:]  # shape (k,)

    # ---- Build training eigenvalue vectors (with stride) ----
    eig_vecs = []
    train_indices = range(window, len(X_train), max(1, stride))

    for i in tqdm(train_indices, desc="MahalanobisCorr Train Windows", leave=False):
        block = X_train[i-window:i]
        corr = np.corrcoef(block, rowvar=False)
        vals = np.linalg.eigvalsh(corr)[-k:]
        eig_vecs.append(vals)

    eig_vecs = np.array(eig_vecs)  # shape (n_train_windows, k)

    # ---- PCA in eigenvalue space ----
    pca_dim_eff = min(pca_dim, k)
    pca = PCA(n_components=pca_dim_eff)
    eig_vecs_pca = pca.fit_transform(eig_vecs)

    base_pca = pca.transform(base_vals.reshape(1, -1))[0]

    # ---- Fit covariance in PCA space ----
    cov = EmpiricalCovariance().fit(eig_vecs_pca)

    # ---- Score test windows ----
    scores = []
    for i in tqdm(range(len(X)), desc="MahalanobisCorr Scoring", leave=False):
        if i < window:
            scores.append(0.0)
            continue

        block = X[i-window:i]
        corr = np.corrcoef(block, rowvar=False)
        vals = np.linalg.eigvalsh(corr)[-k:]
        vals_pca = pca.transform(vals.reshape(1, -1))[0]
        diff = vals_pca - base_pca
        scores.append(cov.mahalanobis(diff.reshape(1, -1))[0])

    return np.array(scores)


def pca_eigen_shift_all(X, window=12, k=20):
    k_eff = min(k, window)

    pca_full = PCA(n_components=k_eff, svd_solver="full")
    pca_full.fit(X_train)
    base_full = pca_full.explained_variance_

    pca_rand = PCA(n_components=k_eff, svd_solver="randomized")
    pca_rand.fit(X_train)
    base_rand = pca_rand.explained_variance_

    pca_inc = IncrementalPCA(n_components=k_eff)
    pca_inc.fit(X_train)
    base_inc = pca_inc.explained_variance_

    scores_full = []
    scores_rand = []
    scores_inc = []

    for i in range(len(X)):
        if i < window:
            scores_full.append(0)
            scores_rand.append(0)
            scores_inc.append(0)
            continue

        block = X[i-window:i]

        p2 = PCA(n_components=k_eff, svd_solver="full")
        p2.fit(block)
        eigs = p2.explained_variance_
        scores_full.append(np.linalg.norm(eigs - base_full))

        p3 = PCA(n_components=k_eff, svd_solver="randomized")
        p3.fit(block)
        eigs = p3.explained_variance_
        scores_rand.append(np.linalg.norm(eigs - base_rand))

        p4 = IncrementalPCA(n_components=k_eff)
        p4.fit(block)
        eigs = p4.explained_variance_
        scores_inc.append(np.linalg.norm(eigs - base_inc))

    return (
        np.array(scores_full),
        np.array(scores_rand),
        np.array(scores_inc)
    )

def pca_reconstruction_error(X, window=12, k=20):
    k_eff = min(k, window)
    pca = PCA(n_components=k_eff)
    pca.fit(X_train)

    scores = []
    for i in range(len(X)):
        if i < window:
            scores.append(0)
            continue

        block = X[i-window:i]
        Z = pca.transform(block)
        X_hat = pca.inverse_transform(Z)
        err = np.mean((block - X_hat)**2)
        scores.append(err)

    return np.array(scores)

def graph_corr_shift(X, window=12):
    base_corr = np.corrcoef(X_train, rowvar=False)
    base_adj = np.abs(base_corr)

    scores = []
    for i in range(len(X)):
        if i < window:
            scores.append(0)
            continue
        block = X[i-window:i]
        corr = np.corrcoef(block, rowvar=False)
        adj = np.abs(corr)
        scores.append(np.sum(np.abs(adj - base_adj)))
    return np.array(scores)

# ============================================================
# DEEPGRAPH / GNN (GraphConv, fast)
# ============================================================

def build_graph_from_corr(X_train, threshold=0.3):
    corr = np.corrcoef(X_train, rowvar=False)
    n = corr.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(i, j)

    if G.number_of_edges() == 0:
        for i in range(n - 1):
            G.add_edge(i, i + 1)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    return edge_index

class GDN(nn.Module):
    """
    GraphConv-based reconstruction model:
    x: [num_nodes, window] -> [num_nodes, window]
    """
    def __init__(self, num_nodes, edge_index, window, hidden_dim=8):
        super().__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.window = window

        self.conv1 = GraphConv(window, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, window)

    def forward(self, x):
        h = F.relu(self.conv1(x, self.edge_index))
        out = self.conv2(h, self.edge_index)
        return out

def extract_window(X_t, i, window):
    """
    Always returns shape (window, num_features)
    Pads the beginning if needed.
    """
    start = i - window
    end = i

    if start < 0:
        block = X_t[0:end]
        pad = block[0].repeat(-start, 1)
        block = torch.cat([pad, block], dim=0)
    else:
        block = X_t[start:end]

    return block  # (window, num_features)

def run_deepgraph(
    X_train,
    X_test,
    window=12,
    epochs=2,
    corr_threshold=0.3,
    use_cuda=False,
    hidden_dim=8,
    train_stride=10,
    num_train_windows=10,
):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    num_nodes = X_train.shape[1]

    edge_index = build_graph_from_corr(X_train, threshold=corr_threshold).to(device)

    model = GDN(
        num_nodes=num_nodes,
        edge_index=edge_index,
        window=window,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    # ---- Select and precompute training windows ----
    all_indices = list(range(window, len(X_train)))
    stride_indices = all_indices[::max(1, train_stride)]
    if len(stride_indices) > num_train_windows:
        # pick evenly spaced subset
        idx = np.linspace(0, len(stride_indices) - 1, num_train_windows).astype(int)
        train_indices = [stride_indices[j] for j in idx]
    else:
        train_indices = stride_indices

    train_windows = []
    for i in tqdm(train_indices, desc="DeepGraph Precompute Train Windows", leave=False):
        block = extract_window(X_train_t, i, window)  # (window, num_nodes)
        x_win = block.T                               # (num_nodes, window)
        train_windows.append(x_win)

    # ---- Train ----
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for x_win in tqdm(train_windows, desc=f"DeepGraph Train Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            recon = model(x_win)
            loss = F.mse_loss(recon, x_win)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        log(f"[DeepGraph] Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.6f}")

    # ---- Score ----
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    scores = []

    for i in tqdm(range(len(X_test)), desc="DeepGraph Scoring", leave=False):
        if i < window:
            scores.append(0.0)
            continue

        block = extract_window(X_test_t, i, window)
        x_win = block.T

        with torch.no_grad():
            recon = model(x_win)
            err = F.mse_loss(recon, x_win).item()

        scores.append(err)

    return np.array(scores)

# ============================================================
# PLACEHOLDER CORALS
# ============================================================

def run_corals(X_train, X_test):
    return np.zeros(len(X_test))

# ============================================================
# EVALUATION
# ============================================================

def evaluate(scores, y):
    if np.all(scores == 0):
        return 0.0, 0.0
    corr = np.corrcoef(scores, y)[0, 1]
    K = int(y.sum())
    if K == 0:
        return corr, 0.0
    top_k_idx = np.argsort(scores)[-K:]
    hit_rate = y[top_k_idx].sum() / K
    return corr, hit_rate

# ============================================================
# RUN ALL METHODS
# ============================================================

log("Starting correlation-based anomaly detection tests...")

methods = {}

# RollingCorr
start = time.time()
methods["RollingCorr"] = cached(
    "RollingCorr",
    df_test,
    lambda: rolling_corr_shift(X_test)
)
log(f"RollingCorr complete in {time.time() - start:.2f}s")

# MahalanobisCorr (fast eigen-based)
start = time.time()
methods["MahalanobisCorr"] = cached(
    "MahalanobisCorr",
    df_test,
    lambda: mahalanobis_corr_score_fast(
        X_test,
        window=MAHALANOBIS_WINDOW,
        k=MAHALANOBIS_K_EIGEN,
        pca_dim=MAHALANOBIS_PCA_DIM,
        stride=MAHALANOBIS_STRIDE,
    ),
)

log(f"MahalanobisCorr complete in {time.time() - start:.2f}s")

# PCA eigen shifts
start = time.time()
pca_full_scores, pca_rand_scores, pca_inc_scores = cached(
    "PCAEigenShiftAll",
    df_test,
    lambda: pca_eigen_shift_all(X_test)
)
methods["PCA_Full"] = pca_full_scores
methods["PCA_Randomized"] = pca_rand_scores
methods["PCA_Incremental"] = pca_inc_scores
log(f"PCA eigen shifts complete in {time.time() - start:.2f}s")

# PCA reconstruction
start = time.time()
methods["PCA_Reconstruction"] = cached(
    "PCA_Reconstruction",
    df_test,
    lambda: pca_reconstruction_error(X_test)
)
log(f"PCA reconstruction complete in {time.time() - start:.2f}s")

# GraphCorrShift
start = time.time()
methods["GraphCorrShift"] = cached(
    "GraphCorrShift",
    df_test,
    lambda: graph_corr_shift(X_test)
)
log(f"GraphCorrShift complete in {time.time() - start:.2f}s")

# DeepGraph / GNN (GraphConv, fast)
start = time.time()
methods["DeepGraph_GNN"] = cached(
    "DeepGraph_GNN",
    df_test,
    lambda: run_deepgraph(
        X_train,
        X_test,
        window=DEEPGRAPH_WINDOW,
        epochs=DEEPGRAPH_EPOCHS,
        corr_threshold=DEEPGRAPH_CORR_THRESHOLD,
        use_cuda=USE_CUDA_FOR_DEEPGRAPH,
        hidden_dim=DEEPGRAPH_HIDDEN_DIM,
        train_stride=DEEPGRAPH_TRAIN_STRIDE,
        num_train_windows=DEEPGRAPH_NUM_TRAIN_WINDOWS,
    )
)
log(f"DeepGraph_GNN complete in {time.time() - start:.2f}s")

# CorALS (placeholder)
start = time.time()
methods["CorALS"] = cached(
    "CorALS",
    df_test,
    lambda: run_corals(X_train, X_test)
)
log(f"CorALS complete in {time.time() - start:.2f}s")

# ============================================================
# RESULTS
# ============================================================

log("Evaluating all methods against ground truth...")

print("\n=== CORRELATION-BASED ANOMALY DETECTION RESULTS ===\n")
for name, scores in methods.items():
    corr, hit = evaluate(scores, y_test)
    print(f"{name:20s}  Corr(y,score)={corr:6.3f}   HitRate={hit:6.3f}")

log("All methods evaluated successfully.")
