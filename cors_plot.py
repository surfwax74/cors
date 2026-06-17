import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_raw_signals(df, feature_cols, n=5):
    """Plot n random raw telemetry channels."""
    cols = np.random.choice(feature_cols, size=n, replace=False)
    plt.figure(figsize=(14, 6))
    for c in cols:
        plt.plot(df["timestamp"], df[c], label=c, alpha=0.8)
    plt.title(f"Raw Signals (n={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drift_cluster(df, cluster_cols):
    """Plot a known drift cluster (list of column names)."""
    plt.figure(figsize=(14, 6))
    for c in cluster_cols:
        plt.plot(df["timestamp"], df[c], label=c, alpha=0.8)
    plt.title("Drift Cluster")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomaly_window(df, feature, anomaly_mask):
    """Plot a single feature with anomaly window highlighted."""
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df[feature], label=feature)
    plt.fill_between(
        df["timestamp"],
        df[feature].min(),
        df[feature].max(),
        where=anomaly_mask,
        color="red",
        alpha=0.2,
        label="Anomaly Window",
    )
    plt.title(f"Feature with Anomaly Window: {feature}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fft_centroid_distribution(df, feature_cols):
    """Plot histogram of FFT centroid features."""
    fft_cols = [c for c in feature_cols if "fft" in c.lower() or "centroid" in c.lower()]
    if not fft_cols:
        print("No FFT centroid columns found.")
        return

    plt.figure(figsize=(10, 5))
    for c in fft_cols:
        sns.kdeplot(df[c], label=c, fill=True, alpha=0.3)
    plt.title("FFT Centroid Distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, feature_cols, n=50):
    """Plot correlation heatmap of a random subset of features."""
    cols = np.random.choice(feature_cols, size=min(n, len(feature_cols)), replace=False)
    corr = df[cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap (n={len(cols)})")
    plt.tight_layout()
    plt.show()


def plot_pca_projection(df, feature_cols, n_components=2):
    """Plot PCA projection of the feature space."""
    from sklearn.decomposition import PCA

    X = df[feature_cols].values
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.5)
    plt.title("PCA Projection of Telemetry Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
