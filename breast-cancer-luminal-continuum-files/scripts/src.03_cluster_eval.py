#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


#config
TOP_VAR_GENES = 2000
N_PCS         = 10
K             = 5
RANDOM_STATE  = 42

FIG_DIR = "figures"
OUT_DIR = "outputs"

SUBTYPE_COLORS = {
    "BRCA_Basal":  "#6A0DAD",
    "BRCA_Her2":   "#FF7F0E",
    "BRCA_LumA":   "#2CA02C",
    "BRCA_LumB":   "#D62728",
    "BRCA_Normal": "#BCBD22",
}



# I/O helpers

def load_data(x_path, y_path):
    X = pd.read_csv(x_path, sep="\t", index_col=0)
    y_df = pd.read_csv(y_path, sep="\t", index_col=0)
    y = y_df["Subtype"] if "Subtype" in y_df.columns else y_df.iloc[:, 0]
    y.name = "Subtype"

    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    mask = y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


def select_top_variable_genes(X, n_top):
    top = X.var(axis=0).sort_values(ascending=False).head(n_top).index
    return X[top].copy()


def preprocess(X):
    X_log = np.log2(X + 1.0)
    scaler = StandardScaler()
    return scaler.fit_transform(X_log.values)



# Clustering

def build_algorithms(k, random_state):
    """Return list of (name, estimator) pairs for all 8 algorithms."""
    algorithms = [
        (
            f"KMeans(k={k})",
            KMeans(n_clusters=k, n_init=50, random_state=random_state),
        ),
    ]

    for cov in ["full", "tied", "diag", "spherical"]:
        algorithms.append((
            f"GMM(k={k},cov={cov})",
            GaussianMixture(
                n_components=k,
                covariance_type=cov,
                n_init=10,
                random_state=random_state,
            ),
        ))

    for linkage in ["ward", "complete", "average"]:
        metric = "euclidean"
        algorithms.append((
            f"Agglo(k={k},link={linkage},metric={metric})",
            AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric),
        ))

    return algorithms


def fit_predict(name, model, Z):
    """Fit model and return cluster labels (handles GMM separately)."""
    if "GMM" in name:
        model.fit(Z)
        return model.predict(Z)
    else:
        return model.fit_predict(Z)


# Evaluation

def evaluate(y_true, labels, Z):
    y_codes = pd.Categorical(y_true).codes
    ari = adjusted_rand_score(y_codes, labels)
    nmi = normalized_mutual_info_score(y_codes, labels)
    sil = silhouette_score(Z, labels)
    return ari, nmi, sil



#figures
def plot_pca_true(Z, y, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    for subtype in sorted(y.unique()):
        mask = (y == subtype).values
        color = SUBTYPE_COLORS.get(subtype, None)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7,
                   label=subtype, color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("TCGA-BRCA PCA (PC1 vs PC2) colored by TRUE subtype")
    ax.legend(markerscale=2, fontsize=9, title="Subtype")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pca_pred(Z, labels, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7, label=f"Cluster {c}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("TCGA-BRCA PCA (PC1 vs PC2) colored by predicted cluster")
    ax.legend(markerscale=2, fontsize=9, title="Cluster")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_heatmap(ct, outpath):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([f"Cluster {c}" for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Subtype")
    ax.set_title("Subtype vs Cluster (counts) — Best Method")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


#main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", default="X_tpm_samples_by_genes.tsv")
    parser.add_argument("--y", default="y_subtype.tsv")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load & preprocess 
    print("Loading data...")
    X, y = load_data(args.x, args.y)
    print(f"  X: {X.shape},  y: {y.shape}")

    X = select_top_variable_genes(X, TOP_VAR_GENES)
    X_scaled = preprocess(X)

    pca = PCA(n_components=N_PCS, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_scaled)
    print(f"  PCA variance explained (PC1-3): "
          f"{pca.explained_variance_ratio_[:3]*100}")

    # Run all 8 algorithms 
    algorithms = build_algorithms(K, RANDOM_STATE)
    results = []

    print(f"\nEvaluating {len(algorithms)} algorithms...")
    for name, model in algorithms:
        labels = fit_predict(name, model, Z)
        ari, nmi, sil = evaluate(y, labels, Z)
        results.append({"method": name, "ARI": ari, "NMI": nmi, "Silhouette": sil})
        print(f"  {name:<45}  ARI={ari:.4f}  NMI={nmi:.4f}  Sil={sil:.4f}")

    # Save all metrics 
    df_results = pd.DataFrame(results).sort_values("NMI", ascending=False)
    all_path = os.path.join(OUT_DIR, "cluster_metrics_all.tsv")
    df_results.to_csv(all_path, sep="\t", index=False)
    print(f"\nSaved: {all_path}")

    # Best method by NMI 
    best_row = df_results.iloc[0]
    best_name = best_row["method"]
    print(f"\nBest method (by NMI): {best_name}")

    best_path = os.path.join(OUT_DIR, "cluster_metrics_best.txt")
    with open(best_path, "w") as f:
        f.write(f"BEST_METHOD\t{best_name}\n")
        f.write(f"k\t{K}\n")
        f.write(f"TOP_VAR_GENES\t{TOP_VAR_GENES}\n")
        f.write(f"N_PCS\t{N_PCS}\n")
        f.write(f"ARI\t{best_row['ARI']:.6f}\n")
        f.write(f"NMI\t{best_row['NMI']:.6f}\n")
        f.write(f"Silhouette\t{best_row['Silhouette']:.6f}\n")
    print(f"Saved: {best_path}")

    # Refit best model for figures 
    best_model = dict(algorithms)[best_name]
    best_labels = fit_predict(best_name, best_model, Z)

    #Contingency table
    ct = pd.crosstab(y, pd.Series(best_labels, index=y.index, name="Cluster"))
    ct_path = os.path.join(OUT_DIR, "subtype_vs_cluster_counts.tsv")
    ct.to_csv(ct_path, sep="\t")
    print(f"Saved: {ct_path}")

    # Figures 
    plot_pca_true(Z, y, os.path.join(FIG_DIR, "pca_true_subtype.png"))
    print(f"Saved: {FIG_DIR}/pca_true_subtype.png")

    plot_pca_pred(Z, best_labels, os.path.join(FIG_DIR, "pca_pred_cluster.png"))
    print(f"Saved: {FIG_DIR}/pca_pred_cluster.png")

    plot_heatmap(ct, os.path.join(FIG_DIR, "subtype_vs_cluster_heatmap.png"))
    print(f"Saved: {FIG_DIR}/subtype_vs_cluster_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
