#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap


# Config

TOP_VAR_GENES = 2000
RANDOM_STATE = 42

FIG_DIR = "figures"
OUT_DIR = "outputs"

SUBTYPE_COLORS = {
    "BRCA_Basal":  "#6A0DAD",
    "BRCA_Her2":   "#FF7F0E",
    "BRCA_LumA":   "#2CA02C",
    "BRCA_LumB":   "#D62728",
    "BRCA_Normal": "#BCBD22",
}


# Load data

def load_data(x_path="X_tpm_samples_by_genes.tsv", y_path="y_subtype.tsv"):
    X = pd.read_csv(x_path, sep="\t", index_col=0)
    y_df = pd.read_csv(y_path, sep="\t", index_col=0)

    if "Subtype" in y_df.columns:
        y = y_df["Subtype"]
    else:
        y = y_df.iloc[:, 0]

    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    mask = y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


# Preprocess

def preprocess_for_umap(X, top_var_genes=2000):
    # log2(TPM + 1)
    X_log = np.log2(X.astype(float) + 1.0)

    # keep top variable genes
    top_genes = (
        X_log.var(axis=0)
        .sort_values(ascending=False)
        .head(top_var_genes)
        .index
    )
    X_sel = X_log[top_genes].copy()

    # z-score genes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel.values)

    return X_scaled, X_sel.columns


# Plot

def plot_umap_true(embedding, y, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))

    for subtype in sorted(y.unique()):
        mask = (y == subtype).values
        color = SUBTYPE_COLORS.get(subtype, None)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.75,
            label=subtype,
            color=color
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("TCGA-BRCA UMAP colored by TRUE subtype")
    ax.legend(title="Subtype", fontsize=9, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# Main

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nSubtype counts:")
    print(y.value_counts())

    print("\nPreprocessing...")
    X_scaled, kept_genes = preprocess_for_umap(X, top_var_genes=TOP_VAR_GENES)
    print(f"After filtering: {X_scaled.shape}")
    print(f"Top variable genes kept: {len(kept_genes)}")

    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_STATE
    )
    embedding = reducer.fit_transform(X_scaled)

    # Save coordinates
    umap_df = pd.DataFrame(
        embedding,
        index=X.index,
        columns=["UMAP1", "UMAP2"]
    )
    umap_df["Subtype"] = y.values
    umap_df.to_csv(f"{OUT_DIR}/umap_true_subtype.tsv", sep="\t")
    print(f"Saved: {OUT_DIR}/umap_true_subtype.tsv")

    # Save figure
    plot_umap_true(embedding, y, f"{FIG_DIR}/umap_true_subtype.png")
    print(f"Saved: {FIG_DIR}/umap_true_subtype.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
