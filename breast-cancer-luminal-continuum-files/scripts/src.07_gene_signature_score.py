#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Config

X_PATH = "X_tpm_samples_by_genes.tsv"
Y_PATH = "y_subtype.tsv"

DE_PATH = "outputs/ambiguous_vs_clear_DE.tsv"

OUT_DIR = "outputs"
FIG_DIR = "figures"

TOP_N_GENES = 25

KEEP_SUBTYPES = ["BRCA_LumA", "BRCA_LumB"]



# Helper: clean Ensembl IDs

def clean_ensembl_id(gene_id):
    return str(gene_id).split(".")[0]



# Convert Ensembl IDs to Gene Symbols

def convert_ensembl_to_symbol(ensembl_ids):
    try:
        import mygene
    except ImportError:
        raise ImportError(
            "The package 'mygene' is not installed. Run: pip install mygene"
        )

    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human"
    )

    mapping = {}

    for r in results:
        query = r.get("query")

        if "symbol" in r:
            mapping[query] = r["symbol"]
        else:
            mapping[query] = "NA"

    return mapping



# Load Data

def load_data():
    X = pd.read_csv(X_PATH, sep="\t", index_col=0)
    y_df = pd.read_csv(Y_PATH, sep="\t", index_col=0)

    if "Subtype" in y_df.columns:
        y = y_df["Subtype"]
    else:
        y = y_df.iloc[:, 0]

    # Match samples
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    # Keep LumA and LumB only
    mask = y.isin(KEEP_SUBTYPES)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # Log transform expression
    X_log = np.log2(X.astype(float) + 1.0)

    # Clean Ensembl IDs in expression matrix
    X_log.columns = [clean_ensembl_id(g) for g in X_log.columns]

    # If duplicate Ensembl IDs exist after removing version numbers, average them
    X_log = X_log.T.groupby(level=0).mean().T

    return X_log, y



# Select Signature Genes

def select_signature_genes():
    de = pd.read_csv(DE_PATH, sep="\t")

    # Clean Ensembl IDs in DE results
    de["Ensembl_ID"] = de["Gene"].apply(clean_ensembl_id)

    # Sort by FDR
    de = de.sort_values("FDR")

    # Genes higher in ambiguous samples
    up_df = (
        de[de["logFC_Ambiguous_vs_Clear"] > 0]
        .head(TOP_N_GENES)
        .copy()
    )

    # Genes lower in ambiguous samples
    down_df = (
        de[de["logFC_Ambiguous_vs_Clear"] < 0]
        .head(TOP_N_GENES)
        .copy()
    )

    up_genes = up_df["Ensembl_ID"].tolist()
    down_genes = down_df["Ensembl_ID"].tolist()

    signature_genes = up_genes + down_genes

    # Convert Ensembl IDs to gene symbols
    print("Converting Ensembl IDs to gene symbols...")
    mapping = convert_ensembl_to_symbol(signature_genes)

    gene_df = pd.DataFrame({
        "Ensembl_ID": signature_genes,
        "Gene_Symbol": [mapping.get(g, "NA") for g in signature_genes],
        "Direction": (
            ["Higher_in_Ambiguous"] * len(up_genes) +
            ["Lower_in_Ambiguous"] * len(down_genes)
        )
    })

    # Save cleaned Ensembl-only file
    gene_df[["Ensembl_ID", "Direction"]].to_csv(
        f"{OUT_DIR}/ambiguity_signature_genes_clean.tsv",
        sep="\t",
        index=False
    )

    # Save full file with gene symbols
    gene_df.to_csv(
        f"{OUT_DIR}/ambiguity_signature_genes_with_symbols.tsv",
        sep="\t",
        index=False
    )

    print(f"Selected {len(up_genes)} genes higher in ambiguous samples.")
    print(f"Selected {len(down_genes)} genes lower in ambiguous samples.")
    print(f"Saved: {OUT_DIR}/ambiguity_signature_genes_with_symbols.tsv")

    return up_genes, down_genes, signature_genes



# Compute Signature Score

def compute_signature_score(X_log, y, up_genes, down_genes):
    up_genes_found = [g for g in up_genes if g in X_log.columns]
    down_genes_found = [g for g in down_genes if g in X_log.columns]

    print(f"Up genes found in expression data: {len(up_genes_found)}")
    print(f"Down genes found in expression data: {len(down_genes_found)}")

    if len(up_genes_found) == 0:
        raise ValueError("No higher-in-ambiguous genes found in expression matrix.")

    # Basic score: genes higher in ambiguous samples
    ambiguity_score = X_log[up_genes_found].mean(axis=1)

    # Refined score: ambiguous-high genes minus ambiguous-low genes
    if len(down_genes_found) > 0:
        refined_score = (
            X_log[up_genes_found].mean(axis=1)
            - X_log[down_genes_found].mean(axis=1)
        )
    else:
        refined_score = ambiguity_score.copy()

    score_df = pd.DataFrame({
        "Sample": X_log.index,
        "Subtype": y.loc[X_log.index],
        "Ambiguity_Score": ambiguity_score,
        "Refined_Ambiguity_Score": refined_score
    })

    score_df.to_csv(
        f"{OUT_DIR}/ambiguity_signature_score.tsv",
        sep="\t",
        index=False
    )

    print(f"Saved: {OUT_DIR}/ambiguity_signature_score.tsv")

    return score_df



# Plot Scores by Subtype

def plot_scores(score_df):
    plt.figure(figsize=(6, 5))

    lumA_scores = score_df[
        score_df["Subtype"] == "BRCA_LumA"
    ]["Refined_Ambiguity_Score"]

    lumB_scores = score_df[
        score_df["Subtype"] == "BRCA_LumB"
    ]["Refined_Ambiguity_Score"]

    groups = [lumA_scores, lumB_scores]

    plt.boxplot(
        groups,
        labels=["LumA", "LumB"],
        showfliers=False
    )

    for i, values in enumerate(groups, start=1):
        x = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(x, values, alpha=0.5, s=12)

    plt.ylabel("Refined Ambiguity Signature Score")
    plt.title("Ambiguity Signature Score by Subtype")

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/ambiguity_signature_score_by_subtype.png",
        dpi=300
    )
    plt.close()

    print(f"Saved: {FIG_DIR}/ambiguity_signature_score_by_subtype.png")



# Main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading expression and subtype data...")
    X_log, y = load_data()

    print("Expression shape after cleaning genes:", X_log.shape)

    print("\nSelecting ambiguity signature genes...")
    up_genes, down_genes, signature_genes = select_signature_genes()

    print("\nComputing ambiguity signature score...")
    score_df = compute_signature_score(X_log, y, up_genes, down_genes)

    print("\nMean refined scores by subtype:")
    print(score_df.groupby("Subtype")["Refined_Ambiguity_Score"].mean())

    print("\nPlotting score distribution...")
    plot_scores(score_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
