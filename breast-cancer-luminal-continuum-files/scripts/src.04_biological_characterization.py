#!/usr/bin/env python3

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


#LOAD & PREPROCESS  (mirrors src_03)


def load_inputs(x_path, y_path):
    X = pd.read_csv(x_path, sep="\t", index_col=0)
    y_df = pd.read_csv(y_path, sep="\t", index_col=0)
    y = y_df.iloc[:, 0] if isinstance(y_df, pd.DataFrame) else y_df

    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def preprocess_pca(X, top_var_genes=2000, n_pcs=10, random_state=0):
    X_log = np.log2(X.astype(float) + 1)
    top_genes = X_log.var(axis=0).sort_values(ascending=False).index[:top_var_genes]
    X_sel = X_log[top_genes]
    X_scaled = StandardScaler().fit_transform(X_sel.values)
    pca = PCA(n_components=n_pcs, random_state=random_state)
    Z = pca.fit_transform(X_scaled)
    # Return log-transformed full matrix for DE (all genes, not just top var)
    return Z, X_log


def get_cluster_labels(Z, k=5, random_state=0):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="tied",
        n_init=5,
        random_state=random_state
    )
    labels = gmm.fit_predict(Z)
    print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels



# ENSEMBL ID -> GENE SYMBOL CONVERSION


def build_id_map(ensembl_ids_with_versions):
    try:
        import mygene
    except ImportError:
        print("[WARNING] mygene not installed. Run: pip install mygene")
        print("  Heatmap will use Ensembl IDs. Install mygene and rerun for gene symbols.")
        return {}

    # Strip version suffixes for querying
    base_ids = list({eid.split(".")[0] for eid in ensembl_ids_with_versions})
    print(f"  Querying mygene.info for {len(base_ids)} unique Ensembl IDs...")

    try:
        mg = mygene.MyGeneInfo()
        # Query in batches — mygene handles large lists automatically
        results = mg.querymany(
            base_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            verbose=False
        )

        base_map = {}
        for r in results:
            if "symbol" in r and not r.get("notfound", False):
                base_map[r["query"]] = r["symbol"]

        print(f"  Mapped {len(base_map)} / {len(base_ids)} IDs successfully")

    except Exception as e:
        print(f"  [WARNING] mygene query failed: {e}")
        print("  Falling back to Ensembl IDs.")
        return {}

    # Build final map including versioned IDs (e.g. ENSG00000141510.15 -> TP53)
    full_map = {}
    for eid in ensembl_ids_with_versions:
        base = eid.split(".")[0]
        full_map[eid] = base_map.get(base, eid)  # fall back to original if not found

    return full_map



# DIFFERENTIAL EXPRESSION
def run_differential_expression(X_log, labels, fdr_threshold=0.05, top_n=200):
   
    results = {}
    unique_clusters = sorted(np.unique(labels))
    n_genes = X_log.shape[1]

    print(f"\nRunning differential expression across {len(unique_clusters)} clusters "
          f"x {n_genes} genes. This may take 1-2 minutes...")

    for cl in unique_clusters:
        in_mask  = labels == cl
        out_mask = labels != cl

        X_in  = X_log.values[in_mask]   # shape (n_in,  n_genes)
        X_out = X_log.values[out_mask]  # shape (n_out, n_genes)

        pvals  = np.zeros(n_genes)
        log2fc = np.zeros(n_genes)

        for g in range(n_genes):
            stat, p = mannwhitneyu(X_in[:, g], X_out[:, g], alternative="two-sided")
            pvals[g]  = p
            # log2 fold-change: mean(in) - mean(out) in log space = log2(FC)
            log2fc[g] = X_in[:, g].mean() - X_out[:, g].mean()

        # FDR correction (Benjamini-Hochberg)
        reject, padj, _, _ = multipletests(pvals, method="fdr_bh")

        df = pd.DataFrame({
            "gene":   X_log.columns,
            "log2fc": log2fc,
            "pval":   pvals,
            "padj":   padj,
            "significant": reject
        }).sort_values("padj")

        results[cl] = df
        n_sig = reject.sum()
        print(f"  Cluster {cl}: {n_sig} significant genes (FDR < {fdr_threshold})")

    return results


def save_de_results(de_results, out_dir):
    de_dir = os.path.join(out_dir, "differential_expression")
    os.makedirs(de_dir, exist_ok=True)
    for cl, df in de_results.items():
        path = os.path.join(de_dir, f"DE_cluster_{cl}_vs_rest.tsv")
        df.to_csv(path, sep="\t", index=False)
    print(f"\nDE results saved to {de_dir}/")



# GENE ONTOLOGY ENRICHMENT  (via gseapy)


def run_go_enrichment(de_results, top_n_genes=100, out_dir="outputs"):
    """
    For the top upregulated genes in each cluster, query Enrichr
    for GO Biological Process terms.
    Saves results and a summary bar chart per cluster.
    """
    try:
        import gseapy as gp
    except ImportError:
        print("\n[WARNING] gseapy not installed. Skipping GO enrichment.")
        print("  Install with: pip install gseapy")
        return {}

    enrich_dir = os.path.join(out_dir, "go_enrichment")
    os.makedirs(enrich_dir, exist_ok=True)

    enrich_results = {}

    for cl, df in de_results.items():
        # Top upregulated significant genes
        up_genes = (
            df[df["significant"] & (df["log2fc"] > 0)]
            .sort_values("log2fc", ascending=False)
            .head(top_n_genes)["gene"]
            .tolist()
        )

        # Strip Ensembl version suffix if present (e.g. ENSG00000141510.15 -> ENSG00000141510)
        up_genes_clean = [g.split(".")[0] for g in up_genes]

        if len(up_genes_clean) < 5:
            print(f"  Cluster {cl}: too few upregulated genes for enrichment, skipping.")
            continue

        print(f"  Running GO enrichment for Cluster {cl} ({len(up_genes_clean)} genes)...")

        try:
            enr = gp.enrichr(
                gene_list=up_genes_clean,
                gene_sets=["GO_Biological_Process_2023"],
                organism="human",
                outdir=None,
                verbose=False
            )

            res = enr.results.copy()
            res = res[res["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
            enrich_results[cl] = res

            # Save full table
            res.to_csv(
                os.path.join(enrich_dir, f"GO_cluster_{cl}.tsv"),
                sep="\t", index=False
            )

            # Plot top 15 terms
            if len(res) > 0:
                top_terms = res.head(15).copy()
                top_terms["-log10(padj)"] = -np.log10(top_terms["Adjusted P-value"] + 1e-300)
                top_terms = top_terms.sort_values("-log10(padj)")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_terms["Term"], top_terms["-log10(padj)"], color="steelblue")
                ax.set_xlabel("-log10(Adjusted P-value)")
                ax.set_title(f"Top GO Biological Process Terms — Cluster {cl}")
                ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", label="FDR=0.05")
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(enrich_dir, f"GO_cluster_{cl}_barplot.png"),
                    dpi=200
                )
                plt.close()
                print(f"    -> {len(res)} significant GO terms found")
            else:
                print(f"    -> No significant GO terms at FDR < 0.05")

        except Exception as e:
            print(f"    -> Enrichr query failed for Cluster {cl}: {e}")

    return enrich_results



# MARKER GENE HEATMAP
def plot_marker_heatmap(X_log, labels, y_true, de_results, id_map,
                        top_n_per_cluster=10, fig_dir="figures"):
    """
    Select top N upregulated marker genes per cluster,
    plot a heatmap of expression across all samples,
    sorted by cluster then subtype.
    """
    print("\nBuilding marker gene heatmap...")

    # Collect top marker genes per cluster
    marker_genes = []
    gene_cluster_map = {}  # gene -> cluster it marks

    for cl, df in de_results.items():
        top = (
            df[df["significant"] & (df["log2fc"] > 0)]
            .sort_values("log2fc", ascending=False)
            .head(top_n_per_cluster)["gene"]
            .tolist()
        )
        for g in top:
            if g not in gene_cluster_map:
                marker_genes.append(g)
                gene_cluster_map[g] = cl

    # Filter to genes that exist in expression matrix
    marker_genes = [g for g in marker_genes if g in X_log.columns]

    if len(marker_genes) == 0:
        print("  No marker genes found for heatmap. Skipping.")
        return

    print(f"  Using {len(marker_genes)} marker genes across {len(de_results)} clusters")

    # Convert Ensembl IDs to gene symbols for y-axis labels
    y_labels = [id_map.get(g, g) for g in marker_genes]

    # Build plotting matrix: samples sorted by cluster
    sort_df = pd.DataFrame({
        "cluster": labels,
        "subtype": y_true.values
    }, index=X_log.index).sort_values(["cluster", "subtype"])

    X_plot = X_log.loc[sort_df.index, marker_genes].T

    # Row annotation: which cluster each gene marks
    row_colors_vals = [gene_cluster_map[g] for g in marker_genes]
    cluster_palette = sns.color_palette("tab10", n_colors=len(de_results))
    row_colors = [cluster_palette[c] for c in row_colors_vals]

    # Column annotation: subtype
    subtype_list = sort_df["subtype"].tolist()
    unique_subtypes = sorted(set(subtype_list))
    subtype_palette = sns.color_palette("Set2", n_colors=len(unique_subtypes))
    subtype_color_map = {s: subtype_palette[i] for i, s in enumerate(unique_subtypes)}
    col_colors = [subtype_color_map[s] for s in subtype_list]

    # Z-score each gene for visualization
    X_z = X_plot.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        X_z,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3, vmax=3,
        xticklabels=False,
        yticklabels=y_labels,
        cbar_kws={"label": "Z-score (log2 TPM+1)"}
    )
    ax.set_xlabel("Samples (sorted by cluster)")
    ax.set_ylabel("Marker Genes")
    ax.set_title("Marker Gene Expression Heatmap\nTop upregulated genes per cluster — GMM(k=5, cov=tied)")

    # Add cluster legend
    from matplotlib.patches import Patch
    cluster_handles = [
        Patch(color=cluster_palette[c], label=f"Cluster {c}")
        for c in sorted(de_results.keys())
    ]
    subtype_handles = [
        Patch(color=subtype_color_map[s], label=s)
        for s in unique_subtypes
    ]
    legend1 = ax.legend(
        handles=cluster_handles, title="Cluster",
        bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=7
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=subtype_handles, title="True Subtype",
        bbox_to_anchor=(1.15, 0.5), loc="upper left", fontsize=7
    )

    plt.tight_layout()
    out_path = os.path.join(fig_dir, "marker_gene_heatmap.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



# SUMMARY TABLE


def save_summary_table(de_results, labels, y_true, out_dir):
    """
    Save a clean summary: for each cluster, the top 20 marker genes
    with their log2FC and adjusted p-value. Easy to paste into your thesis.
    """
    rows = []
    for cl, df in de_results.items():
        top = (
            df[df["significant"] & (df["log2fc"] > 0)]
            .sort_values("log2fc", ascending=False)
            .head(20)
        )
        top = top.copy()
        top.insert(0, "cluster", cl)
        rows.append(top)

    summary = pd.concat(rows, ignore_index=True)
    path = os.path.join(out_dir, "marker_genes_summary.tsv")
    summary.to_csv(path, sep="\t", index=False)
    print(f"\nMarker gene summary saved to {path}")
    return summary



# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x",             default="X_tpm_samples_by_genes.tsv")
    parser.add_argument("--y",             default="y_subtype.tsv")
    parser.add_argument("--k",             type=int, default=5)
    parser.add_argument("--top_var_genes", type=int, default=2000)
    parser.add_argument("--n_pcs",         type=int, default=10)
    parser.add_argument("--top_de_genes",  type=int, default=100,
                        help="Top upregulated genes per cluster for GO enrichment")
    parser.add_argument("--top_heatmap",   type=int, default=10,
                        help="Top marker genes per cluster for heatmap")
    parser.add_argument("--random_state",  type=int, default=0)
    parser.add_argument("--out_dir",       default="outputs")
    parser.add_argument("--fig_dir",       default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ── Load ──
    print("=" * 60)
    print("Step 1: Loading data...")
    X, y = load_inputs(args.x, args.y)
    print(f"  X: {X.shape}, y: {y.shape}")

    # ── Preprocess & PCA ──
    print("\nStep 2: Preprocessing & PCA...")
    Z, X_log = preprocess_pca(X, args.top_var_genes, args.n_pcs, args.random_state)
    print(f"  PCA shape: {Z.shape}")

    # ── Cluster ──
    print("\nStep 3: Clustering (GMM k=5, cov=tied)...")
    labels = get_cluster_labels(Z, args.k, args.random_state)

    # ── Differential Expression ──
    print("\nStep 4: Differential Expression...")
    de_results = run_differential_expression(X_log, labels)
    save_de_results(de_results, args.out_dir)

    # ── Gene Symbol Conversion ──
    print("\nStep 5: Converting Ensembl IDs to gene symbols (requires internet)...")
    id_map = build_id_map(X_log.columns.tolist())

    # ── Marker Gene Summary ──
    print("\nStep 6: Saving marker gene summary...")
    summary = save_summary_table(de_results, labels, y, args.out_dir)
    print("\nTop 5 marker genes per cluster:")
    for cl in sorted(de_results.keys()):
        top5 = (
            de_results[cl][de_results[cl]["significant"] & (de_results[cl]["log2fc"] > 0)]
            .sort_values("log2fc", ascending=False)
            .head(5)["gene"]
            .tolist()
        )
        # Show gene symbols if available
        top5_labels = [id_map.get(g, g) for g in top5]
        print(f"  Cluster {cl}: {top5_labels}")

    # ── Heatmap ──
    print("\nStep 7: Plotting marker gene heatmap...")
    plot_marker_heatmap(X_log, labels, y, de_results, id_map,
                        top_n_per_cluster=args.top_heatmap,
                        fig_dir=args.fig_dir)

    # ── GO Enrichment ──
    print("\nStep 8: GO Biological Process Enrichment (requires internet)...")
    run_go_enrichment(de_results, top_n_genes=args.top_de_genes, out_dir=args.out_dir)

    print("\n" + "=" * 60)
    print("Done! Output summary:")
    print(f"  outputs/differential_expression/  — DE results per cluster")
    print(f"  outputs/marker_genes_summary.tsv  — top marker genes table")
    print(f"  outputs/go_enrichment/            — GO enrichment tables + plots")
    print(f"  figures/marker_gene_heatmap.png   — marker gene heatmap (gene symbols)")
    print("=" * 60)


if __name__ == "__main__":
    main()
