import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#paths
X_path = "X_tpm_samples_by_genes.tsv"
y_path = "y_subtype.tsv"

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

#loads
X = pd.read_csv(X_path, sep="\t", index_col=0)   # rows = samples, cols = genes
y = pd.read_csv(y_path, sep="\t", index_col=0)   # index = samples, column = Subtype
# y could load as 1-col dataframe; make it a Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Ensure same order
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)
print("Subtype counts:\n", y.value_counts())


# log2(TPM + 1)
X_log = np.log2(X + 1)

# Filter: keep top N most variable genes (speed + signal)
TOP_N = 5000  # change to 2000 if your laptop is slow; 5000 is usually fine
gene_var = X_log.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(TOP_N).index
X_filt = X_log[top_genes]

print("After top-variance filtering:", X_filt.shape)


# Standardize (z-score per gene)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_filt.values)

# PCA
pca = PCA(n_components=10, random_state=0)
scores = pca.fit_transform(X_scaled)

expl = pca.explained_variance_ratio_
print("Explained variance (first 5 PCs):", expl[:5])

scores_df = pd.DataFrame(
    scores,
    index=X_filt.index,
    columns=[f"PC{i}" for i in range(1, scores.shape[1] + 1)]
)
scores_df["Subtype"] = y.values
scores_df.to_csv("outputs/pca_scores.tsv", sep="\t")
print("Saved: outputs/pca_scores.tsv")


# Plot 1: PC1 vs PC2 colored by subtype
plt.figure(figsize=(7, 5))
for subtype in sorted(y.unique()):
    mask = (y == subtype).values
    plt.scatter(scores[mask, 0], scores[mask, 1], s=10, alpha=0.7, label=subtype)

plt.xlabel(f"PC1 ({expl[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({expl[1]*100:.1f}%)")
plt.title("TCGA-BRCA PCA (log2(TPM+1), top variable genes)")
plt.legend(markerscale=2, fontsize=8)
plt.tight_layout()
plt.savefig("figures/pca_pc1_pc2.png", dpi=300)
plt.close()
print("Saved: figures/pca_pc1_pc2.png")


# Plot 2: explained variance bar plot
plt.figure(figsize=(7, 4))
plt.bar(range(1, len(expl) + 1), expl * 100)
plt.xlabel("Principal Component")
plt.ylabel("Explained variance (%)")
plt.title("PCA explained variance")
plt.xticks(range(1, len(expl) + 1))
plt.tight_layout()
plt.savefig("figures/pca_variance.png", dpi=300)
plt.close()
print("Saved: figures/pca_variance.png")
