import pandas as pd

RAW = "RawData"
expr_path = f"{RAW}/TCGA-BRCA.star_tpm.tsv"
clin_path = f"{RAW}/brca_tcga_pan_can_atlas_2018_clinical_data.tsv"

# Load expression (genes x samples)
expr = pd.read_csv(expr_path, sep="\t")
expr = expr.set_index("Ensembl_ID")
print("Expression (genes x samples):", expr.shape)


# Keep only PRIMARY TUMOR samples (TCGA sample type = "01")
# Positions 13-15 correspond to the sample type code

tumor_cols = [c for c in expr.columns if c[13:15] == "01"]
expr = expr[tumor_cols]
print("Expression after keeping tumor (genes x tumor samples):", expr.shape)


# Normalize expression sample IDs to first 15 chars:

expr.columns = [c[:15] for c in expr.columns]


expr = expr.groupby(expr.columns, axis=1).mean()

print("Expression after collapsing duplicate IDs (genes x unique tumor samples):", expr.shape)


#load the clinical subtype file in this step
clin = pd.read_csv(clin_path, sep="\t", comment="#")
clin = clin[["Sample ID", "Subtype"]].dropna()
print("Clinical (samples w/ subtype):", clin.shape)

# Normalize clinical IDs to first 15 chars too
clin["Sample ID"] = clin["Sample ID"].str[:15]
clin = clin.drop_duplicates(subset="Sample ID")

#match the samples
expr_samples = set(expr.columns)
clin_samples = set(clin["Sample ID"])
common = sorted(expr_samples.intersection(clin_samples))

print("Matched samples:", len(common))

#build the final dataset
expr = expr[common]
clin = clin.set_index("Sample ID").loc[common]

X = expr.T  # samples x genes
y = clin["Subtype"]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

print("\nSubtype counts:\n")
print(y.value_counts())


#save for next steps 
X.to_csv("X_tpm_samples_by_genes.tsv", sep="\t")
y.to_csv("y_subtype.tsv", sep="\t", header=True)

print("\nSaved: X_tpm_samples_by_genes.tsv and y_subtype.tsv")
