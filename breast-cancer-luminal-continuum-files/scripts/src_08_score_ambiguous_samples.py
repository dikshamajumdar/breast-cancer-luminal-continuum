import pandas as pd


# FILE PATHS

expr_file = "RawData/TCGA-BRCA.star_tpm.tsv"
ambiguous_file = "outputs/ambiguous_samples.tsv"
signature_file = "outputs/ambiguity_signature_genes_with_symbols.tsv"


# LOAD DATA

print("Loading data...")

expr = pd.read_csv(expr_file, sep="\t")
ambiguous = pd.read_csv(ambiguous_file, sep="\t")
signature = pd.read_csv(signature_file, sep="\t")


# CLEAN GENE IDS

print("Cleaning gene IDs...")

expr["Ensembl_ID"] = expr["Ensembl_ID"].str.split(".").str[0]


# CLEAN SAMPLE IDS

# Expression file has sample IDs like TCGA-D8-A146-01A
# SVM file may have shorter IDs like TCGA-D8-A146
# We shorten both to the first 12 characters: TCGA-D8-A146

expr.columns = [
    col[:12] if col != "Ensembl_ID" else col
    for col in expr.columns
]

ambiguous = ambiguous.rename(columns={"Unnamed: 0": "Sample_ID"})
ambiguous["Sample_ID"] = ambiguous["Sample_ID"].str[:12]

# remove duplicate columns after shortening TCGA IDs
expr = expr.loc[:, ~expr.columns.duplicated()]


# GET SIGNATURE GENES

sig_genes = signature["Ensembl_ID"].unique()

expr_sig = expr[expr["Ensembl_ID"].isin(sig_genes)]

print(f"Signature genes found: {expr_sig.shape[0]}")


# SET INDEX

expr_sig = expr_sig.set_index("Ensembl_ID")


# GET AMBIGUOUS SAMPLE IDS

ambiguous_samples = ambiguous["Sample_ID"].tolist()

ambiguous_samples = [
    s for s in ambiguous_samples
    if s in expr_sig.columns
]

print(f"Ambiguous samples found in expression data: {len(ambiguous_samples)}")


# EXTRACT EXPRESSION

expr_ambiguous = expr_sig[ambiguous_samples]


# SPLIT GENES BY DIRECTION

higher_genes = signature[
    signature["Direction"] == "Higher_in_Ambiguous"
]["Ensembl_ID"].tolist()

lower_genes = signature[
    signature["Direction"] == "Lower_in_Ambiguous"
]["Ensembl_ID"].tolist()

higher_genes = [g for g in higher_genes if g in expr_ambiguous.index]
lower_genes = [g for g in lower_genes if g in expr_ambiguous.index]

print(f"Higher-in-ambiguous genes used: {len(higher_genes)}")
print(f"Lower-in-ambiguous genes used: {len(lower_genes)}")

# CALCULATE SCORES

print("Calculating scores...")

higher_score = expr_ambiguous.loc[higher_genes].mean(axis=0)
lower_score = expr_ambiguous.loc[lower_genes].mean(axis=0)

ambiguity_score = higher_score - lower_score


# CREATE RESULT TABLE

results = pd.DataFrame({
    "Sample_ID": ambiguity_score.index,
    "Higher_Ambiguous_Gene_Score": higher_score.values,
    "Lower_Ambiguous_Gene_Score": lower_score.values,
    "Ambiguity_Score": ambiguity_score.values
})


# MERGE WITH SVM DATA

results = results.merge(ambiguous, on="Sample_ID", how="left")

# BIOLOGICAL ASSIGNMENT

median_score = results["Ambiguity_Score"].median()

results["Biological_Assignment"] = results["Ambiguity_Score"].apply(
    lambda x: "LuminalB_like" if x > median_score else "LuminalA_like"
)


# SAVE OUTPUT

results.to_csv(
    "ambiguous_biological_assignment.tsv",
    sep="\t",
    index=False
)

print("\nDone.")
print(results.head())
print("\nSaved: ambiguous_biological_assignment.tsv")
