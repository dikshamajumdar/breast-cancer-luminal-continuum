import pandas as pd
from scipy.stats import ttest_ind


# LOAD EXPRESSION DATA


expr = pd.read_csv(
    "X_tpm_samples_by_gene_symbols.tsv",
    sep="\t",
    index_col=0
)

print("Expression shape:", expr.shape)



# LOAD SAMPLE FILES

clear_probs = pd.read_csv(
    "outputs/clear_samples.tsv",
    sep="\t",
    index_col=0
)

ambiguous_probs = pd.read_csv(
    "outputs/ambiguous_samples.tsv",
    sep="\t",
    index_col=0
)

probs = pd.concat([clear_probs, ambiguous_probs])
probs = probs[["Probability_LumB"]]

print("Probability shape:", probs.shape)



# MATCH SAMPLES


common_samples = expr.index.intersection(probs.index)

expr = expr.loc[common_samples]
probs = probs.loc[common_samples]

expr["Probability_LumB"] = probs["Probability_LumB"]

print("Matched samples:", expr.shape[0])



# DEFINE GROUPS


clear_luma = expr[expr["Probability_LumB"] < 0.25]

ambiguous = expr[
    (expr["Probability_LumB"] >= 0.40) &
    (expr["Probability_LumB"] <= 0.60)
]

clear_lumb = expr[expr["Probability_LumB"] > 0.75]

print("Clear LumA:", clear_luma.shape[0])
print("Ambiguous:", ambiguous.shape[0])
print("Clear LumB:", clear_lumb.shape[0])



# REMOVE PROBABILITY COLUMN


clear_luma_expr = clear_luma.drop(columns=["Probability_LumB"])
ambiguous_expr = ambiguous.drop(columns=["Probability_LumB"])
clear_lumb_expr = clear_lumb.drop(columns=["Probability_LumB"])



# DIFFERENTIAL EXPRESSION


def differential_expression(group1, group2, group1_name, group2_name):

    results = []

    for gene in group1.columns:
        group1_values = group1[gene]
        group2_values = group2[gene]

        stat, p = ttest_ind(
            group1_values,
            group2_values,
            equal_var=False,
            nan_policy="omit"
        )

        logFC = group1_values.mean() - group2_values.mean()

        results.append([gene, logFC, p])

    results_df = pd.DataFrame(
        results,
        columns=["Gene", "logFC", "pvalue"]
    )

    results_df = results_df.sort_values(
        by="logFC",
        ascending=False
    )

    deg_file = f"{group1_name}_vs_{group2_name}_DEGs.csv"
    results_df.to_csv(deg_file, index=False)

    higher = results_df[results_df["logFC"] > 0].head(300)
    lower = results_df[results_df["logFC"] < 0].tail(300)

    higher_file = f"{group1_name}_higher_than_{group2_name}_genes.txt"
    lower_file = f"{group1_name}_lower_than_{group2_name}_genes.txt"

    higher[["Gene"]].to_csv(
        higher_file,
        index=False,
        header=False
    )

    lower[["Gene"]].to_csv(
        lower_file,
        index=False,
        header=False
    )

    print("Saved:", deg_file)
    print("Saved:", higher_file)
    print("Saved:", lower_file)

    return results_df



# RUN COMPARISONS


deg_amb_vs_luma = differential_expression(
    ambiguous_expr,
    clear_luma_expr,
    "Ambiguous",
    "Clear_LumA"
)

deg_amb_vs_lumb = differential_expression(
    ambiguous_expr,
    clear_lumb_expr,
    "Ambiguous",
    "Clear_LumB"
)
