import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

survival = pd.read_csv("TCGA-BRCA.survival.tsv", sep="\t")
clear_df = pd.read_csv("clear_samples.tsv", sep="\t")
ambiguous_df = pd.read_csv("ambiguous_samples.tsv", sep="\t")

# Rename first column of classifier files to Sample
clear_df = clear_df.rename(columns={clear_df.columns[0]: "Sample"})
ambiguous_df = ambiguous_df.rename(columns={ambiguous_df.columns[0]: "Sample"})

# Create patient IDs from TCGA barcodes
clear_df["Patient"] = clear_df["Sample"].str[:12]
ambiguous_df["Patient"] = ambiguous_df["Sample"].str[:12]
survival["Patient"] = survival["_PATIENT"]

# Create groups
clear_luma = clear_df[clear_df["Probability_LumB"] < 0.3].copy()
clear_lumb = clear_df[clear_df["Probability_LumB"] > 0.7].copy()

clear_luma["Group"] = "Clear LumA"
clear_lumb["Group"] = "Clear LumB"
ambiguous_df["Group"] = "Ambiguous"

combined = pd.concat([clear_luma, ambiguous_df, clear_lumb])

# Merge using patient ID
merged = combined.merge(
    survival,
    on="Patient",
    how="inner"
)

merged = merged[["Sample", "Patient", "Group", "OS", "OS.time"]]
merged = merged.dropna()

print("Merged Data:")
print(merged.head())

print("\nGroup counts:")
print(merged["Group"].value_counts())

merged.to_csv("merged_survival_groups.tsv", sep="\t", index=False)
print("\nSaved: merged_survival_groups.tsv")

# -----------------------------
# Kaplan-Meier Survival Analysis
# -----------------------------

kmf = KaplanMeierFitter()

plt.figure(figsize=(8,6))

groups = ["Clear LumA", "Ambiguous", "Clear LumB"]

for group in groups:

    subset = merged[merged["Group"] == group]

    kmf.fit(
        durations=subset["OS.time"],
        event_observed=subset["OS"],
        label=group
    )

    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan-Meier Survival Analysis")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")

plt.tight_layout()
plt.savefig("kaplan_meier_survival.png", dpi=300)

print("\nSaved: kaplan_meier_survival.png")

# -----------------------------
# Log-rank statistical testing
# -----------------------------

luma = merged[merged["Group"] == "Clear LumA"]
lumb = merged[merged["Group"] == "Clear LumB"]
ambiguous = merged[merged["Group"] == "Ambiguous"]

# LumA vs LumB
result1 = logrank_test(
    luma["OS.time"],
    lumb["OS.time"],
    event_observed_A=luma["OS"],
    event_observed_B=lumb["OS"]
)

print("\nLog-rank Test: Clear LumA vs Clear LumB")
print("p-value:", result1.p_value)

# Ambiguous vs LumB
result2 = logrank_test(
    ambiguous["OS.time"],
    lumb["OS.time"],
    event_observed_A=ambiguous["OS"],
    event_observed_B=lumb["OS"]
)

print("\nLog-rank Test: Ambiguous vs Clear LumB")
print("p-value:", result2.p_value)

# Ambiguous vs LumA
result3 = logrank_test(
    ambiguous["OS.time"],
    luma["OS.time"],
    event_observed_A=ambiguous["OS"],
    event_observed_B=luma["OS"]
)

print("\nLog-rank Test: Ambiguous vs Clear LumA")
print("p-value:", result3.p_value)
