import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ambiguous_biological_assignment.tsv", sep="\t")

plt.figure(figsize=(6,5))

sns.boxplot(
    x="Biological_Assignment",
    y="Ambiguity_Score",
    data=df
)

sns.stripplot(
    x="Biological_Assignment",
    y="Ambiguity_Score",
    data=df,
    color="black",
    alpha=0.6
)

plt.title("Ambiguous Samples Separate into Distinct Biological Groups")
plt.xlabel("")
plt.ylabel("Ambiguity Score")

plt.savefig("ambiguity_score_plot.png", dpi=300, bbox_inches="tight")
plt.show()
