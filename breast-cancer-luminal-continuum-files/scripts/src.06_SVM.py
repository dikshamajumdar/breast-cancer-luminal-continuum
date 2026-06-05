#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# Config

TOP_VAR_GENES = 2000
TEST_SIZE = 0.20
RANDOM_STATE = 42

FIG_DIR = "figures"
OUT_DIR = "outputs"

KEEP_SUBTYPES = ["BRCA_LumA", "BRCA_LumB"]


# Load Data

def load_luma_lumb_data(
    x_path="X_tpm_samples_by_genes.tsv",
    y_path="y_subtype.tsv"
):
    X = pd.read_csv(x_path, sep="\t", index_col=0)
    y_df = pd.read_csv(y_path, sep="\t", index_col=0)

    if "Subtype" in y_df.columns:
        y = y_df["Subtype"]
    else:
        y = y_df.iloc[:, 0]

    # Match sample IDs
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    # Keep LumA and LumB only
    mask = y.isin(KEEP_SUBTYPES)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    return X, y


# Preprocess

def preprocess_expression(X):
    # log transform
    X_log = np.log2(X.astype(float) + 1.0)

    # top variable genes
    top_genes = (
        X_log.var(axis=0)
        .sort_values(ascending=False)
        .head(TOP_VAR_GENES)
        .index
    )

    X_sel = X_log[top_genes].copy()
    return X_sel


# Save Confusion Matrix

def save_confusion_matrix(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("SVM Confusion Matrix (LumA vs LumB)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# Save ROC Curve

def save_roc_curve(fpr, tpr, roc_auc, outpath):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM ROC Curve (Detecting Luminal B)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# Save Probability Distribution

def save_probability_distribution(y_prob, y_test, outpath):
    plt.figure(figsize=(6, 5))

    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.6, label="LumA")
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.6, label="LumB")

    plt.xlabel("Predicted Probability (LumB)")
    plt.ylabel("Number of Samples")
    plt.title("SVM Probability Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# Main

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    X, y = load_luma_lumb_data()

    print("\nSubtype counts:")
    print(y.value_counts())

    print("\nPreprocessing...")
    X_sel = preprocess_expression(X)
    print("Filtered shape:", X_sel.shape)

    # Encode labels
    y_binary = y.map({
        "BRCA_LumA": 0,
        "BRCA_LumB": 1
    })

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y_binary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Train SVM
    print("\nTraining SVM...")
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,   # needed for ROC + probabilities
        random_state=RANDOM_STATE
    )

    svm.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]

  
    # Metrics
 
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    save_confusion_matrix(
        cm,
        ["LumA", "LumB"],
        f"{FIG_DIR}/confusion_matrix.png"
    )

    print(f"Saved: {FIG_DIR}/confusion_matrix.png")


    # ROC Curve
   
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"AUC: {roc_auc:.4f}")

    save_roc_curve(
        fpr,
        tpr,
        roc_auc,
        f"{FIG_DIR}/roc_curve.png"
    )

    print(f"Saved: {FIG_DIR}/roc_curve.png")

  
    # Probability Distribution 

    save_probability_distribution(
        y_prob,
        y_test,
        f"{FIG_DIR}/probability_distribution.png"
    )

    print(f"Saved: {FIG_DIR}/probability_distribution.png")


    # Save Predictions

    pred_df = pd.DataFrame({
        "True": y_test,
        "Predicted": y_pred,
        "Probability_LumB": y_prob
    }, index=X_test.index)

    pred_df.to_csv(f"{OUT_DIR}/svm_predictions.tsv", sep="\t")

    print("\nDone.")


if __name__ == "__main__":
    main()
