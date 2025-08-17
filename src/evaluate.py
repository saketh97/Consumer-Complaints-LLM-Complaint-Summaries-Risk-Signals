# src/evaluate.py
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Heuristic rules mapping CFPB Issue/Product → our taxonomy
def coerce(issue: str, product: str) -> str:
    t = f"{issue} {product}".lower()

    # Collections
    if "debt" in t and ("collect" in t or "collection" in t):
        return "Collections"

    # Data privacy / credit reporting
    if ("credit report" in t or "credit reporting" in t or "incorrect information" in t
        or "privacy" in t or "identity theft" in t or "data breach" in t):
        return "DataPrivacy"

    # Fraud / unauthorized
    if ("fraud" in t or "scam" in t or "unauthorized" in t or "identity theft" in t
        or "dispute charge" in t or "account taken over" in t or "chargeback" in t):
        return "Fraud"

    # Fees / billing
    if ("fee" in t or "fees" in t or "charge" in t or "billing" in t
        or "overdraft" in t or "late fee" in t or "interest charge" in t):
        # If explicitly billing-y language, prefer Fees/Billing. You can split if you want granularity.
        if "billing" in t:
            return "Billing"
        return "Fees"

    # Mis-selling / misleading sales
    if ("misleading" in t or "deceptive" in t or "sales practice" in t
        or "sold me" in t or "upsell" in t or "mis sell" in t or "missell" in t or "mis-selling" in t):
        return "MisSelling"

    return "Other"

def eval_risk(scored_csv: str):
    df = pd.read_csv(scored_csv)

    # check columns present
    need = {"risk_category", "Issue", "Product"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"Expected columns missing: {missing}")

    # map true labels
    y_true = [coerce(str(i), str(p)) for i, p in zip(df["Issue"], df["Product"])]
    y_pred = df["risk_category"].astype(str).tolist()

    # quick sanity on distribution
    print("True label distribution:", pd.Series(y_true).value_counts())
    print("Pred label distribution:", pd.Series(y_pred).value_counts())

    print("\n" + classification_report(y_true, y_pred, digits=3))

    # optional confusion matrix peek
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        cats = ["Billing","Collections","DataPrivacy","Fees","Fraud","MisSelling","Other"]
        cm = confusion_matrix(y_true, y_pred, labels=cats)
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(cm)
        ax.set_xticks(range(len(cats)), cats, rotation=45, ha="right")
        ax.set_yticks(range(len(cats)), cats)
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, cm[i,j], ha="center", va="center")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    eval_risk("data/processed/complaints_scored.csv")
