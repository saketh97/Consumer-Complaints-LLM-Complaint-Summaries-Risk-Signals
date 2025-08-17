# src/evaluate.py
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

CATS = ["Billing","Collections","DataPrivacy","Fees","Fraud","MisSelling","Other"]

def has_any(s, terms):
    s = s.lower()
    return any(t in s for t in terms)

def coerce(issue: str, product: str) -> str:
    t_issue = str(issue).lower()
    t_prod  = str(product).lower()
    t = f"{t_issue} {t_prod}"

    # Product-driven shortcuts
    if "credit reporting" in t_prod or "credit reporting" in t_issue or "credit report" in t:
        return "DataPrivacy"
    if "debt collection" in t_prod:
        return "Collections"

    # Collections
    if has_any(t, ["debt collection", "collector", "collection agency", "debt validation"]):
        return "Collections"

    # DataPrivacy / credit reporting & privacy
    if has_any(t, [
        "incorrect information", "credit report", "credit reporting", "privacy",
        "data breach", "investigation", "reinserted", "mixed file", "blocked"
    ]):
        return "DataPrivacy"

    # Fraud / unauthorized
    if has_any(t, [
        "fraud", "scam", "unauthorized", "identity theft", "account taken over",
        "fake", "phishing", "chargeback"
    ]):
        return "Fraud"

    # Fees / Billing split
    if has_any(t, ["fee", "fees", "overdraft", "nsf", "maintenance fee", "annual fee",
                   "late fee", "interest charge", "charge", "charged twice"]):
        # heuristically separate billing-vs-fees
        if has_any(t, ["billing", "statement", "payment allocation", "due date", "bill"]):
            return "Billing"
        return "Fees"

    # Mis-selling / deceptive
    if has_any(t, ["mis-selling", "misselling", "mis selling", "deceptive", "misleading",
                   "upsell", "sold me", "bait and switch"]):
        return "MisSelling"

    return "Other"

def eval_risk(scored_csv: str):
    df = pd.read_csv(scored_csv)
    need = {"risk_category", "Issue", "Product"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"Expected columns missing: {missing}")

    y_true = [coerce(i, p) for i, p in zip(df["Issue"], df["Product"])]
    y_pred = df["risk_category"].astype(str).tolist()

    print("True label distribution:\n", pd.Series(y_true).value_counts(), "\n")
    print("Pred label distribution:\n", pd.Series(y_pred).value_counts(), "\n")

    # cleaner report (silence undefined-metric warnings for empty classes)
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Optional: confusion matrix
    try:
        import numpy as np, matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred, labels=CATS)
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CATS)), CATS, rotation=45, ha="right")
        ax.set_yticks(range(len(CATS)), CATS)
        for i in range(len(CATS)):
            for j in range(len(CATS)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    eval_risk("data/processed/complaints_scored.csv")
