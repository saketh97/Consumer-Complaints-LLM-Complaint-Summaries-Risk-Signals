import pandas as pd
from sklearn.metrics import classification_report

MAP={
    "Billing Disputes": "Billing",
    "Incorect information on credit report": "DataPrivacy",
    "Debt Collection":"Collections",
    "Fraud or scam":"Fraud",
    "Fees or charges":"Fees",
}

def coerce(y:str) -> str:
    return MAP.get(y,"Other")

def evla_risk(scored_csv: str):
    df = pd.read_csv(scored_csv)
    if not {"risk_category","Issue"}.issubset(df.columns):
        raise ValueError("Expected Columns : risk_category, Issue")
    y_true = df["Issue"].astype(str).map(coerce)
    y_pred = df["risk_category"].astype(str)
    print(classification_report(y_true,y_pred,digits=3)) 
    
if __name__=="__main__":
    evla_risk("data/processed/complaints_scored.csv")