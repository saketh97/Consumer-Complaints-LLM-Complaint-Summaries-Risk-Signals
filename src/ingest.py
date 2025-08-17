import pandas as pd
from pathlib import Path

RAW=Path('data/raw')
PROC = Path('data/processed')

def load_sample(in_csv: str, out_csv: str,n: int=5000):
    RAW.mkdir(parents=True,exist_ok=True)
    PROC.mkdir(parents=True,exist_ok=True)
    
    df = pd.read_csv(in_csv)
    
    # keep rows with narrative text
    col_text = "Consumer complaint narrative"
    cols_keep = [c for c in [col_text,"Product","Issue","Company","Data received","State"] if c in df.columns]
    df = df[cols_keep].dropna(subset=[col_text]).copy()
    
    # thin to n and basic cleanup
    
    if len(df) > n: 
        df=df.sample(n,random_state=42)
        df.rename(columns={col_text:"text"},inplace=True)
        df["text"] = df["text"].str.strip()
        
        df.to_csv(out_csv, index=False)
        print(f"saved -> {out_csv}, rows={len(df)}")
        
if __name__=="__main__":
    load_sample("data/raw/complaints.csv","data/processed/complaints_clean.csv",n=5000)
