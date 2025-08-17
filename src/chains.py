import os, time
import mlflow
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Dict,Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from schema import LLMOutput

load_dotenv()

def _llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0.2)

parser = PydanticOutputParser(pydantic_object=LLMOutput)

SYSTEM = """ You are a careful financial complaints analyst.
Return ONLY valid JSON matching the schema and be neutral, concise, and specific.
Risk categories: Billing, Collections, DataPrivacy, Fees, MisSelling, Fraud, Other.
IMPORTANT:
- 'risk_confidence' MUST be a bare number between 0 and 1 (no quotes, not a percentage).
"""

USER_TMPL="""
Text: \"\"\"{text}\"\"\"

Respond in JSON only, matching this schema:
{schema}

Note: risk_confidence must be a bare number (e.g., 0.74), not "0.74" and not "74%".
"""

def analyze_text(text: str, run_mlflow: bool=True) -> Dict[str,Any]:
    llm=_llm()
    prompt=USER_TMPL.format(text=text, schema=parser.get_format_instructions())
    msgs=[SystemMessage(content=SYSTEM),HumanMessage(content=prompt)]
    
    t0=time.time()
    resp=llm.invoke(msgs)
    latency=time.time()-t0
    
    #parse JSON to Pydantic
    try:
        result: LLMOutput = parser.parse(resp.content)
        obj= result.model_dump()
    except ValidationError as e:
        obj={"parse_error":str(e), "raw":resp.content}
        
    # MLflow logging
    if run_mlflow:
        mlflow.set_experiment("finance-llm")
        with mlflow.start_run(run_name="analyse_single", nested=True):
            mlflow.log_metric("latency_s",latency)
            usage = getattr(resp,"response_metadata",{}).get("token_usage",{})
            for k,v in usage.items():
                mlflow.log_metric(f"tokens_{k}", v if isinstance(v,(int,float)) else 0)
    return obj

def analyse_csv(in_csv: str, out_csv: str, limit: int=200):
    df = pd.read_csv(in_csv).dropna(subset=["text"])
    if len(df) > limit: df = df.sample(limit,random_state=42)
    
    rows =[]
    mlflow.set_experiment("finance-llm")
    with mlflow.start_run(run_name="batch_analyze"):
        for _,r  in df.iterrows():
            res= analyze_text(r["text"],run_mlflow=False)
            rows.append({**r.to_dict(), **res})
    
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    print(f"saved -> {out_csv},rows={len(rows)}")

