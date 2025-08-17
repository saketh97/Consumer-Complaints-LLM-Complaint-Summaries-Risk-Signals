# --- add at the very top (with other imports)
import os, time, argparse, json
import contextlib
from pathlib import Path
import mlflow
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from schema import LLMOutput

load_dotenv()

# === MLflow tracking location: force to <project-root>/mlruns ===
ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("finance-llm")
# ===============================================================

def _llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)

pyd_parser = PydanticOutputParser(pydantic_object=LLMOutput)  # renamed to avoid name clashes

SYSTEM = """ You are a careful financial complaints analyst.
Return ONLY valid JSON matching the schema and be neutral, concise, and specific.
Risk categories (Return exactly one of these): Billing, Collections, DataPrivacy, Fees, MisSelling, Fraud, Other.
IMPORTANT:
- 'risk_confidence' MUST be a number between 0 and 1 (no quotes, not a string, not a percentage).
- Arrays must contain plain strings.
- Do not include any keys not in the schema.
"""

USER_TMPL = """
Text: \"\"\"{text}\"\"\"

Respond in JSON only, matching this schema:
{schema}

Note: risk_confidence must be a bare number (e.g., 0.74), not "0.74" and not "74%".
"""

def analyze_text(text: str, run_mlflow: bool = True) -> Dict[str, Any]:
    llm = _llm()
    prompt = USER_TMPL.format(text=text, schema=pyd_parser.get_format_instructions())
    msgs = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]

    t0 = time.time()
    resp = llm.invoke(msgs)
    latency = time.time() - t0

    # parse JSON to Pydantic
    try:
        result: LLMOutput = pyd_parser.parse(resp.content)
        obj = result.model_dump()
    except ValidationError as e:
        obj = {"parse_error": str(e), "raw": resp.content}

    # MLflow logging (safe nested behavior)
    if run_mlflow:
        parent_active = mlflow.active_run() is not None
        with mlflow.start_run(run_name="analyze_single", nested=parent_active):
            mlflow.log_metric("latency_s", latency)
            usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"tokens_{k}", v)

            # optional context params
            mlflow.log_param("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    return obj

def analyze_csv(in_csv: str, out_csv: str, limit: int = 200):
    df = pd.read_csv(in_csv).dropna(subset=["text"])
    if len(df) > limit:
        df = df.sample(limit, random_state=42)

    rows = []
    with mlflow.start_run(run_name="batch_analyze"):
        for i, r in df.iterrows():
            res = analyze_text(r["text"], run_mlflow=True)  # nested under the batch run
            rows.append({**r.to_dict(), **res})
            # log progress/latency per step (optional)
            mlflow.log_metric("processed", i + 1, step=i + 1)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}, rows={len(rows)}")

if __name__ == "__main__":
    cli = argparse.ArgumentParser(prog="chains", description="LLM complaint analyzer (LangChain + Pydantic + MLflow)")
    sub = cli.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Analyze a single text")
    p_single.add_argument("-t", "--text", required=True, help="Complaint text to analyze")
    p_single.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging for this call")

    p_batch = sub.add_parser("batch", help="Analyze a CSV file")
    p_batch.add_argument("-i", "--in-csv", required=True, help="Input CSV with a 'text' column")
    p_batch.add_argument("-o", "--out-csv", required=True, help="Output CSV path to write results")
    p_batch.add_argument("-n", "--limit", type=int, default=200, help="Max rows to process")

    args = cli.parse_args()

    if args.cmd == "single":
        res = analyze_text(args.text, run_mlflow=not args.no_mlflow)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "batch":
        analyze_csv(args.in_csv, args.out_csv, limit=args.limit)
