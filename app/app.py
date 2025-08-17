# --- add project/src to Python path so we can import chains.py ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -----------------------------------------------------------------

from chains import analyze_text

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import mlflow
st.set_page_config(page_title="Finance LLM — Complaints", page_icon="💳", layout="wide")
load_dotenv()  # load .env if present

# Session state defaults
if "last_single" not in st.session_state:
    st.session_state.last_single = None
if "last_batch_df" not in st.session_state:
    st.session_state.last_batch_df = None

# -------------------------------------------------------------------
# Sidebar: OpenAI config + options
# -------------------------------------------------------------------
st.sidebar.title("⚙️ OpenAI Settings")

detected = "set" if os.getenv("OPENAI_API_KEY") else "not set"
st.sidebar.caption(f"OPENAI_API_KEY is **{detected}**")

openai_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
openai_model = st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if st.sidebar.button("Use OpenAI key for this session"):
    if not openai_key:
        st.sidebar.error("Please provide your OPENAI_API_KEY.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENAI_MODEL"] = openai_model
        # make sure no Azure vars interfere
        for k in [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT",
        ]:
            os.environ.pop(k, None)
        st.sidebar.success("OpenAI configured ✅")

log_mlflow = st.sidebar.checkbox("Log to MLflow", value=True)
st.sidebar.caption("Tip: run `mlflow ui` in a terminal to inspect runs.")
st.sidebar.markdown(f"**MLflow tracking URI:** `{mlflow.get_tracking_uri()}`")

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("💳 Finance LLM — Complaint Summaries & Risk Signals")
st.caption("OpenAI + LangChain structured outputs • Pydantic • Optional MLflow logging")

# Optional banner if you added one
banner_path = Path("assets/banner.png")
if banner_path.exists():
    # display in a centered column so it doesn't get too tall
    l, m, r = st.columns([1, 2, 1])
    with m:
        st.image(str(banner_path), use_container_width=True)

# -------------------------------------------------------------------
# Tabs: single / batch
# -------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["📝 Single analysis", "📄 Batch CSV"])

# ----------------------- Single analysis ----------------------------
with tab_single:
    st.subheader("Analyze one complaint")

    sample = "Bank charged me a late fee despite auto-pay; I contacted support twice with no resolution."
    text = st.text_area("Complaint text", value=sample, height=160)

    colA, colB = st.columns([1, 3])
    with colA:
        go = st.button("Analyze")
    with colB:
        st.write("")

    if go:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                t0 = time.time()
                try:
                    res = analyze_text(text, run_mlflow=log_mlflow)
                    st.session_state.last_single = res
                    st.success(f"Done in {time.time() - t0:.2f}s")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

    if st.session_state.last_single:
        res = st.session_state.last_single

        st.markdown("### Result")
        st.markdown("**Summary**")
        st.write(res.get("summary", ""))

        cat = res.get("risk_category", "Other")
        conf = res.get("risk_confidence", None)
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
        st.markdown(f"**Risk category:** `{cat}`   |   **Confidence:** `{conf_str}`")

        ext = res.get("extraction", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("Customer Entities")
            st.write(ext.get("customer_entities", []))
        with c2:
            st.caption("Company Entities")
            st.write(ext.get("company_entities", []))
        with c3:
            st.caption("Amounts")
            st.write(ext.get("amounts", []))
        with c4:
            st.caption("Dates")
            st.write(ext.get("dates", []))

# ------------------------- Batch analysis ---------------------------
with tab_batch:
    st.subheader("Analyze a CSV of complaints")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if df.empty:
            st.warning("CSV appears empty.")
            st.stop()

        cols = list(df.columns)
        default_text_col = "text" if "text" in cols else cols[0]
        text_col = st.selectbox("Text column", options=cols, index=cols.index(default_text_col))

        max_rows = min(500, len(df))
        n_max = st.slider("Max rows to process", min_value=10, max_value=max_rows, value=min(100, max_rows), step=10)

        if len(df) > n_max:
            df_proc = df.sample(n_max, random_state=42).reset_index(drop=True)
        else:
            df_proc = df.copy().reset_index(drop=True)

        if st.button("Run batch analysis"):
            progress = st.progress(0)
            out_rows = []
            start = time.time()

            for i, row in df_proc.iterrows():
                text_i = str(row[text_col])
                try:
                    res = analyze_text(text_i, run_mlflow=log_mlflow)
                except Exception as e:
                    res = {
                        "summary": "",
                        "risk_category": "Other",
                        "risk_confidence": 0.0,
                        "extraction": {
                            "customer_entities": [],
                            "company_entities": [],
                            "amounts": [],
                            "dates": [],
                        },
                        "error": str(e),
                    }
                out_rows.append({**row.to_dict(), **res})
                progress.progress(int((i + 1) / len(df_proc) * 100))

            out_df = pd.DataFrame(out_rows)
            st.session_state.last_batch_df = out_df
            progress.empty()
            st.success(f"Batch done: {len(out_df)} rows in {time.time() - start:.1f}s")

    if st.session_state.last_batch_df is not None:
        out_df = st.session_state.last_batch_df
        st.markdown("### Results preview")
        st.dataframe(out_df.head(50), use_container_width=True)

        if "risk_category" in out_df.columns:
            st.markdown("**Risk category distribution**")
            dist = out_df["risk_category"].value_counts().sort_index()
            st.bar_chart(dist)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download results CSV",
            data=csv_bytes,
            file_name="complaints_scored.csv",
            mime="text/csv",
        )

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.caption("Built with OpenAI + LangChain structured outputs, Pydantic, and MLflow.")
