# src/schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List

ALLOWED = {"Billing","Collections","DataPrivacy","Fees","MisSelling","Fraud","Other"}
# very lightweight synonym map → our 7 buckets
SYN = {
    "billing": "Billing",
    "charges": "Billing",
    "fees": "Fees",
    "collections": "Collections",
    "debt collection": "Collections",
    "credit reporting": "DataPrivacy",
    "data privacy": "DataPrivacy",
    "privacy": "DataPrivacy",
    "fraud": "Fraud",
    "scam": "Fraud",
    "mis-selling": "MisSelling",
    "misselling": "MisSelling",
    "mis selling": "MisSelling",
}

class Extraction(BaseModel):
    customer_entities: List[str] = Field(default_factory=list)
    company_entities: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)

class LLMOutput(BaseModel):
    summary: str = Field(..., description="2-3 sentence neutral summary of the complaint")
    risk_category: str = Field(..., description="One of: Billing, Collections, DataPrivacy, Fees, MisSelling, Fraud, Other")
    risk_confidence: float = Field(..., ge=0, le=1)
    extraction: Extraction

    @field_validator("risk_category", mode="before")
    def _norm_cat(cls, v):
        s = str(v).strip()
        # fast path
        if s in ALLOWED:
            return s
        low = s.lower()
        # normalize via synonyms
        for k, tgt in SYN.items():
            if k in low:
                return tgt
        return "Other"

    @field_validator("risk_confidence", mode="before")
    def _coerce_conf(cls, v):
        # accept "0.9", "90%", 0.9, etc.
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                s = s[:-1]
                v = float(s) / 100.0
            else:
                v = float(s)
        v = float(v)
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        return v
