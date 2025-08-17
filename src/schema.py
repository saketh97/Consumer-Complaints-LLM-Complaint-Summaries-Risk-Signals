# src/schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List

class Extraction(BaseModel):
    customer_entities: List[str] = Field(default_factory=list)
    company_entities: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)

class LLMOutput(BaseModel):
    summary: str = Field(..., description="2-3 sentence neutral summary of the complaint")
    risk_category: str = Field(..., description="One of: Billing, Collections, DataPrivacy, Fees, MisSelling, Fraud, Other")
    risk_confidence: float = Field(..., ge=0, le=1)  # <-- float, not str
    extraction: Extraction

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
        # clamp to [0,1]
        v = float(v)
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        return v
