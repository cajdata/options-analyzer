# app/models.py

from pydantic import BaseModel
from typing import Optional

class ParsedOptionData(BaseModel):
    ticker: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None
    expiration: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

class AnalysisResult(BaseModel):
    score: Optional[float] = None
    rating: Optional[str] = None
    message: Optional[str] = None

class AnalysisResponse(BaseModel):
    parsed_data: ParsedOptionData
    analysis_result: AnalysisResult
