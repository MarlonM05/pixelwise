from pydantic import BaseModel
from typing import List

class ClassifyRequest(BaseModel):
    # Das Skript erwartet eine Liste von Listen (28 Zeilen x 28 Spalten)
    pixels: List[List[int]]

class ClassifyResponse(BaseModel):
    prediction: str
    confidence: float
    scores: dict[str, float]
