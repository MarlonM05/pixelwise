import numpy as np
from fastapi import FastAPI
from app.schemas import ClassifyRequest, ClassifyResponse
from app.classifier import classify_batch

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1"}

@app.get("/results")
def results():
    return {"results": [], "note": "persistence not yet implemented"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    # Macht aus der Liste von Listen ein passendes NumPy Array
    arr = np.array(req.pixels, dtype=np.uint8)[np.newaxis]
    # Führt die Batch-Inferenz aus und gibt das erste Element zurück
    return classify_batch(arr)[0]
