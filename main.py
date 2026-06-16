import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import ClassifyRequest, ClassifyResponse
from app.classifier import classify_batch
from app.database import get_db, PredictionModel

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1"}

@app.get("/results")
def results(db: Session = Depends(get_db)):
    # Holt die letzten 100 Vorhersagen aus der Datenbank
    records = db.query(PredictionModel).order_by(PredictionModel.created_at.desc()).limit(100).all()
    return [
        {
            "id": r.id,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat()
        } for r in records
    ]

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest, db: Session = Depends(get_db)):
    try:
        # Array für Inferenz vorbereiten
        arr = np.array(req.pixels, dtype=np.uint8)[np.newaxis]
        res = classify_batch(arr)[0]
        
        # Ergebnis in der PostgreSQL-Datenbank speichern
        db_prediction = PredictionModel(
            prediction=res["prediction"],
            confidence=res["confidence"]
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return res
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
