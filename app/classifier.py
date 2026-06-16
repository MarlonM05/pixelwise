import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/digit_classifier_v1.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modell-Datei nicht gefunden unter: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def classify_batch(images: np.ndarray) -> list:
    """
    Nimmt ein NumPy-Array von Bildern (Batch) entgegen und gibt die Vorhersagen zurück.
    Eingabeform: (Batch-Größe, 28, 28)
    """
    model = get_model()
    
    batch_size = images.shape[0]
    flattened = images.reshape(batch_size, -1)
    
    predictions = model.predict(flattened)
    probabilities = model.predict_proba(flattened)
    
    results = []
    for i in range(batch_size):
        pred_label = int(predictions[i])
        
        # Finde den korrekten Array-Index (0-8) für die vorhergesagte Klasse heraus
        pred_index = np.where(model.classes_ == predictions[i])[0][0]
        conf = float(probabilities[i][pred_index])
        
        results.append({
            "prediction": str(pred_label),
            "confidence": conf
        })
        
    return results
