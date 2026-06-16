# predict.py
import numpy as np
from app.classifier import classify_batch
from sklearn.datasets import fetch_openml

print("Lade MNIST-Testdaten herunter (kann beim ersten Mal kurz dauern)...")
# Holt die echten MNIST-Daten über scikit-learn
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Wir nehmen die ersten 5 Bilder, bringen sie in das 2D-Format (28x28)
# und wandeln sie in den korrekten Datentyp (uint8) um
images = X[:5].reshape(-1, 28, 28).astype(np.uint8)
truth = y[:5]

print("Führe Modell-Inferenz aus...")
results = classify_batch(images)

print("\n--- Testergebnisse ---")
for i, (r, t) in enumerate(zip(results, truth)):
    print(f"Bild {i+1}: Vorhersage = {r['prediction']} (Konfidenz: {r['confidence']:.2f}) | Echtes Label = {t}")
