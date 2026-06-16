const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d');
let painting = false;

// Zeichen-Einstellungen (Weiße Tinte auf schwarzem Grund)
ctx.strokeStyle = 'white';
ctx.lineWidth = 20;
ctx.lineCap = 'round';

canvas.addEventListener('mousedown', () => painting = true);
canvas.addEventListener('mouseup', () => { painting = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(e) {
    if (!painting) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = "Zeichne eine Ziffer (1-9)...";
}

// Initialisiere Canvas als schwarz
clearCanvas();

function sendImage() {
    // Erstelle ein temporäres 28x28 Canvas zum Runterskalieren des Bildes
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = [];

    // Konvertiere die RGBA-Werte in ein 2D-Graustufen-Array (28x28)
    for (let i = 0; i < 28; i++) {
        const row = [];
        for (let j = 0; j < 28; j++) {
            const index = (i * 28 + j) * 4;
            row.push(imgData.data[index]); // Nutze den R-Kanal für Graustufe
        }
        pixels.push(row);
    }

    document.getElementById('result').innerText = "Analysiere...";

    // Schicke das Raster per POST an die API
    fetch('/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels: pixels })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('result').innerText = 
            `Vorhersage: ${data.prediction} (Konfidenz: ${(data.confidence * 100).toFixed(1)}%)`;
    })
    .catch(err => {
        document.getElementById('result').innerText = "Fehler bei der Verbindung!";
        console.error(err);
    });
}
