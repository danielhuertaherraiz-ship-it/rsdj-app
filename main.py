from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
import math, io
from PIL import Image
import pytesseract

app = FastAPI(title="RSDJ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def entropy(text):
    freq = Counter(text)
    total = len(text)
    if total == 0:
        return 0
    return round(-sum((c/total)*math.log2(c/total) for c in freq.values()), 3)

def analyze_text(text):
    words = text.split()
    avg = round(sum(len(w) for w in words) / len(words), 2)
    rep = round(1 - len(set(words)) / len(words), 2)
    ent = entropy("".join(words))

    freq = Counter(w.lower() for w in words)
    particles = [w for w,c in freq.items() if len(w)<=3 and c>1][:8]

    hypothesis = "Estructura mixta"
    if avg < 4 and rep < 0.3:
        hypothesis = "Lengua aislante probable"
    elif avg > 7 and rep > 0.4:
        hypothesis = "Lengua aglutinante probable"
    elif ent > 4.5:
        hypothesis = "Texto cifrado / no lingüístico"

    return {
        "texto": text,
        "hipotesis": hypothesis,
        "longitud_media": avg,
        "repeticion": rep,
        "entropia": ent,
        "particulas": particles
    }

@app.post("/analyze")
def analyze(data: dict):
    return analyze_text(data["text"])

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_text(text)
