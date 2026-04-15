from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
import math
import io
from PIL import Image
import pytesseract

app = FastAPI(title="RSDJ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calcular_entropia(texto):
    freq = Counter(texto)
    total = len(texto)
    if total == 0:
        return 0
    return round(-sum((c/total)*math.log2(c/total) for c in freq.values()), 3)

def analizar_texto(texto):
    palabras = texto.split()
    long_media = round(sum(len(p) for p in palabras) / len(palabras), 2)
    repeticion = round(1 - len(set(palabras)) / len(palabras), 2)
    entropia = calcular_entropia("".join(palabras))

    freq = Counter(p.lower() for p in palabras)
    particulas = [w for w, c in freq.items() if len(w) <= 3 and c > 1]

    hipotesis = "Estructura mixta"
    if long_media < 4 and repeticion < 0.3:
        hipotesis = "Lengua aislante probable"
    elif long_media > 7 and repeticion > 0.4:
        hipotesis = "Lengua aglutinante probable"
    elif entropia > 4.5:
        hipotesis = "Texto cifrado / no lingüístico"

    return {
        "texto": texto,
        "hipotesis": hipotesis,
        "longitud_media": long_media,
        "repeticion": repeticion,
        "entropia": entropia,
        "particulas": particulas[:8],
    }

@app.post("/analyze")
def analyze(data: dict):
    return analizar_texto(data["text"])

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    imagen = Image.open(io.BytesIO(await file.read()))
    texto = pytesseract.image_to_string(imagen, lang="spa+eng")
    return analizar_texto(texto)
