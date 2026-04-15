from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from collections import Counter
from datetime import datetime
import math, io
import numpy as np
from PIL import Image
import pytesseract

# --------------------
# APP
# --------------------
app = FastAPI(title="RSDJ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# DATABASE
# --------------------
DATABASE_URL = "sqlite:///./rsdj.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    source = Column(String)
    text = Column(String)
    hypothesis = Column(String)
    avg_length = Column(Float)
    repetition = Column(Float)
    entropy = Column(Float)
    zipf = Column(Float)
    ttr = Column(Float)
    bigram_conc = Column(Float)
    trigram_conc = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --------------------
# METRICS
# --------------------
def entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(
        -sum((c / total) * math.log2(c / total) for c in freq.values()), 3
    )

def zipf_score(words):
    if len(words) < 10:
        return None
    freq = Counter(w.lower() for w in words)
    counts = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)
    return round(np.corrcoef(np.log(ranks), np.log(counts))[0, 1], 3)

def ttr_score(words):
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)

def ngram_concentration(text: str, n: int, top_k: int = 5):
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    freq = Counter(ngrams)
    total = sum(freq.values())
    top = sum(c for _, c in freq.most_common(top_k))
    return round(top / total, 3)

# --------------------
# CONFIDENCE SYSTEM
# --------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(x, b))

PRESETS = {
    "general": {
        "entropy": 0.25,
        "zipf": 0.20,
        "ttr": 0.20,
        "repetition": 0.20,
        "ngram": 0.15
    },
    "forense": {
        "entropy": 0.30,
        "zipf": 0.25,
        "ttr": 0.15,
        "repetition": 0.15,
        "ngram": 0.15
    },
    "ia": {
        "entropy": 0.20,
        "zipf": 0.15,
        "ttr": 0.30,
        "repetition": 0.20,
        "ngram": 0.15
    }
}

def confidence_score(ent, zipf, ttr, repetition, bigram_c, trigram_c, preset="general"):
    p = PRESETS.get(preset, PRESETS["general"])

    entropy_s = clamp((ent - 2.5) / (4.8 - 2.5))
    zipf_s = clamp((abs(zipf) - 0.6) / (1.0 - 0.6)) if zipf else 0.0
    ttr_s = clamp((ttr - 0.2) / (0.6 - 0.2))
    repetition_s = clamp(1 - repetition)
    ngram_s = clamp(1 - ((bigram_c + trigram_c) / 2))

    score = (
        p["entropy"] * entropy_s +
        p["zipf"] * zipf_s +
        p["ttr"] * ttr_s +
        p["repetition"] * repetition_s +
        p["ngram"] * ngram_s
    )

    if zipf is None:
        score -= 0.05
    if ent > 5.2 and ngram_s > 0.8:
        score -= 0.1
    if ttr < 0.2:
        score -= 0.1

    return round(clamp(score), 3)

def confidence_label(score):
    if score >= 0.85:
        return "Muy alta confianza"
    if score >= 0.70:
        return "Alta confianza"
    if score >= 0.50:
        return "Confianza media"
    if score >= 0.30:
        return "Baja confianza"
    return "No concluyente"

def confidence_explanation(ent, zipf, ttr, repetition, bigram_c, trigram_c):
    exp = []

    if 3.2 <= ent <= 4.8:
        exp.append("La entropía se sitúa en un rango típico del lenguaje estructurado.")
    elif ent > 5.0:
        exp.append("La entropía es elevada, compatible con estructura no lingüística.")
    else:
        exp.append("La entropía es baja, indicando repetición o simplificación.")

    if zipf and abs(zipf) > 0.85:
        exp.append("La distribución léxica sigue la ley de Zipf.")
    else:
        exp.append("La distribución léxica se desvía del patrón Zipf.")

    if ttr < 0.25:
        exp.append("La variedad léxica es reducida.")
    else:
        exp.append("La variedad léxica es adecuada.")

    if (bigram_c + trigram_c) / 2 > 0.2:
        exp.append("Existe concentración significativa de n‑gramas.")

    return exp

# --------------------
# ANALYSIS CORE
# --------------------
def analyze_and_store(text: str, source: str, preset="general"):
    words = text.split()

    avg = round(sum(len(w) for w in words) / len(words), 2) if words else 0
    rep = round(1 - len(set(words)) / len(words), 2) if words else 0
    ent = entropy("".join(words))
    zipf = zipf_score(words)
    ttr = ttr_score(words)
    bigram_c = ngram_concentration(text.lower(), 2)
    trigram_c = ngram_concentration(text.lower(), 3)

    hypothesis = "Estructura mixta"
    if zipf and zipf < -0.9 and ttr > 0.4 and bigram_c > 0.15:
        hypothesis = "Lenguaje natural probable"
    elif ent > 4.5 and bigram_c < 0.08:
        hypothesis = "Texto no lingüístico / cifrado"
    elif ttr < 0.25:
        hypothesis = "Texto repetitivo o simplificado"

    conf = confidence_score(ent, zipf, ttr, rep, bigram_c, trigram_c, preset)

    db = SessionLocal()
    entry = Analysis(
        source=source,
        text=text,
        hypothesis=hypothesis,
        avg_length=avg,
        repetition=rep,
        entropy=ent,
        zipf=zipf,
        ttr=ttr,
        bigram_conc=bigram_c,
        trigram_conc=trigram_c,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()

    return {
        "id": entry.id,
        "texto": text,
        "hipotesis": hypothesis,
        "preset": preset,
        "confidence": conf,
        "confidence_label": confidence_label(conf),
        "confidence_explanation": confidence_explanation(
            ent, zipf, ttr, rep, bigram_c, trigram_c
        ),
        "longitud_media": avg,
        "repeticion": rep,
        "entropia": ent,
        "zipf": zipf,
        "ttr": ttr,
        "bigram_conc": bigram_c,
        "trigram_conc": trigram_c,
    }

# --------------------
# ENDPOINTS
# --------------------
@app.post("/analyze")
def analyze(data: dict):
    preset = data.get("preset", "general")
    return analyze_and_store(data["text"], source="text", preset=preset)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_and_store(text, source="ocr")

@app.get("/history")
def history():
    db = SessionLocal()
    records = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(20).all()
    db.close()
    return [
        {
            "id": r.id,
            "hypothesis": r.hypothesis,
            "entropy": r.entropy,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]

@app.get("/analysis/{id}")
def get_analysis(id: int):
    db = SessionLocal()
    r = db.query(Analysis).filter(Analysis.id == id).first()
    db.close()
    return {
        "texto": r.text,
        "hipotesis": r.hypothesis,
        "longitud_media": r.avg_length,
        "repeticion": r.repetition,
        "entropia": r.entropy,
        "zipf": r.zipf,
        "ttr": r.ttr,
        "bigram_conc": r.bigram_conc,
        "trigram_conc": r.trigram_conc,
    }
