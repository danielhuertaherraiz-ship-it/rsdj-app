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
    log_ranks = np.log(ranks)
    log_counts = np.log(counts)
    return round(np.corrcoef(log_ranks, log_counts)[0, 1], 3)

def ttr_score(words):
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)

# --------------------
# ANALYSIS CORE
# --------------------
def analyze_and_store(text: str, source: str):
    words = text.split()

    avg = round(sum(len(w) for w in words) / len(words), 2) if words else 0
    rep = round(1 - len(set(words)) / len(words), 2) if words else 0
    ent = entropy("".join(words))
    zipf = zipf_score(words)
    ttr = ttr_score(words)

    hypothesis = "Estructura mixta"
    if zipf and zipf < -0.9 and ttr > 0.4:
        hypothesis = "Lenguaje natural probable"
    elif ent > 4.5:
        hypothesis = "Texto no lingüístico / cifrado"
    elif ttr < 0.25:
        hypothesis = "Texto repetitivo o simplificado"

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
    )
    db.add(entry)
    db.commit()
    db.close()

    return {
        "texto": text,
        "hipotesis": hypothesis,
        "longitud_media": avg,
        "repeticion": rep,
        "entropia": ent,
        "zipf": zipf,
        "ttr": ttr,
    }

# --------------------
# ENDPOINTS
# --------------------
@app.post("/analyze")
def analyze(data: dict):
    return analyze_and_store(data["text"], source="text")

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
    }
