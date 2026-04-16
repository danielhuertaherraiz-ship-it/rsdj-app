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

# ============================================================
# APP
# ============================================================

app = FastAPI(title="RSDJ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE
# ============================================================

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
    lfv_state = Column(String)
    lfv_sequence = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============================================================
# METRICS (RSDJ)
# ============================================================

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

# ============================================================
# LFV — FASE 1
# ============================================================

def lfv_phase_1(words):
    if not words:
        return "indeterminado"
    ttr = len(set(words)) / len(words)
    if ttr < 0.3:
        return "densificacion"
    if ttr > 0.6:
        return "expansion"
    return "estabilizacion"

# ============================================================
# LFV — FASE 2 (SECUENCIAL)
# ============================================================

def lfv_phase_2(words, window=20):
    seq = []
    for i in range(0, len(words), window):
        seq.append(lfv_phase_1(words[i:i+window]))
    return seq

# ============================================================
# ANALYSIS CORE
# ============================================================

def analyze_and_store(text: str, source: str):
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

    lfv_state = lfv_phase_1(words)
    lfv_seq = lfv_phase_2(words)

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
        lfv_state=lfv_state,
        lfv_sequence=",".join(lfv_seq)
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()

    return {
        "id": entry.id,
        "texto": text,
        "hipotesis": hypothesis,
        "entropia": ent,
        "zipf": zipf,
        "ttr": ttr,
        "bigram_conc": bigram_c,
        "trigram_conc": trigram_c,
        "lfv_fase_1": lfv_state,
        "lfv_fase_2": lfv_seq
    }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RSDJ Backend",
        "endpoints": ["/analyze", "/ocr", "/compare", "/analysis/{id}", "/feed"]
    }

@app.post("/analyze")
def analyze(data: dict):
    return analyze_and_store(data["text"], source="text")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_and_store(text, source="ocr")

@app.post("/compare")
def compare(data: dict):
    a = analyze_and_store(data["textA"], source="compare")
    b = analyze_and_store(data["textB"], source="compare")

    distance = round(
        math.sqrt(
            (a["entropia"] - b["entropia"])**2 +
            (a["ttr"] - b["ttr"])**2
        ), 3
    )

    return {"distancia_estructural": distance, "A": a, "B": b}

@app.get("/analysis/{id}")
def get_analysis(id: int):
    db = SessionLocal()
    r = db.query(Analysis).filter(Analysis.id == id).first()
    db.close()
    return {
        "id": r.id,
        "texto": r.text,
        "hipotesis": r.hypothesis,
        "entropia": r.entropy,
        "zipf": r.zipf,
        "ttr": r.ttr,
        "lfv_fase_1": r.lfv_state,
        "lfv_fase_2": r.lfv_sequence.split(",") if r.lfv_sequence else [],
        "created_at": r.created_at.isoformat()
    }

# ============================================================
# COMUNIDAD — FASE 0 (FEED PÚBLICO)
# ============================================================

@app.get("/feed")
def feed(limit: int = 20):
    db = SessionLocal()
    rows = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .all()
    )
    db.close()

    return [
        {
            "id": r.id,
            "texto_preview": r.text[:240] + ("…" if len(r.text) > 240 else ""),
            "hipotesis": r.hypothesis,
            "entropia": r.entropy,
            "ttr": r.ttr,
            "lfv_fase_1": r.lfv_state,
            "lfv_fase_2": r.lfv_sequence.split(",") if r.lfv_sequence else [],
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]
