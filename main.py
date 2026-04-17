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

# =========================
# APP
# =========================

app = FastAPI(title="RSDJ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DATABASE
# =========================

DATABASE_URL = "sqlite:///./rsdj.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================
# MODELS
# =========================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    text = Column(String)
    hypothesis = Column(String)
    entropy = Column(Float)
    zipf = Column(Float)
    ttr = Column(Float)
    bigram_conc = Column(Float)
    trigram_conc = Column(Float)
    lfv_state = Column(String)
    lfv_sequence = Column(String)
    lfv_translation = Column(String)
    lfv_semantic = Column(String)
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer)
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer)
    user_id = Column(Integer)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Reaction(Base):
    __tablename__ = "reactions"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer)
    type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# =========================
# METRICS
# =========================

def entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in freq.values()), 3)

def zipf_score(words):
    if len(words) < 10:
        return None
    freq = Counter(w.lower() for w in words)
    counts = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)
    return round(np.corrcoef(np.log(ranks), np.log(counts))[0, 1], 3)

def ttr_score(words):
    return round(len(set(words)) / len(words), 3) if words else 0.0

def ngram_concentration(text, n, top_k=5):
    if len(text) < n:
        return 0.0
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    freq = Counter(grams)
    total = sum(freq.values())
    top = sum(c for _, c in freq.most_common(top_k))
    return round(top / total, 3)

# =========================
# LFV
# =========================

def lfv_phase_1(words):
    if not words:
        return "indeterminado"
    ttr = len(set(words)) / len(words)
    if ttr < 0.3:
        return "densificacion"
    if ttr > 0.6:
        return "expansion"
    return "estabilizacion"

def lfv_phase_2(words, window=20):
    return [lfv_phase_1(words[i:i+window]) for i in range(0, len(words), window)]

def lfv_phase_3(seq):
    if not seq:
        return "Sin dinámica estructural clara."
    if "expansion" in seq and "densificacion" in seq:
        return "El texto se desarrolla y luego se condensa."
    if all(s == "expansion" for s in seq):
        return "El texto presenta expansión continua."
    if all(s == "densificacion" for s in seq):
        return "El texto es altamente repetitivo."
    return "El texto combina varias dinámicas."

def lfv_phase_4(seq):
    if not seq:
        return "Interpretación no disponible."
    parts = []
    if seq[0] == "expansion":
        parts.append("El texto introduce información nueva.")
    if "densificacion" in seq:
        parts.append("Se observan patrones repetidos.")
    if "estabilizacion" in seq:
        parts.append("La estructura se estabiliza.")
    return " ".join(parts)

# =========================
# CORE
# =========================

def analyze_and_store(text, source, user_id=None):
    db = SessionLocal()
    words = text.split()

    analysis = Analysis(
        source=source,
        text=text,
        hypothesis="El texto presenta estructura funcional no aleatoria.",
        entropy=entropy(text),
        zipf=zipf_score(words),
        ttr=ttr_score(words),
        bigram_conc=ngram_concentration(text, 2),
        trigram_conc=ngram_concentration(text, 3),
        lfv_state=lfv_phase_1(words),
        lfv_sequence=str(lfv_phase_2(words)),
        lfv_translation=lfv_phase_3(lfv_phase_2(words)),
        lfv_semantic=lfv_phase_4(lfv_phase_2(words)),
        user_id=user_id
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    db.close()

    return {
        "id": analysis.id,
        "entropy": analysis.entropy,
        "zipf": analysis.zipf,
        "ttr": analysis.ttr,
        "bigram": analysis.bigram_conc,
        "trigram": analysis.trigram_conc,
        "lfv_fase_1": analysis.lfv_state,
        "lfv_fase_2": eval(analysis.lfv_sequence),
        "lfv_fase_3": analysis.lfv_translation,
        "lfv_fase_4": analysis.lfv_semantic,
        "hipotesis": analysis.hypothesis
    }

# =========================
# ENDPOINTS
# =========================

@app.post("/analyze")
def analyze(data: dict):
    return analyze_and_store(data["text"], "text", data.get("user_id"))

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_and_store(text, "ocr")

@app.post("/comment")
def comment(data: dict):
    db = SessionLocal()
    c = Comment(
        analysis_id=data["analysis_id"],
        user_id=data.get("user_id"),
        content=data["content"]
    )
    db.add(c)
    db.commit()
    db.close()
    return {"status": "ok"}

@app.post("/compare_semantic")
def compare_semantic(data: dict):
    a = analyze_and_store(data["textA"], "compare")
    b = analyze_and_store(data["textB"], "compare")
    return {"comparacion_lfv": lfv_phase_5(a["lfv_fase_2"], b["lfv_fase_2"])}

@app.get("/analysis/{id}")
def get_analysis(id: int):
    db = SessionLocal()
    r = db.query(Analysis).filter(Analysis.id == id).first()
    db.close()
    return {
        "id": r.id,
        "texto": r.text,
        "hipotesis": r.hypothesis,
        "lfv_fase_3": r.lfv_translation,
        "lfv_fase_4": r.lfv_semantic
    }

@app.get("/feed")
def feed(limit: int = 20):
    db = SessionLocal()
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
    db.close()
    return [
        {
            "id": r.id,
            "texto_preview": r.text[:200] + ("…" if len(r.text) > 200 else ""),
            "hipotesis": r.hypothesis,
            "lfv_fase_4": r.lfv_semantic
        }
        for r in rows
    ]

@app.post("/react")
def react(data: dict):
    db = SessionLocal()
    r = Reaction(
        analysis_id=data["analysis_id"],
        type=data["type"]
    )
    db.add(r)
    db.commit()
    db.close()
    return {"status": "ok"}

@app.get("/reactions/{analysis_id}")
def get_reactions(analysis_id: int):
    db = SessionLocal()
    rows = db.query(Reaction).filter(Reaction.analysis_id == analysis_id).all()
    db.close()
    counts = {}
    for r in rows:
        counts[r.type] = counts.get(r.type, 0) + 1
    return counts

@app.get("/user/{user_id}/analyses")
def user_analyses(user_id: int):
    db = SessionLocal()
    rows = (
        db.query(Analysis)
        .filter(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    db.close()
    return [
        {
            "id": r.id,
            "texto_preview": r.text[:200] + ("…" if len(r.text) > 200 else ""),
            "hipotesis": r.hypothesis,
            "lfv_fase_4": r.lfv_semantic,
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]
