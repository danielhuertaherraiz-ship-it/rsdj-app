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
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
    grams = [text[i:i+n] for i in range(len(text) - n + 1)]
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

def lfv_phase_5(a, b):
    if not a or not b:
        return "No hay datos suficientes para comparar."
    diff = []
    if a.count("expansion") != b.count("expansion"):
        diff.append("Diferente grado de expansión.")
    if a.count("densificacion") != b.count("densificacion"):
        diff.append("Diferente nivel de repetición.")
    if not diff:
        diff.append("Estructura muy similar.")
    return " ".join(diff)

# =========================
# CORE
# =========================

def analyze_and_store(text, source, user_id=None):
    words = text.split()

    ent = entropy("".join(words))
    zipf = zipf_score(words)
    ttr = ttr_score(words)
    bg = ngram_concentration(text.lower(), 2)
    tg = ngram_concentration(text.lower(), 3)

    hypothesis = "Estructura mixta"
    if zipf and zipf < -0.9 and ttr > 0.4 and bg > 0.15:
        hypothesis = "Lenguaje natural probable"
    elif ent > 4.5 and bg < 0.08:
        hypothesis = "Texto no lingüístico"
    elif ttr < 0.25:
        hypothesis = "Texto repetitivo"

    lfv1 = lfv_phase_1(words)
    lfv2 = lfv_phase_2(words)
    lfv3 = lfv_phase_3(lfv2)
    lfv4 = lfv_phase_4(lfv2)

    db = SessionLocal()
    a = Analysis(
        source=source,
        text=text,
        hypothesis=hypothesis,
        entropy=ent,
        zipf=zipf,
        ttr=ttr,
        bigram_conc=bg,
        trigram_conc=tg,
        lfv_state=lfv1,
        lfv_sequence=",".join(lfv2),
        lfv_translation=lfv3,
        lfv_semantic=lfv4,
        user_id=user_id
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    db.close()

    return {
        "id": a.id,
        "hipotesis": hypothesis,
        "entropia": ent,
        "lfv_fase_1": lfv1,
        "lfv_fase_2": lfv2,
        "lfv_fase_3": lfv3,
        "lfv_fase_4": lfv4
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
    return [{
        "id": r.id,
        "texto_preview": r.text[:200] + ("…" if len(r.text) > 200 else ""),
        "hipotesis": r.hypothesis,
        "lfv_fase_4": r.lfv_semantic
    } for r in rows]
