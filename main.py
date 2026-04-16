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
    avg_length = Column(Float)
    repetition = Column(Float)
    entropy = Column(Float)
    zipf = Column(Float)
    ttr = Column(Float)
    bigram_conc = Column(Float)
    trigram_conc = Column(Float)
    lfv_state = Column(String)
    lfv_sequence = Column(String)
    lfv_translation = Column(String)
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
# METRICS (RSDJ)
# =========================

def entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(-sum((c/total) * math.log2(c/total) for c in freq.values()), 3)

def zipf_score(words):
    if len(words) < 10:
        return None
    freq = Counter(w.lower() for w in words)
    counts = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)
    return round(np.corrcoef(np.log(ranks), np.log(counts))[0, 1], 3)

def ttr_score(words):
    return round(len(set(words)) / len(words), 3) if words else 0.0

def ngram_concentration(text: str, n: int, top_k: int = 5):
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    freq = Counter(ngrams)
    total = sum(freq.values())
    top = sum(c for _, c in freq.most_common(top_k))
    return round(top / total, 3)

# =========================
# LFV FASE 1–2–3
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

def lfv_phase_3_translation(seq):
    if not seq:
        return "No se detecta una dinámica estructural clara."
    if "expansion" in seq and "densificacion" in seq:
        return "El texto se desarrolla y posteriormente se condensa estructuralmente."
    if all(s == "expansion" for s in seq):
        return "El texto mantiene una dinámica de expansión continua."
    if all(s == "densificacion" for s in seq):
        return "El texto presenta una estructura repetitiva y condensada."
    if "estabilizacion" in seq:
        return "El texto tiende a estabilizar su estructura."
    return "El texto combina dinámicas estructurales diversas."

# =========================
# ANALYSIS CORE
# =========================

def analyze_and_store(text: str, source: str, user_id=None):
    words = text.split()

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

    lfv1 = lfv_phase_1(words)
    lfv2 = lfv_phase_2(words)
    lfv3 = lfv_phase_3_translation(lfv2)

    db = SessionLocal()
    entry = Analysis(
        source=source,
        text=text,
        hypothesis=hypothesis,
        entropy=ent,
        zipf=zipf,
        ttr=ttr,
        bigram_conc=bigram_c,
        trigram_conc=trigram_c,
        lfv_state=lfv1,
        lfv_sequence=",".join(lfv2),
        lfv_translation=lfv3,
        user_id=user_id
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()

    return {
        "id": entry.id,
        "hipotesis": hypothesis,
        "entropia": ent,
        "zipf": zipf,
        "ttr": ttr,
        "lfv_fase_1": lfv1,
        "lfv_fase_2": lfv2,
        "lfv_fase_3": lfv3
    }

# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"status": "ok", "service": "RSDJ Backend"}

@app.post("/users")
def create_user(data: dict):
    db = SessionLocal()
    u = User(username=data["username"])
    db.add(u)
    db.commit()
    db.refresh(u)
    db.close()
    return {"id": u.id, "username": u.username}

@app.post("/analyze")
def analyze(data: dict):
    return analyze_and_store(data["text"], "text", data.get("user_id"))

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_and_store(text, "ocr")

@app.post("/like")
def like(data: dict):
    db = SessionLocal()
    db.add(Like(analysis_id=data["analysis_id"], user_id=data.get("user_id")))
    db.commit()
    db.close()
    return {"status": "liked"}

@app.post("/comment")
def comment(data: dict):
    db = SessionLocal()
    db.add(Comment(
        analysis_id=data["analysis_id"],
        user_id=data.get("user_id"),
        content=data["content"]
    ))
    db.commit()
    db.close()
    return {"status": "commented"}

@app.get("/analysis/{id}")
def get_analysis(id: int):
    db = SessionLocal()
    r = db.query(Analysis).filter(Analysis.id == id).first()
    likes = db.query(Like).filter(Like.analysis_id == id).count()
    comments = db.query(Comment).filter(Comment.analysis_id == id).all()
    db.close()

    return {
        "id": r.id,
        "texto": r.text,
        "hipotesis": r.hypothesis,
        "entropia": r.entropy,
        "lfv_fase_3": r.lfv_translation,
        "likes": likes,
        "comments": [
            {"content": c.content, "user_id": c.user_id}
            for c in comments
        ]
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
            "lfv": r.lfv_state,
            "likes": 0
        }
        for r in rows
    ]
