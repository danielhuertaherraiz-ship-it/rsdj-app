from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from collections import Counter
from datetime import datetime
import math, io, re
import numpy as np
from PIL import Image
import pytesseract
from pydantic import BaseModel
from typing import List, Dict

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
# ANALYSIS CORE
# --------------------
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
        "longitud_media": avg,
        "repeticion": rep,
        "entropia": ent,
        "zipf": zipf,
        "ttr": ttr,
        "bigram_conc": bigram_c,
        "trigram_conc": trigram_c,
    }

# --------------------
# ENDPOINTS RSDJ
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
        "bigram_conc": r.bigram_conc,
        "trigram_conc": r.trigram_conc,
    }

# ============================================================
# ======================= LFV MODULE =========================
# ============================================================

# --------------------
# LFV MODELS
# --------------------
class LFVRequest(BaseModel):
    text: str

class LFVSegment(BaseModel):
    token: str
    roles: List[str]
    function: str

class LFVResponse(BaseModel):
    segments: List[LFVSegment]
    synthesis: Dict[str, str]

# --------------------
# LFV DICTIONARIES
# --------------------
FORMS = {"cho", "chol", "chor", "che", "chey", "cheor", "cthol", "chody"}
PROCESSES = {"aiin", "daiin", "otaiin", "kaiin", "oiin", "saiin"}
OPERATORS = {"qo", "qok", "qot", "ok", "ot"}
MODIFIERS_PREFIX = {"sh", "ch", "kch", "cph", "pch", "tch"}
CLOSURES = {"dy", "dal", "dam", "ar", "or", "am", "ody", "dain"}

# --------------------
# LFV FUNCTIONS
# --------------------
def detect_roles(token: str) -> List[str]:
    roles = []
    base = re.sub(r"[^a-z]", "", token.lower())

    for m in MODIFIERS_PREFIX:
        if base.startswith(m):
            roles.append("M")
    for o in OPERATORS:
        if base.startswith(o):
            roles.append("O")
    for f in FORMS:
        if f in base:
            roles.append("F")
    for p in PROCESSES:
        if p in base:
            roles.append("P")
    for c in CLOSURES:
        if base.endswith(c):
            roles.append("C")

    return list(dict.fromkeys(roles))

def functional_translation(roles: List[str]) -> str:
    if set(["F", "P", "C"]).issubset(roles):
        return "ciclo funcional completo"
    if roles == ["F"]:
        return "forma estable"
    if roles == ["P"]:
        return "proceso activo"
    if roles == ["C"]:
        return "cierre estructural"
    if "F" in roles and "P" in roles:
        return "expansión estructural"
    if "P" in roles and "C" in roles:
        return "proceso con cierre"
    if "O" in roles:
        return "regulación direccional"
    return "variación estructural"

def synthesize(segments: List[LFVSegment]) -> Dict[str, str]:
    roles = [r for s in segments for r in s.roles]
    if roles.count("F") > roles.count("C"):
        return {
            "patron": "forma → proceso",
            "estado": "expansión",
            "rol": "fase constructiva"
        }
    return {
        "patron": "proceso → cierre",
        "estado": "consolidación",
        "rol": "fase estabilizada"
    }

# --------------------
# LFV ENDPOINT
# --------------------
@app.post("/lfv/analyze", response_model=LFVResponse)
def lfv_analyze(data: LFVRequest):
    tokens = re.split(r"[.\s]+", data.text.strip())
    segments = []

    for t in tokens:
        if not t:
            continue
        r = detect_roles(t)
        segments.append(
            LFVSegment(
                token=t,
                roles=r,
                function=functional_translation(r)
            )
        )

    return LFVResponse(
        segments=segments,
        synthesis=synthesize(segments)
    )
