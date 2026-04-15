from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from collections import Counter
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import math, io
import numpy as np
from PIL import Image
import pytesseract

# ============================================================
# APP
# ============================================================

app = FastAPI(title="RSDJ Backend + LFV")

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
# LFV CORE (INTEGRADO)
# ============================================================

class LFVState(Enum):
    NACIMIENTO = "nacimiento"
    CONSOLIDACION = "consolidacion"
    EXPANSION = "expansion"
    DENSIFICACION = "densificacion"
    CIERRE = "cierre"

class LFVUnit(Enum):
    F = "forma"
    P = "proceso"
    O = "operador"
    M = "modificador"
    C = "cierre"

@dataclass
class LFVEvent:
    unidad: str
    token: str
    efecto: str

@dataclass
class LFVSegment:
    estado: str
    eventos: List[LFVEvent]
    efecto_estructural: str

class LFVEngine:
    def __init__(self):
        self.estado = LFVState.NACIMIENTO
        self.segmentos: List[LFVSegment] = []

    def detectar_unidad(self, token: str) -> LFVUnit:
        if token.endswith(("dy", "dain", "dam", "dal", "dor", "rodg")):
            return LFVUnit.C
        if token.startswith(("qo", "qok", "qot")):
            return LFVUnit.O
        if "aiin" in token or "oiin" in token:
            return LFVUnit.P
        if token.startswith(("sh", "ch", "kch", "cph", "tch", "ckh")):
            return LFVUnit.M
        return LFVUnit.F

    def transicion(self, unidad: LFVUnit):
        if unidad == LFVUnit.F and self.estado == LFVState.NACIMIENTO:
            self.estado = LFVState.CONSOLIDACION
        elif unidad == LFVUnit.P:
            self.estado = LFVState.EXPANSION
        elif unidad == LFVUnit.M:
            self.estado = LFVState.DENSIFICACION
        elif unidad == LFVUnit.C:
            self.estado = LFVState.CIERRE

    def analizar(self, texto: str) -> Dict:
        self.estado = LFVState.NACIMIENTO
        self.segmentos = []

        for seg in texto.replace("\n", " ").split("."):
            tokens = seg.strip().split()
            if not tokens:
                continue

            eventos = []
            for t in tokens:
                u = self.detectar_unidad(t)
                self.transicion(u)
                eventos.append(
                    LFVEvent(
                        unidad=u.value,
                        token=t,
                        efecto=f"{u.value} → {self.estado.value}"
                    )
                )

            self.segmentos.append(
                LFVSegment(
                    estado=self.estado.value,
                    eventos=eventos,
                    efecto_estructural=f"Estado LFV fijado en {self.estado.value}"
                )
            )

        return {
            "estado_inicial": LFVState.NACIMIENTO.value,
            "estado_final": self.estado.value,
            "segmentos": [
                {
                    "estado": s.estado,
                    "eventos": [e.__dict__ for e in s.eventos],
                    "efecto_estructural": s.efecto_estructural
                } for s in self.segmentos
            ],
            "sintesis_LFV": (
                "Ciclo LFV completo con cierre."
                if self.estado == LFVState.CIERRE
                else "Ciclo LFV abierto."
            )
        }

# ============================================================
# ANALYSIS CORE (RSDJ + LFV)
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

    lfv_result = LFVEngine().analizar(text)

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
        "rsdj": {
            "hipotesis": hypothesis,
            "entropia": ent,
            "zipf": zipf,
            "ttr": ttr,
            "bigram": bigram_c,
            "trigram": trigram_c,
        },
        "lfv": lfv_result
    }

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/analyze")
def analyze(data: dict):
    return analyze_and_store(data["text"], source="text")

@app.post("/lfv/analyze")
def analyze_lfv(data: dict):
    return LFVEngine().analizar(data["text"])

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_and_store(text, source="ocr")
