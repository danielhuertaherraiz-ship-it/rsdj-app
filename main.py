from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from collections import Counter
from datetime import datetime
from enum import Enum
from typing import List, Dict
import math, io
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
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    text = Column(String)
    entropy = Column(Float)
    lfv_state = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============================================================
# MÉTRICAS RSDJ
# ============================================================

def entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(
        -sum((c / total) * math.log2(c / total) for c in freq.values()),
        3
    )

# ============================================================
# LFV CORE
# ============================================================

class LFVState(Enum):
    NACIMIENTO = "nacimiento"
    EXPANSION = "expansion"
    DENSIFICACION = "densificacion"
    CIERRE = "cierre"

class LFVEngine:
    def __init__(self):
        self.estado = LFVState.NACIMIENTO
        self.eventos = []

    def analizar(self, text: str) -> Dict:
        self.estado = LFVState.NACIMIENTO
        self.eventos = []

        for token in text.split():
            if token.endswith(("dy", "dain", "dam", "dal", "dor")):
                self.estado = LFVState.CIERRE
            elif "aiin" in token:
                self.estado = LFVState.EXPANSION
            elif token.startswith(("sh", "ch", "kch")):
                self.estado = LFVState.DENSIFICACION

            self.eventos.append({
                "token": token,
                "estado": self.estado.value
            })

        return {
            "estado_final": self.estado.value,
            "eventos": self.eventos
        }

    def traducir(self) -> str:
        return {
            "nacimiento":
                "La forma inicia su configuración estructural.",
            "expansion":
                "El sistema entra en una fase activa de desarrollo.",
            "densificacion":
                "La estructura se condensa mediante modificaciones internas.",
            "cierre":
                "El proceso se completa y la estructura queda fijada."
        }[self.estado.value]

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/analyze")
def analyze(data: dict):
    text = data.get("text", "")
    ent = entropy(text)

    lfv = LFVEngine()
    lfv_result = lfv.analizar(text)

    db = SessionLocal()
    db.add(Analysis(
        source="text",
        text=text,
        entropy=ent,
        lfv_state=lfv_result["estado_final"]
    ))
    db.commit()
    db.close()

    return {
        "entropia": ent,
        "lfv": lfv_result,
        "traduccion_lfv": lfv.traducir()
    }

@app.post("/lfv/translate")
def lfv_translate(data: dict):
    engine = LFVEngine()
    analisis = engine.analizar(data.get("text", ""))

    return {
        "estado": analisis["estado_final"],
        "eventos": analisis["eventos"],
        "traduccion": engine.traducir()
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = pytesseract.image_to_string(image)
    return {"text": text}

@app.get("/history")
def history():
    db = SessionLocal()
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(20).all()
    db.close()
    return [
        {
            "id": r.id,
            "entropy": r.entropy,
            "lfv_state": r.lfv_state,
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]
