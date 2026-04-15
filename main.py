from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from collections import Counter
from datetime import datetime
import math, io

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
# DATABASE (SQLite)
# --------------------
DATABASE_URL = "sqlite:///./rsdj.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    source = Column(String)        # "text" | "ocr"
    text = Column(String)
    hypothesis = Column(String)
    avg_length = Column(Float)
    repetition = Column(Float)
    entropy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --------------------
# ANALYSIS LOGIC
# --------------------
def entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(
        -sum((c / total) * math.log2(c / total) for c in freq.values()), 3
    )

def analyze_and_store(text: str, source: str):
    words = text.split()

    avg = round(sum(len(w) for w in words) / len(words), 2) if words else 0
    rep = round(1 - len(set(words)) / len(words), 2) if words else 0
    ent = entropy("".join(words))

    hypothesis = "Estructura mixta"
    if avg < 4 and rep < 0.3:
        hypothesis = "Lengua aislante probable"
    elif avg > 7 and rep > 0.4:
        hypothesis = "Lengua aglutinante probable"
    elif ent > 4.5:
        hypothesis = "Texto cifrado / no lingüístico"

    db = SessionLocal()
    entry = Analysis(
        source=source,
        text=text,
        hypothesis=hypothesis,
        avg_length=avg,
        repetition=rep,
        entropy=ent,
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
    records = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(20)
        .all()
    )
    db.close()

    return [
        {
            "id": r.id,
            "source": r.source,
            "hypothesis": r.hypothesis,
            "entropy": r.entropy,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
