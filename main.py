from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from collections import Counter
from datetime import datetime
from typing import Optional
import math
import io

from PIL import Image
import pytesseract

# --------------------
# APP
# --------------------
app = FastAPI(
    title="RSDJ Backend",
    version="1.0.0",
)

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

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)  # "text" | "ocr"
    text = Column(String, nullable=False)
    hypothesis = Column(String, nullable=False)
    avg_length = Column(Float, nullable=False)
    repetition = Column(Float, nullable=False)
    entropy = Column(Float, nullable=False)
    particles = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------
# ANALYSIS LOGIC
# --------------------
def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return round(
        -sum((c / total) * math.log2(c / total) for c in freq.values()),
        3,
    )


def analyze_text(
    text: str,
    source: str,
    db: Optional[Session] = None,
):
    words = text.split()

    avg = round(sum(len(w) for w in words) / len(words), 2) if words else 0.0
    rep = round(1 - len(set(words)) / len(words), 2) if words else 0.0
    ent = calculate_entropy("".join(words))

    freq = Counter(w.lower() for w in words)
    particles = [w for w, c in freq.items() if len(w) <= 3 and c > 1][:8]

    hypothesis = "Estructura mixta"
    if avg < 4 and rep < 0.3:
        hypothesis = "Lengua aislante probable"
    elif avg > 7 and rep > 0.4:
        hypothesis = "Lengua aglutinante probable"
    elif ent > 4.5:
        hypothesis = "Texto cifrado / no lingüístico"

    result = {
        "texto": text,
        "hipotesis": hypothesis,
        "longitud_media": avg,
        "repeticion": rep,
        "entropia": ent,
        "particulas": particles,
    }

    if db is not None:
        entry = Analysis(
            source=source,
            text=text,
            hypothesis=hypothesis,
            avg_length=avg,
            repetition=rep,
            entropy=ent,
            particles=",".join(particles),
        )
        db.add(entry)
        db.commit()

    return result


# --------------------
# ENDPOINTS
# --------------------
@app.post("/analyze")
def analyze(payload: dict, db: Session = Depends(get_db)):
    text = payload.get("text", "")
    return analyze_text(text=text, source="text", db=db)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang="spa+eng")
    return analyze_text(text=text, source="ocr", db=db)


@app.get("/history")
def history(db: Session = Depends(get_db)):
    records = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(20)
        .all()
    )

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
