import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://pixelwise_user:dein_sicheres_passwort@localhost:5432/pixelwise")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionModel(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
