import os
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Local default works when Postgres runs in Docker (port 5432 exposed)
# Docker default (DATABASE_URL env var) uses the service name "db"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpassword@localhost:5432/financial_rag"
)

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


def get_db():
    """FastAPI dependency — one DB session per request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(retries=5, delay=3):
    """Creates tables if they don't exist. Retries — Postgres may
    still be starting up when the API container boots."""
    for attempt in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables ready")
            return
        except Exception as e:
            print(f"  DB not ready (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to database after retries")