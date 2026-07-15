import os
from pathlib import Path

# Points to the folder ABOVE api/ — where chroma_db lives
BASE_DIR = Path(__file__).parent.parent

CHROMA_PATH     = os.getenv("CHROMA_PATH",     str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL",       "phi4-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_TEMPERATURE = 0
LLM_NUM_CTX     = 4096
LLM_NUM_PREDICT = 400

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))  # candidates fed INTO reranker