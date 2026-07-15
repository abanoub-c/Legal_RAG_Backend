"""
pinecone_hybrid_retriever.py
Drop this into your api/ folder.

Two-part hybrid search against a Pinecone serverless index (metric=dotproduct):
  - Dense: voyage-4 via the official voyageai SDK (matches whatever
    pinecone_index_setup.py used to populate the index -- if you change
    VOYAGE_EMBED_MODEL here, it must also match the model the index was
    actually built with, or you'll get a dimension mismatch or silently
    wrong results)
  - Sparse: pinecone-text's BM25Encoder, loaded from params fitted once at
    indexing time -- pure statistical computation, no API calls

RBAC: one Pinecone namespace per practice-area category.
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import voyageai

load_dotenv(find_dotenv())

VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# NOTE: this default must match pinecone_index_setup.py's default exactly.
# The old draft of this file defaulted to "legal-contract-chunks" (no
# "-voyage" suffix) -- that's not where your data actually lives.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-contract-chunks-voyage")
EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4")
BM25_PARAMS_PATH = os.getenv("BM25_PARAMS_PATH", "data/bm25_params.json")

ALL_PRACTICE_AREAS = ["ip_tech", "general_commercial", "corporate_ma", "employment_hr"]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Loaded once at import time. Must be the exact bm25_params.json produced
# by the indexing run -- a BM25 encoder fit on a different corpus will
# silently produce lower-quality sparse vectors, not an error.
bm25 = BM25Encoder()
bm25.load(BM25_PARAMS_PATH)


@dataclass
class RetrievedChunk:
    text: str
    category: str
    source_file: str
    chunk_index: int
    score: float


def embed_query(text: str) -> list[float]:
    # input_type="query" here, NOT "document" -- Voyage prepends a
    # different instruction string for each side. Mixing them up doesn't
    # error, it just quietly retrieves worse.
    response = vo.embed(texts=[text], model=EMBED_MODEL, input_type="query")
    return response.embeddings[0]


def _sanitize_categories(allowed_categories: list[str]) -> list[str]:
    """Never trust a category value that isn't one of the four canonical
    practice areas, even if it somehow got this far. Fails closed (returns
    fewer/no namespaces) rather than open."""
    return [c for c in allowed_categories if c in ALL_PRACTICE_AREAS]


class PineconeHybridRetriever:
    def retrieve(
        self,
        query: str,
        allowed_categories: list[str],
        k: int = 20,
        hyde_passage: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        allowed_categories is whatever your RBAC layer resolves for the
        current user (already expanded to all 4 areas upstream for
        managing_partner / unrestricted users -- this function doesn't
        do that resolution itself, so pass an explicit list either way).

        hyde_passage: if given, its embedding drives the DENSE search
        instead of the raw query's. The sparse/BM25 side always uses the
        real query regardless -- keyword matching should reflect what the
        user actually typed, not a hypothetical generated passage.
        """
        categories = _sanitize_categories(allowed_categories)
        if not categories:
            return []

        dense_query_text = hyde_passage or query
        dense_vector = embed_query(dense_query_text)
        # encode_queries, NOT encode_documents -- BM25Encoder weights these
        # two differently on purpose.
        sparse_vector = bm25.encode_queries(query)

        candidates: list[RetrievedChunk] = []
        for category in categories:
            results = index.query(
                namespace=category,
                vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=k,
                include_metadata=True,
            )
            for match in results.get("matches", []):
                meta = match["metadata"]
                candidates.append(
                    RetrievedChunk(
                        text=meta.get("text", ""),
                        category=category,
                        source_file=meta.get("source_file", ""),
                        chunk_index=meta.get("chunk_index", 0),
                        score=match["score"],
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]


retriever = PineconeHybridRetriever()


# if __name__ == "__main__":
#     # Quick smoke test against your live index before wiring into rag_chain.py.
#     test_query = "What are the termination provisions in this agreement?"

#     print("-- as admin (all 4 areas) --")
#     for r in retriever.retrieve(test_query, allowed_categories=ALL_PRACTICE_AREAS, k=3):
#         print(f"  [{r.score:.4f}] {r.category} / {r.source_file} (chunk {r.chunk_index})")
#         print(f"    {r.text[:150]}...")

#     print("\n-- as carol (employment_hr, general_commercial only) --")
#     for r in retriever.retrieve(
#         test_query, allowed_categories=["employment_hr", "general_commercial"], k=3
#     ):
#         print(f"  [{r.score:.4f}] {r.category} / {r.source_file} (chunk {r.chunk_index})")