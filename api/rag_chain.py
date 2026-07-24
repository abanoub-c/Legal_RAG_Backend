import json
from functools import lru_cache
import os
import cohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Pinecone_hybrid_retriever import retriever as _retriever
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_groq import ChatGroq
from config import (
    GOOGLE_API_KEY,
    NVIDIA_API_KEY,
    GROQ_API_KEY,
    LLM_NUM_PREDICT,
    COHERE_API_KEY,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)

# ── Model registry ──────────────────────────────────────────────────────
# Single source of truth for which models the UI's selector can request.
_MODEL_REGISTRY = {
    # 1. Google Gemini (Generous free tier)
    "gemini-2.5-flash": lambda: ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        max_output_tokens=LLM_NUM_PREDICT,
        google_api_key=GOOGLE_API_KEY,
    ),
    
    # 2. NVIDIA Nemotron (Using your existing NVIDIA API Key)
    "nemotron-3-ultra": lambda: ChatNVIDIA(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        max_completion_tokens=LLM_NUM_PREDICT,
        nvidia_api_key=NVIDIA_API_KEY,
    ),
    
    # 3. NVIDIA Mixtral (Uses the same NVIDIA key, no new account needed!)
    "mixtral-8x7b": lambda: ChatNVIDIA(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        max_completion_tokens=LLM_NUM_PREDICT,
        nvidia_api_key=NVIDIA_API_KEY,
    ),

    # 4. Groq Llama 3.3 70B (Generous free tier, incredibly fast inference)
    "llama-3.3-70b": lambda: ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=LLM_NUM_PREDICT,
        groq_api_key=GROQ_API_KEY,
    ),
}
DEFAULT_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=None)
def get_llm(model_name: str):
    """Returns a ready-to-use chat model instance for the given name, cached
    so each distinct model is only ever instantiated once."""
    try:
        return _MODEL_REGISTRY[model_name]()
    except KeyError:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {sorted(_MODEL_REGISTRY)}"
        )


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant helping analyze contracts for a law firm.
Answer using ONLY the provided context from the firm's contract repository.
Cite the contract category and source file in your answer; do not write any source in the answer just write the answer.
If the answer is not in the context, say so clearly — do not guess."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# HyDE: generates a hypothetical contract clause from the question. That clause
# (not the raw question) gets embedded for dense retrieval -- a short or casually
# phrased question and the contract clause that answers it are worded very
# differently, but a generated clause written in the same style as your real
# chunks tends to land closer to it in embedding space. Uses whichever model
# the request selected, same as the final answer.
_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You draft hypothetical excerpts from legal contracts for a retrieval system.
Given a question, write ONE short hypothetical clause or passage, in formal legal
drafting style, that would plausibly answer it — as if quoting directly from a
real contract. Do not answer the question conversationally, do not add caveats
or disclaimers, and do not mention that this is hypothetical. Output only the
clause text itself, 2-4 sentences."""),
    ("human", "Question: {question}\n\nHypothetical contract clause:"),
])

_GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checker for a legal RAG system.
Given a CONTEXT (retrieved contract excerpts) and an ANSWER generated from that context,
determine whether every factual claim in the ANSWER is directly supported by the CONTEXT.

Respond with ONLY valid JSON, no markdown, no code fences, no extra text:
{{"grounded": true or false, "unsupported_claims": ["claim text", ...]}}

"unsupported_claims" should list any specific statements in the ANSWER that are not
backed by the CONTEXT. If the answer explicitly says it doesn't know, that counts as grounded."""),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


def _check_grounding(context: str, answer: str, grounding_chain) -> dict:
    """Returns {'grounded': bool | None, 'unsupported_claims': list[str]}.
    None means the check itself failed (API error / bad JSON) -- treated as
    'unknown', not as a hallucination, so it never blocks the response.

    grounding_chain is passed in explicitly (rather than read from a module
    global) because it depends on the per-request selected model -- there's
    no fixed chain this function could otherwise close over.
    """
    try:
        raw = grounding_chain.invoke({"context": context, "answer": answer})
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return {
            "grounded": bool(parsed.get("grounded")),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except Exception as e:
        print(f"[grounding check failed, skipping]: {e}")
        return {"grounded": None, "unsupported_claims": []}


_co = cohere.Client(COHERE_API_KEY)


def _rerank(question: str, chunks: list, top_n: int):
    """Reranks an already-retrieved list of chunk objects via Cohere.
    Attaches .rerank_score to each chunk, returns them sorted best-first."""
    if not chunks:
        return []

    response = _co.rerank(
        model=RERANK_MODEL,
        query=question,
        documents=[chunk.text for chunk in chunks],
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index]     # map back to original chunk object
        chunk.rerank_score = result.relevance_score
        reranked.append(chunk)
    return reranked


def query_rag(
    question: str,
    allowed_practice_areas: list[str],
    k: int = RERANK_TOP_N,
    candidate_pool: int = 20, 
    use_hyde: bool = True,
    model: str = DEFAULT_MODEL,
    use_grounding: bool = True,
):
    """
    allowed_practice_areas must already be RBAC-resolved (see auth.resolve_category_filter
    in main.py's /query endpoint) — this function does not re-check permissions, it
    only ever queries the Weaviate tenants it's handed.

    model: key into _MODEL_REGISTRY. Raises ValueError (see get_llm) if unknown.
    use_hyde: when True, generates a hypothetical contract clause and uses ITS
    embedding for the dense half of retrieval. The real question is still always
    used for BGE-M3's sparse/lexical scoring either way.

    Returns (answer: str, sources: list[dict], grounding: dict) -- always a
    3-tuple, including on the "no documents found" early-return path.
    """
    if not allowed_practice_areas:
        return (
            "Please select at least one practice area from the settings menu to perform a search.",
            [],
            {"grounded": True, "unsupported_claims": []},
        )
    
    llm = get_llm(model)
    # LCEL composition (`|`) is cheap -- just wiring runnables together, no
    # network calls -- so building these per-request to use the selected
    # model costs nothing meaningful; the expensive part (client construction)
    # is what get_llm() caches.
    chain = _PROMPT | llm | StrOutputParser()
    hyde_chain = _HYDE_PROMPT | llm | StrOutputParser()
    grounding_chain = _GROUNDING_PROMPT | llm | StrOutputParser()

    try:
        hyde_passage = hyde_chain.invoke({"question": question}) if use_hyde else None
    except Exception as e:
        print(f"[HyDE failed, falling back to raw query]: {e}")
        hyde_passage = None

    # Stage 1: wide retrieval (dense BGE-M3, on the hypothetical passage if HyDE
    # is on, + sparse on the real question), per allowed tenant
    all_chunks = _retriever.retrieve(
        question,
        allowed_categories=allowed_practice_areas,
        k=candidate_pool,
        hyde_passage=hyde_passage,
    )

    if not all_chunks:
        return (
            "No relevant documents were found in the categories you have access to.",
            [],
            {"grounded": None, "unsupported_claims": []},
        )

    # Stage 2: Cohere rerank down to top-k
    try:
        chunks = _rerank(question, all_chunks, top_n=k)
    except Exception as e:
        print(f"[rerank failed, falling back to unranked top-{k}]: {e}")
        chunks = all_chunks[:k]

    # Format context with source labels
    context = "\n\n---\n\n".join([
        f"[{chunk.category} | {chunk.source_file} | chunk {chunk.chunk_index}]\n{chunk.text}"
        for chunk in chunks
    ])

    # Generate answer
    answer = chain.invoke({"context": context, "question": question})

    # Stage 3: verify the answer is actually supported by the retrieved context
    grounding = _check_grounding(context, answer, grounding_chain) if use_grounding else {"grounded": None, "unsupported_claims": []}

    # Build source list
    sources = [{
        "category": chunk.category,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "score": round(chunk.score, 4),
        "rerank_score": round(chunk.rerank_score, 4) if getattr(chunk, "rerank_score", None) is not None else None,
        "chunk_preview": chunk.text[:150].strip(),
    } for chunk in chunks]

    return answer, sources, grounding
