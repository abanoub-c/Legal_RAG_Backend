from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    requested_categories: Optional[List[str]] = None
    k: int = 5  # chunks to retrieve (default 5)
    candidate_pool: int = 20  
    use_hyde: bool = True
    model: str = "gemini-2.5-flash"
    use_grounding: bool = True
    conversation_id: Optional[str] = None
 


class Source(BaseModel):
    category: str
    source_file: str
    chunk_index: int
    score: float
    rerank_score: float
    chunk_preview: str  # first 150 chars -- shows where the answer came from


class Grounding(BaseModel):
    grounded: Optional[bool] = None
    unsupported_claims: List[str] = []

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    processing_time: float
    grounded: Optional[bool] = None          # None = check failed/skipped
    unsupported_claims: List[str] = []
    conversation_id: str

class Token(BaseModel):
    access_token: str
    token_type: str


class UserOut(BaseModel):
    username: str
    full_name: str
    role: str
    allowed_practice_areas: Optional[str] = None  # comma-separated string, or None = unrestricted

    class Config:
        from_attributes = True  # lets Pydantic read SQLAlchemy objects directly


class ConversationEntry(BaseModel):
    id: int
    conversation_id: str
    question: str
    answer: str
    allowed_practice_areas: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True



class ConversationSummary(BaseModel):
    conversation_id: str
    first_question: str
    message_count: int
    last_message_at: datetime