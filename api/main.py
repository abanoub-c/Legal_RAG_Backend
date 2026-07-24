import time
from datetime import timedelta
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uuid

from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    get_current_user,
    resolve_category_filter,
)
from database import get_db, init_db
from models import ConversationHistory, User
from rag_chain import query_rag
from schemas import ConversationEntry, QueryRequest, QueryResponse, Source, Token, UserOut ,ConversationSummary

app = FastAPI(
    title="Legal RAG API",
    description="Query a categorized CUAD contract repository with role-based access control",
    version="2.0.0",
)

# Allow requests from your React UI (any origin for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Legal RAG API is running"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    allowed_categories = resolve_category_filter(current_user, request.requested_categories)
    conversation_id = request.conversation_id or str(uuid.uuid4())

    start = time.time()
    try:
        answer, sources , grounding = query_rag(
            question=request.question,
            allowed_practice_areas=allowed_categories,
            k=request.k,
            candidate_pool=request.candidate_pool, 
            use_hyde =request.use_hyde, 
            model=request.model,
            use_grounding=request.use_grounding,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # allowed_categories is always a list at this point -- join it to match the
    # String column on ConversationHistory.
    history_entry = ConversationHistory(
        user_id=current_user.id,
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
        allowed_practice_areas=",".join(allowed_categories),
        grounded=grounding["grounded"],
    )
    db.add(history_entry)
    db.commit()

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[Source(**s) for s in sources],
        processing_time=round(time.time() - start, 2),
        grounded=grounding["grounded"],
        unsupported_claims=grounding["unsupported_claims"],
        conversation_id=conversation_id,
    )


@app.get("/history", response_model=List[ConversationSummary])
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    entries = (
        db.query(ConversationHistory)
        .filter(ConversationHistory.user_id == current_user.id)
        .order_by(ConversationHistory.created_at.asc())
        .all()
    )

    threads: dict[str, dict] = {}
    for entry in entries:
        cid = entry.conversation_id
        if cid not in threads:
            threads[cid] = {
                "conversation_id": cid,
                "first_question": entry.question,
                "message_count": 0,
                "last_message_at": entry.created_at,
            }
        threads[cid]["message_count"] += 1
        threads[cid]["last_message_at"] = entry.created_at

    summaries = sorted(threads.values(), key=lambda t: t["last_message_at"], reverse=True)
    return summaries[:limit]

@app.get("/history/{conversation_id}", response_model=List[ConversationEntry])
def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entries = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == current_user.id,
            ConversationHistory.conversation_id == conversation_id,
        )
        .order_by(ConversationHistory.created_at.asc())
        .all()
    )
    if not entries:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return entries

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Query records associated with this conversation ID AND the current logged-in user
    records = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.conversation_id == conversation_id,
            ConversationHistory.user_id == current_user.id
        )
        .all()
    )
    
    if not records:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    # 2. Delete the records
    for record in records:
        db.delete(record)
        
    # 3. Commit to Neon DB to make deletion permanent!
    db.commit()
    
    return {"status": "success", "message": f"Conversation {conversation_id} deleted"}

@app.post("/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
