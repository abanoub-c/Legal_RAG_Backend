from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text , Boolean
from sqlalchemy.orm import relationship
import uuid as uuid_lib

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    allowed_practice_areas = Column(String, nullable=True)  # "ip_tech" | "ip_tech,general_commercial" | NULL = all

    conversations = relationship("ConversationHistory", back_populates="user")


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    conversation_id = Column(String, nullable=False, index=True, default=lambda: str(uuid_lib.uuid4()))
    answer = Column(Text, nullable=False)
    allowed_practice_areas = Column(String, nullable=True)  # categories this query was actually scoped to
    created_at = Column(DateTime, default=datetime.utcnow)
    grounded = Column(Boolean, nullable=True)   

    user = relationship("User", back_populates="conversations")