import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import get_db
from models import User

# ── Config ──────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def resolve_category_filter(user: User, requested: Optional[List[str]] = None) -> List[str]:
    """RBAC -- resolves the final list of categories a query is allowed to search.

    `user.allowed_practice_areas` is a comma-separated string column on the User
    row; NULL/empty means unrestricted (admin / managing partner).

    `requested` is the optional client-supplied narrowing list from QueryRequest.
    It must actually be a list here -- if it's a bare string, `set(requested)`
    below silently iterates individual characters instead of category names.
    """
    RBAC_CATEGORIES = ["ip_tech", "general_commercial", "corporate_ma", "employment_hr"]

    if not user.allowed_practice_areas:
        allowed = RBAC_CATEGORIES
    else:
        allowed = [c.strip() for c in user.allowed_practice_areas.split(",")]

    # FIXED: Check explicitly for None so that an empty list [] is processed
    if requested is not None:
        unauthorized = set(requested) - set(allowed)
        if unauthorized:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Not authorized for: {sorted(unauthorized)}")
        return requested
        
    return allowed