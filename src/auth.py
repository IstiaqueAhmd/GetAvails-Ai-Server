from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import os

# JWT Configuration - uses the same secret as the main backend
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# OAuth2 scheme for token extraction (tokenUrl is just for OpenAPI docs)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=True)


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode and validate a JWT token from the main backend.
    
    Args:
        token: JWT token string
    
    Returns:
        uid from token if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid: str = payload.get("uid")
        if uid is None:
            return None
        return uid
    except JWTError:
        return None


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Validate JWT token and extract user ID.
    
    This is a FastAPI dependency that validates tokens from the main backend.
    It does NOT look up users in a local database - just validates the token
    and returns the uid.
    
    Args:
        token: JWT token from Authorization header
    
    Returns:
        User ID (uid) as a string
    
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    uid = decode_access_token(token)
    if uid is None:
        raise credentials_exception
    
    return uid
