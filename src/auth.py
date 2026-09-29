from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import os

# JWT Configuration - uses the same secret as the main backend
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Tokens are issued by the main backend, not this service. Clients send them as
# "Authorization: Bearer <token>". HTTPBearer (rather than OAuth2PasswordBearer)
# also gives Swagger UI a field to paste a token into. auto_error is off so a
# missing header returns a 401 with a clear message instead of FastAPI's 403.
bearer_scheme = HTTPBearer(auto_error=False)


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode and validate a JWT token from the main backend.

    Args:
        token: JWT token string

    Returns:
        user_id from token if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Reject non-access tokens (e.g. refresh tokens) if a type is present.
        token_type = payload.get("token_type")
        if token_type is not None and token_type != "access":
            return None

        user_id = payload.get("user_id")
        if user_id is None:
            return None
        return str(user_id)
    except JWTError:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Validate JWT token and extract user ID.

    This is a FastAPI dependency that validates tokens from the main backend.
    It does NOT look up users in a local database - just validates the token
    and returns the user_id.

    Args:
        credentials: Bearer credentials from the Authorization header

    Returns:
        User ID (user_id) as a string

    Raises:
        HTTPException: If the header is missing or the token is invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: send an 'Authorization: Bearer <token>' header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise credentials_exception

    return user_id
