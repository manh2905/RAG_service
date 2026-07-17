from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import get_settings

security = HTTPBearer()

def verify_internal_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify the Bearer token matches INTERNAL_SECRET.
    Used to secure internal API endpoints from Node.js backend.
    """
    settings = get_settings()
    if credentials.credentials != settings.INTERNAL_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
