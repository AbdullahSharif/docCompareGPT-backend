# app/utils/auth.py

from fastapi import Request, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.utils.jwt import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload
