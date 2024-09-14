# app/models/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from bson import ObjectId

class User(BaseModel):
    id: Optional[ObjectId] = None
    username: str
    email: EmailStr
    password: str
    userType=""
