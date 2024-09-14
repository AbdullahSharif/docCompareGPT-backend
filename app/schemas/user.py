# app/schemas/user.py

from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    user_type: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    username: str
    email: EmailStr
    message:str

class UserLoginResponse(BaseModel):
    username: str
    email: EmailStr
    message:str
    access_token: str
    message: str
    user_type: str


    class Config:
        # orm_mode is now from_attributes in Pydantic v2.x
        from_attributes = True
