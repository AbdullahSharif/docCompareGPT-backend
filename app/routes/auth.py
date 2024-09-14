# app/routes/auth.py

from fastapi import APIRouter, HTTPException
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.utils.hash import hash_password, verify_password
from app.utils.jwt import create_access_token
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from bson import ObjectId
from datetime import timedelta

router = APIRouter()

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.MONGO_DB]


# sign up 
@router.post("/signup", response_model=UserResponse)
async def signup(user_data: UserCreate):

    user = await db["users"].find_one({"email": user_data.email})

    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pwd = hash_password(user_data.password)
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_pwd,
        "user_type": user_data.user_type
    }
    
    result = await db["users"].insert_one(new_user)
    new_user["id"] = str(result.inserted_id)
    
    return UserResponse(username=new_user["username"], email=new_user["email"], message ="User created successfully")


# login 
@router.post("/login")
async def login(user_data: UserLogin):
    print("user_data",user_data)
    user = await db["users"].find_one({"email": user_data.email})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not verify_password(user_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Create JWT token
    access_token = create_access_token(data={"sub": user["email"]})
    
    return {
        "username": user["username"],
        "email": user["email"],
        "message": "You have been logged in successfully",
        "access_token": access_token,
        "token_type": "bearer",
        "user_type": user["user_type"]
    }
