from fastapi import APIRouter, Depends, HTTPException, status
from app.utils.auth import get_current_user
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from bson import ObjectId

router = APIRouter()
client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.MONGO_DB]

# route for getting all users by admin
@router.get("/admin/all-users")
async def get_all_users(current_user: dict = Depends(get_current_user)):
    user = await db["users"].find_one({"email": current_user["sub"]})
    if user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="You are not authorized to perform this action")
    users = []
    async for user in db["users"].find({"user_type": {"$ne": "admin"}}):
        user["_id"] = str(user["_id"])
        users.append(user)
    return users



# route for deleting a user by admin
@router.delete("/admin/all-users/{user_id}")
async def delete_user(user_id: str,current_user: dict = Depends(get_current_user)):
    user = await db["users"].find_one({"email": current_user["sub"]})
    if user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="You are not authorized to perform this action")

    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["user_type"] == "admin":
        raise HTTPException(status_code=404, detail=" Admin cannot be removed ")

    result = await db["users"].delete_one({"_id":  ObjectId(user_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}