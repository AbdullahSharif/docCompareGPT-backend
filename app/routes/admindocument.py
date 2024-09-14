from fastapi import File, UploadFile, APIRouter,Depends, WebSocket, Form,HTTPException
import shutil
from app.utils.auth import get_current_user
import os
from ..utils.rag_utilis import load_pdf_content, divide_into_chunks, get_embeddings, batch_upsert
from client import get_openai_client
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from ..models.standard_document import Standard_Document
from pinecone.grpc import PineconeGRPC as Pinecone
router = APIRouter()
client = AsyncIOMotorClient(settings.MONGO_URI)
from datetime import datetime
db = client[settings.MONGO_DB]
from openai import OpenAI
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()
index_name = 'document-index-20240902105359'

index_names=["admin-upload-index-1","admin-upload-index-2","admin-upload-index-3"]
api_key = settings.OPENAI_API_KEY1
@router.post("/admin/standard-document")
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        user = await db["users"].find_one({"email": current_user["sub"]})
        if user["user_type"] != "admin":
            raise HTTPException(status_code=403, detail="You are not authorized to perform this action")
         
        # Define the target directory (one level up)
        target_directory = os.path.join(os.path.dirname(__file__), '../uploaded documents')

        # Create the directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Create the full file path with an f-string
        file_path = os.path.join(target_directory, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        doc = load_pdf_content(file_path)
        # Delete the file
        chunks = divide_into_chunks(doc)
               
        client = OpenAI(api_key=api_key)
        pine = settings.PINECONE_API_KEY
        pc = Pinecone(api_key=pine)
       
        # try each index_name in index_names:
        for index_name in index_names:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            namespace_name = f"{name}-{timestamp}"
            embeddings = get_embeddings(chunks,client)
            success = batch_upsert(pc, index_name, namespace_name, embeddings, chunks)
            if success:
                # Create a document record
                admin_document = Standard_Document(
                    nameofdoc=name,
                    description=description,
                    index_name=index_name,
                    namespace_name=namespace_name,
                    by_admin=True
                )
                # Insert the document record into MongoDB
                await db["standard_document"].insert_one(admin_document.model_dump())
                print("Document record inserted into MongoDB")
                # Remove the file from the server after processing
                os.remove(file_path)
                return {"message": "File uploaded successfully"}
            else:
                print(f"Failed to upsert to {index_name}, trying the next index...")

        # If all indexes fail
        raise HTTPException(status_code=500, detail="Failed to upsert embeddings to all indexes")


    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/admin/standard-document/{document_id}")
async def delete_document(document_id: str,current_user: dict = Depends(get_current_user)):
    try:
        user = await db["users"].find_one({"email": current_user["sub"]})
        if user["user_type"] != "admin":
            raise HTTPException(status_code=403, detail="You are not authorized to perform this action")
        # Convert document_id to ObjectId
        document_object_id = ObjectId(document_id)

        # Fetch the document from MongoDB by its ObjectId
        document = await db["standard_document"].find_one({"_id": document_object_id})
        
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        # Extract necessary information from the document
        index_name = document["index_name"]
        namespace_name = document["namespace_name"]

        # Initialize Pinecone client
        pine = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pine)

        index=pc.Index(index_name)

        index.delete(delete_all=True, namespace=namespace_name)
    
        # Remove the document from MongoDB
        delete_result = await db["standard_document"].delete_one({"_id": document_object_id})

        if delete_result.deleted_count == 1:
            return {"message": "Document and associated embeddings deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document from MongoDB")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/admin/standard-document")
async def get_all_documents(current_user: dict = Depends(get_current_user)):
    user = await db["users"].find_one({"email": current_user["sub"]})
    if user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="You are not authorized to perform this action")
    documents = []
    async for document in db["standard_document"].find({"by_admin":True}):
        document["_id"] = str(document["_id"])
        documents.append(document)
    return documents