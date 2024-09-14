from fastapi import File, UploadFile, Depends,APIRouter, WebSocket, Form,HTTPException
import shutil
import os
from ..utils.rag_utilis import load_pdf_content, divide_into_chunks, get_embeddings, batch_upsert,generate_answer,query_pinecone
from client import get_openai_client
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from ..models.standard_document import Standard_Document
from pinecone.grpc import PineconeGRPC as Pinecone
router = APIRouter()
client = AsyncIOMotorClient(settings.MONGO_URI)
from datetime import datetime
db = client[settings.MONGO_DB]
from bson import ObjectId
from openai import OpenAI
from dotenv import load_dotenv
from app.utils.auth import get_current_user
from pydantic import BaseModel


load_dotenv()
index_name = 'personal-document-upload-index'
api_key = settings.OPENAI_API_KEY1

class ChatRequest(BaseModel):
    question: str

@router.post("/user/document-for-chat")
async def upload_file(
    file: UploadFile = File(...),

    current_user: dict = Depends(get_current_user)
):
    try:
        
        # Define the target directory (one level up)
        target_directory = os.path.join(os.path.dirname(__file__), '../uploaded documents')
        print("File",file)

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
        embeddings = get_embeddings(chunks,client)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        namespace_name = f"nameless-{timestamp}"
        admin_document = Standard_Document(
            nameofdoc="namelessbyuser",
            description="descriptionlessbyuser",
            index_name=index_name,
            namespace_name=namespace_name,
            by_admin=False
        )
        success = batch_upsert(pc, index_name, namespace_name, embeddings,chunks)
        if success:
            # Insert the document record into MongoDB
            document_id = await db["standard_document"].insert_one(admin_document.model_dump())
            print("Document record inserted into MongoDB")
            # Remove the file from the server after processing
            os.remove(file_path)

            return {"message": "File uploaded successfully","document_id": str(document_id.inserted_id)}
        else:
            return {"error": "Failed to upsert embeddings to Pinecone"}
    except Exception as e:
        return {"error": str(e)}





@router.post("/user/chat")
async def chat_with_document( body: ChatRequest,
                                 current_user: dict = Depends(get_current_user)
 
):
    try:
       
        query = body.question
        cursor = db["standard_document"].find({}, {"namespace_name": 1, "_id": 0})
        documents = await cursor.to_list(length=None)
        
        if (len(documents) == 0):
            return {"response": "I am not trained on any document yet."}        
        # Initialize OpenAI and Pinecone clients
        # openai_client = OpenAI(api_key=api_key)
        # pinecone_client = Pinecone(api_key=pinecone_api_key)

        openai_client = OpenAI(api_key=api_key)
        pine = os.getenv("PINECONE_API_KEY")
        pinecone_client = Pinecone(api_key=pine)
      

        # Generate embeddings for the query
        query_embedding = get_embeddings([query], openai_client)
        

        # Query Pinecone for relevant document chunks
        chunks = query_pinecone(pinecone_client, "admin-upload-index-1", documents, query_embedding)
        if not chunks:
            return {"message": "No relevant information found in the document."}
        
        # Generate a response using the retrieved chunks
        response = generate_answer(chunks, query, openai_client)
        
        return {"response": response}
    
    except Exception as e:
    
        raise HTTPException(status_code=500, detail= str(e))


@router.get("/user/document-for-chat",)
async def get_all_documents(

    current_user: dict = Depends(get_current_user)
):
    user = await db["users"].find_one({"email": current_user["sub"]})

    documents = []
    async for document in db["standard_document"].find({"by_admin": True}):
        document["_id"] = str(document["_id"])
        documents.append(document)
    return documents