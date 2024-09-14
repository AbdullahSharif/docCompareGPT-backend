from fastapi import File, UploadFile, APIRouter,Depends, WebSocket, Form,HTTPException
import shutil
from app.utils.auth import get_current_user
import os
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
from ..utils.comparedoc import load_pdf_content, divide_into_chunks, preprocess_document, get_embeddings, batch_upsert,embed_question,get_context,get_similar_chunks,generate_response,generate_report
import traceback
import concurrent.futures
import asyncio

load_dotenv()

index_name = "compare-document-index"
api_key1 = settings.OPENAI_API_KEY1
api_key2 =settings.OPENAI_API_KEY2
question = ["important points,rules,in compliance with,content,table of content,standards, policy, requirements, guidelines, best practices"]

router = APIRouter()

def process_document(file, file_path, client, pc, index_name, question):
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the document
        doc = load_pdf_content(file_path)
        # doc = preprocess_document(doc)
        chunks = divide_into_chunks(doc)

        # Generate embeddings
        embeddings = get_embeddings(chunks, client)

        # Create namespace for the embeddings
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        namespace_name = f"{file.filename}-{timestamp}"

        # Upsert embeddings into Pinecone
        success = batch_upsert(pc, index_name, namespace_name, embeddings, chunks)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to upsert embeddings, please try again")

        # Get embeddings for the question and generate context
        question_embedding = get_embeddings(question, client)
        standard_context = get_context(question_embedding[0], chunks, get_similar_chunks, pc, index_name, namespace_name)

        # Generate response
        response = generate_response(client, context=standard_context)

        # Clean up: Remove file and delete namespace
        os.remove(file_path)
        index = pc.Index(index_name)
        index.delete(namespace=namespace_name, delete_all=True)

        return {"Response": response, "Similiar Chunks": standard_context}

    except AttributeError as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"AttributeError: {str(e)}\nTraceback:\n{tb_str}")
    except Exception as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}\nTraceback:\n{tb_str}")

async def process_documents_in_parallel(process_document, standard_file, standard_file_path, standard_client, 
                                        personal_file, personal_file_path, personal_client, 
                                        pc1,pc2, index_name, question,
                                        ):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        # Run both document processes concurrently in the thread pool
        future_1 = loop.run_in_executor(executor, process_document, standard_file, standard_file_path, standard_client, pc1, index_name, question)
        future_2 = loop.run_in_executor(executor, process_document, personal_file, personal_file_path, personal_client, pc2, index_name, question)

        # Await both results
        result_1 = await future_1
        result_2 = await future_2

    return result_1, result_2

@router.post("/admin/compare-document")
async def compare_document(
    standard_file: UploadFile = File(...),
    personal_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        print(current_user)
        standard_client = OpenAI(api_key=api_key1)
        personal_client = OpenAI(api_key=api_key2)
        pine1 = os.getenv("PINECONE_API_KEY")
        pine2 = os.getenv("PINECONE_API_KEY")
        pc1 = Pinecone(api_key=pine1)
        pc2 = Pinecone(api_key=pine2)
        target_directory = os.path.join(os.path.dirname(__file__), '../uploaded documents')

        # Create the directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Define file paths
        standard_file_path = os.path.join(target_directory, standard_file.filename)
        personal_file_path = os.path.join(target_directory, personal_file.filename)
        
        if (standard_file_path == personal_file_path):
            raise HTTPException(status_code=400, detail="Please upload two different files")
        # Process documents concurrently
        standard_result, personal_result = await process_documents_in_parallel(
            process_document,
            standard_file, standard_file_path, standard_client,
            personal_file, personal_file_path, personal_client,
            pc1,pc2, index_name, question
        )
        standard_response = standard_result.get("Response")
        personal_response = personal_result.get("Response")
        report = generate_report(standard_client,standard_response, personal_response)
        return {"Report": report}
    except AttributeError as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"AttributeError: {str(e)}\nTraceback:\n{tb_str}")
    
    except Exception as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}\nTraceback:\n{tb_str}")
