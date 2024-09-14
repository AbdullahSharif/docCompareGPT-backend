from fastapi import FastAPI
from app.routes import auth,admin,admindocument,userchat,generateReport
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
app = FastAPI()
from dotenv import load_dotenv
import os
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")


app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(admindocument.router)
app.include_router (userchat.router)
app.include_router(generateReport.router)

index_name = 'document-index-20240902105359'

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI AI Project"}