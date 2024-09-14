from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC as Pinecone

def get_pinecone_client():
    # Load environment variables from .env file
    load_dotenv()
    pine = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pine)
    return pc

