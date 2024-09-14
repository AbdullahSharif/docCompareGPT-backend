# # # app/config.py
# from dotenv import load_dotenv
# import os
# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     MONGO_URI: str = "mongodb://localhost:27017"
#     MONGO_DB: str = "fastapi_auth"
#     SECRET_KEY: str = "MSS is Developed"  

#     class Config:
#         # Optional if you use .env file
#         # env_file = ".env"
#         pass

# settings = Settings()


from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB: str
    SECRET_KEY: str
    OPENAI_API_KEY1: str
    OPENAI_API_KEY2: str
    PINECONE_API_KEY: str

    class Config:
        # Specify the path to the .env file if it's not in the root directory
        env_file = ".env"

settings = Settings()
