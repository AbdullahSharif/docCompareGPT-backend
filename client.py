# client.py
from dotenv import load_dotenv
import os
import openai  # Use `import openai` instead of `from openai import OpenAI`

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def get_openai_client():
    return openai

