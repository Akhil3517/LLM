import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    print(f"Error during client initialization in config.py: {e}")
    pc = None
    genai = None

PINECONE_INDEX_NAME = "hackrx-gemini-index"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_GENERATION_MODEL = "gemini-2.5-pro"