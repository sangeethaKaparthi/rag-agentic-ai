import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")