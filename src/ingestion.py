from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (
    GOOGLE_API_KEY,
    PDF_PATH,
    PINECONE_INDEX_NAME
)


def run_ingestion():
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536
    )

    print("Creating embeddings and storing in Pinecone...")

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("Ingestion completed successfully.")

    return vector_store


if __name__ == "__main__":
    run_ingestion()