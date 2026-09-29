# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using the
provided **Agentic AI eBook** as its knowledge source.

The application uses **LangGraph** to orchestrate the RAG workflow,
**Pinecone** as the vector database, **Google Gemini** for embeddings and
response generation, and **FastAPI** to expose the chatbot through a REST API.

---

## 1. Project Objective

The objective of this project is to build a document-grounded AI chatbot that:

- Ingests the Agentic AI eBook PDF.
- Extracts and splits the document into smaller text chunks.
- Converts the chunks into vector embeddings.
- Stores the embeddings in Pinecone.
- Retrieves the most relevant document chunks for a user question.
- Uses LangGraph to orchestrate retrieval and answer generation.
- Generates answers using only the retrieved document context.
- Refuses to answer when the requested information cannot be found in the
  retrieved document context.
- Exposes the RAG workflow through a FastAPI `/chat` endpoint.
- Returns the generated answer, retrieved context, and a confidence score.

---

## 2. Technologies Used

- Python 3.11
- LangChain
- LangGraph
- Google Gemini
- Gemini Embeddings
- Pinecone
- FastAPI
- Uvicorn
- PyPDF
- Python-dotenv
- Requests

---

## 3. Architecture

The application follows this RAG pipeline:

```text
                    Agentic AI eBook PDF
                             |
                             v
                       PyPDFLoader
                             |
                             v
                    Text Chunking
             RecursiveCharacterTextSplitter
                             |
                             v
                    Gemini Embeddings
                             |
                             v
                         Pinecone
                    Vector Database
                             |
                             |
User Question --------------+
        |
        v
      FastAPI
       /chat
        |
        v
     LangGraph
        |
        v
 Retrieve Top 3 Relevant Chunks
        |
        v
    Gemini LLM
        |
        v
  Grounded Response
        |
        +---------------------------+
        |                           |
        v                           v
  Final Answer              Retrieved Context
        |
        v
  Confidence Score
4. Project Structure
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
│
├── app.py
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
File Responsibilities
data/Ebook-Agentic-AI.pdf

The Agentic AI eBook used as the knowledge source for the RAG system.

src/config.py

Loads environment variables and defines project configuration such as:

Google Gemini API key
Pinecone API key
Pinecone index name
PDF path
src/ingestion.py

Responsible for:

Loading the PDF using PyPDFLoader.
Splitting the document into text chunks.
Creating Gemini embeddings.
Storing the chunks and embeddings in Pinecone.
src/graph.py

Defines the LangGraph RAG workflow.

The graph contains:

START
  |
  v
retrieve
  |
  v
generate
  |
  v
END

The graph state contains:

question
context
answer
score
app.py

Creates the FastAPI application and exposes the /chat endpoint.

tests_sample_queries.py

Contains benchmark questions used to verify retrieval, grounded generation,
and out-of-document behavior.

5. Environment Setup
Prerequisites

Python 3.10 or higher is required.

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the project dependencies:

pip install -r requirements.txt
6. Environment Variables

Create a .env file in the project root.

GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index

Do not commit the .env file to GitHub.

A .env.example file is included as a template.

7. Pinecone Configuration

The Pinecone index used by the application is configured as:

Index Name: agentic-ai-index
Dimension: 1536
Metric: cosine

The embedding output is configured to use a 1536-dimensional representation
so that it can be stored in the Pinecone index.

8. Document Ingestion

The source PDF is located at:

data/Ebook-Agentic-AI.pdf

The ingestion pipeline uses PyPDFLoader to load the document and
RecursiveCharacterTextSplitter to create text chunks.

The current configuration uses:

chunk_size = 1500
chunk_overlap = 200

The PDF contains 60 pages.

The ingestion process produced 83 chunks.

Run the ingestion script from the project root:

python -m src.ingestion

Expected output:

Loading PDF...
Loaded 60 pages.
Created 83 chunks.
Creating embeddings and storing in Pinecone...
Ingestion completed successfully.
9. Running the FastAPI Application

Start the FastAPI application:

python -m uvicorn app:app --reload

The API will be available at:

http://127.0.0.1:8000

FastAPI Swagger UI is available at:

http://127.0.0.1:8000/docs
10. API Endpoint
POST /chat

The endpoint accepts a JSON request containing a question.

Request
{
  "query": "What is Agentic AI according to the eBook?"
}
Response
{
  "final_answer": "According to the provided document, Agentic AI is defined as AI systems capable of autonomous decision-making, with the capabilities to learn and adapt to new situations.",
  "retrieved_context": [
    "Retrieved document chunk 1...",
    "Retrieved document chunk 2...",
    "Retrieved document chunk 3..."
  ],
  "confidence_score": 0.95
}

The response contains:

final_answer - the generated answer based on the retrieved context.
retrieved_context - the top 3 retrieved document chunks.
confidence_score - the confidence value returned by the RAG workflow.
11. RAG Workflow

The LangGraph workflow consists of two main nodes.

Retrieve Node

The retrieve node:

Receives the user's question.
Searches Pinecone using the question embedding.
Retrieves the top 3 relevant document chunks.
Stores the retrieved chunks in the graph state.
Generate Node

The generate node:

Combines the retrieved chunks into a context.
Sends the context and user question to the Gemini model.
Instructs the model to answer only using the provided context.
Returns the generated answer.
Assigns the confidence score.

The generation prompt instructs the model to respond with:

I cannot answer based on the provided document.

when the answer cannot be found in the provided context.

12. Grounding Behavior

The chatbot is designed to answer questions only using information available in
the retrieved document context.

For questions where the retrieved context does not provide sufficient
information, the chatbot returns:

I cannot answer based on the provided document.

For example:

Question:
Who won the 2022 FIFA World Cup?

Answer:
I cannot answer based on the provided document.

This demonstrates the out-of-document grounding behavior of the system.

13. Benchmark Testing

The project includes five benchmark queries based on the assignment requirements.

Query 1
What is Agentic AI according to the eBook?

Result:

According to the provided document, Agentic AI is defined as AI systems
capable of autonomous decision-making, with the capabilities to learn and
adapt to new situations.

Status:

HTTP 200
Confidence: 0.95
Query 2
How do AI agents differ from traditional automation systems?

Result:

The system explains the distinction between traditional rule-based
automation/RPA and Agentic AI, including autonomy, adaptation, handling
unstructured inputs, and goal-driven actions.

Status:

HTTP 200
Confidence: 0.95
Query 3
What are the core components of an Agentic Architecture?

Result:

I cannot answer based on the provided document.

Status:

HTTP 200
Confidence: 0.95

The system returned the refusal because the retrieved context did not provide
sufficient information to generate a grounded answer for this specific query.

Query 4
What role does memory play in Agentic AI workflows?

Result:

The system explains the roles of long-term memory and short-term memory in
Agentic AI workflows.

Long-term memory stores previous interactions, successful approaches, and
human demonstrations. Short-term memory represents the agent's current
conversation context.

Status:

HTTP 200
Confidence: 0.95
Query 5
Who won the 2022 FIFA World Cup?

Result:

I cannot answer based on the provided document.

Status:

HTTP 200
Confidence: 0.95

This is an out-of-document validation query. The chatbot does not use external
knowledge to answer the question.

Running the Tests

Start the FastAPI server:

python -m uvicorn app:app --reload

Then open another terminal, activate the virtual environment, and run:

python tests_sample_queries.py
14. Why RAG?

Retrieval-Augmented Generation combines document retrieval with language
generation.

Instead of allowing the language model to answer entirely from its general
knowledge, this application first retrieves relevant information from the
Agentic AI eBook.

The workflow is:

User Question
      |
      v
Retrieve relevant document chunks
      |
      v
Provide retrieved chunks as context
      |
      v
Generate answer using the context

This helps keep responses grounded in the provided Agentic AI eBook.

15. Model Implementation Note

The original assignment reference specifies OpenAI models for embeddings and
LLM generation.

During implementation, the OpenAI API could not be used because the available
API account had insufficient quota.

Therefore, this implementation uses Google Gemini for:

Text embeddings
LLM response generation

Pinecone remains the vector database and LangGraph remains the workflow
orchestration layer.

The overall RAG workflow remains:

Document
   |
   v
Chunking
   |
   v
Embeddings
   |
   v
Pinecone
   |
   v
LangGraph Retrieval
   |
   v
LLM Generation
   |
   v
FastAPI Response
16. Security

API credentials are stored in environment variables.

The following files should not be committed to GitHub:

.env
venv/
__pycache__/
*.pyc

The repository contains .env.example with placeholder values instead of
actual credentials.

17. Complete Setup and Execution
Step 1: Activate the virtual environment
venv\Scripts\activate
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Configure environment variables

Create .env:

GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
Step 4: Run document ingestion
python -m src.ingestion
Step 5: Start FastAPI
python -m uvicorn app:app --reload
Step 6: Open Swagger UI

Open:

http://127.0.0.1:8000/docs
Step 7: Run benchmark tests

Open another terminal, activate the virtual environment, and run:

python tests_sample_queries.py
18. Conclusion

This project demonstrates a complete document-grounded RAG pipeline using:

Agentic AI eBook
      |
      v
PyPDFLoader
      |
      v
Text Chunking
      |
      v
Gemini Embeddings
      |
      v
Pinecone
      |
      v
LangGraph Retrieval
      |
      v
Gemini Generation
      |
      v
FastAPI

The system provides answers based on the Agentic AI eBook, returns the
retrieved context and confidence score, and refuses to answer when the
required information is not available in the retrieved document context.