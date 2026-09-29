from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_pinecone import PineconeVectorStore

from src.config import (
    GOOGLE_API_KEY,
    PINECONE_INDEX_NAME
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536
    )

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    def retrieve_node(state: AgentState):

        docs = retriever.invoke(
            state["question"]
        )

        context_texts = [
            doc.page_content
            for doc in docs
        ]

        return {
            "context": context_texts
        }

    def generate_node(state: AgentState):

        context_str = "\n\n".join(
            state["context"]
        )

        prompt = f"""
You are a strict document-based assistant.

Answer the question ONLY using the information
contained in the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
respond exactly:

I cannot answer based on the provided document.

Context:
{context_str}

Question:
{state["question"]}
"""

        response = llm.invoke(prompt)

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
            )

        if state["context"]:
            confidence = 0.95
        else:
            confidence = 0.0

        return {
            "answer": answer,
            "score": confidence
        }

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()