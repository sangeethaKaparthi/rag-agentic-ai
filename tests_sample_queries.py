import requests

BASE_URL = "http://127.0.0.1:8000/chat"

queries = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]

for query in queries:
    response = requests.post(
        BASE_URL,
        json={"query": query}
    )

    print("\n" + "=" * 80)
    print("QUESTION:", query)
    print("STATUS:", response.status_code)

    data = response.json()

    print("ANSWER:", data["final_answer"])
    print("CONFIDENCE:", data["confidence_score"])