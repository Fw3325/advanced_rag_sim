# Multi-query RAG
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

def rewrite_query(llm, user_query):
    system = SystemMessage(content="""
        You are an expert researcher. Your task is to rewrite the user's question
        into 3-5 research sub-queries to help retrieve more relevant evidence.

        Output strictly as a JSON list of strings:
        ["sub_query1", "sub_query2", "sub_query3"]
    """)

    user_msg = HumanMessage(content=f"Rewrite this question into sub-queries: {user_query}")

    response = llm.invoke([system, user_msg]).content.strip()
    cleaned_response = response.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("LLM returned invalid format:", response)
        return [user_query]  # fallback