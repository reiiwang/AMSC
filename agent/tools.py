"""Tools available to the health advisor agent."""
import json
from pathlib import Path

from langchain_core.tools import tool

from rag.retriever import retrieve as rag_retrieve

_MOCK_DATA_PATH = Path(__file__).parent.parent / "data" / "mock_conversations.json"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the health knowledge base for information about medications,
    diseases, lab results, or lifestyle topics. Use this when the user asks
    about specific medical terms, drug side effects, or health management tips."""
    return rag_retrieve(query)


@tool
def get_user_health_profile(user_id: str) -> str:
    """Retrieve the health profile and past conversation history for a given user.
    Use this to recall the user's known conditions, medications, and previous concerns."""
    data = json.loads(_MOCK_DATA_PATH.read_text())
    user = next((u for u in data["users"] if u["user_id"] == user_id), None)
    if not user:
        return f"No profile found for user_id: {user_id}"
    profile = user["profile"]
    return (
        f"用戶：{user['name']}，年齡：{profile['age']}，"
        f"已知病症：{', '.join(profile['conditions'])}"
    )


TOOLS = [search_knowledge_base, get_user_health_profile]
