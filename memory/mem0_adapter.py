"""Mem0 adapter using ChromaDB for persistence, separate from RAG's .chroma/ store."""
from mem0 import Memory

from memory.base import BaseMemory

TOP_K = 5

_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_health_memories",
            "path": ".mem0_store",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
    "history_db_path": ".mem0_store/history.db",
}


class Mem0Adapter(BaseMemory):
    def __init__(self):
        self._mem = Memory.from_config(_CONFIG)

    def save(self, user_id: str, messages: list[dict]) -> None:
        # mem0 expects [{"role": "user"|"assistant", "content": "..."}]
        normalized = []
        for m in messages:
            role = m.get("role", "")
            if role == "human":
                role = "user"
            elif role in ("ai",):
                role = "assistant"
            if role in ("user", "assistant"):
                normalized.append({"role": role, "content": m.get("content", "")})

        if normalized:
            self._mem.add(normalized, user_id=user_id)

    def retrieve(self, user_id: str, query: str) -> str:
        if not query:
            return ""
        results = self._mem.search(query, user_id=user_id, limit=TOP_K)
        memories = results.get("results", [])
        if not memories:
            return ""
        return "\n".join(f"- {m['memory']}" for m in memories)
