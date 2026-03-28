"""LangMem adapter using LangGraph InMemoryStore + LangMem memory manager,
with JSON file persistence under .langmem_store/."""
import asyncio
import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langmem import create_memory_manager
from langgraph.store.memory import InMemoryStore

from memory.base import BaseMemory

TOP_K = 5
NAMESPACE = ("memories",)
STORE_DIR = Path(".langmem_store")


def _get_embed_fn():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed(texts: list[str]) -> list[list[float]]:
        return embedder.embed_documents(texts)

    return embed


def _user_file(user_id: str) -> Path:
    STORE_DIR.mkdir(exist_ok=True)
    return STORE_DIR / f"{user_id}.json"


def _load(user_id: str) -> dict[str, str]:
    """Load persisted memories as {id: content}."""
    f = _user_file(user_id)
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def _dump(user_id: str, memories: dict[str, str]) -> None:
    _user_file(user_id).write_text(json.dumps(memories, ensure_ascii=False, indent=2))


class LangMemAdapter(BaseMemory):
    def __init__(self):
        self._store = InMemoryStore(
            index={"dims": 1536, "embed": _get_embed_fn()}
        )
        self._manager = create_memory_manager(
            "openai:gpt-4o-mini",
            instructions=(
                "你是一位健康顧問的記憶管理員。"
                "從對話中抽取用戶的健康狀況、用藥、症狀變化及重要事件，以繁體中文記錄。"
            ),
            enable_inserts=True,
            enable_updates=True,
            enable_deletes=False,
        )
        self._loaded: set[str] = set()

    def _namespace(self, user_id: str) -> tuple:
        return NAMESPACE + (user_id,)

    def _ensure_loaded(self, user_id: str) -> None:
        """Load persisted memories into InMemoryStore on first access."""
        if user_id in self._loaded:
            return
        namespace = self._namespace(user_id)
        for mem_id, content in _load(user_id).items():
            self._store.put(namespace, key=mem_id, value={"content": content})
        self._loaded.add(user_id)

    def save(self, user_id: str, messages: list[dict]) -> None:
        from langchain_core.messages import HumanMessage, AIMessage
        from langmem.knowledge.extraction import Memory as LangMemMemory

        lc_messages = []
        for m in messages:
            role, content = m.get("role", ""), m.get("content", "")
            if role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                lc_messages.append(AIMessage(content=content))

        if not lc_messages:
            return

        self._ensure_loaded(user_id)
        namespace = self._namespace(user_id)

        existing_items = self._store.search(namespace, limit=50)
        existing_memories = [
            (item.key, LangMemMemory(content=item.value.get("content", "")))
            for item in existing_items
        ]

        async def _run():
            return await self._manager.ainvoke(
                {"messages": lc_messages, "existing": existing_memories}
            )

        extracted = asyncio.run(_run())

        persisted = _load(user_id)
        for mem in extracted:
            content = mem.content
            if hasattr(content, "content"):
                content = content.content
            elif not isinstance(content, str):
                content = str(content)
            self._store.put(namespace, key=mem.id, value={"content": content})
            persisted[mem.id] = content

        _dump(user_id, persisted)

    def retrieve(self, user_id: str, query: str) -> str:
        if not query:
            return ""
        self._ensure_loaded(user_id)
        namespace = self._namespace(user_id)
        results = self._store.search(namespace, query=query, limit=TOP_K)
        lines = [item.value.get("content", "") for item in results if item.value.get("content")]
        return "\n".join(f"- {line}" for line in lines)
