"""MemGPT-inspired adapter: LLM self-manages its own memory via tool calls.

Design based on:
  - "MemGPT: Towards LLMs as Operating Systems" (Packer et al., UC Berkeley)
    via https://www.leoniemonigatti.com/papers/memgpt.html
  - Letta (MemGPT framework) implementation details
    via https://medium.com/@piyush.jhamb4u/stateful-ai-agents-a-deep-dive-into-letta-memgpt-memory-models-a2ffc01a7ea1

This is an independent implementation of the MemGPT philosophy.
It does NOT use the official letta/letta package (which requires a Docker server).

Key design difference from other adapters:
  - Memory is managed BY THE LLM itself via tool calls
  - save() is a no-op — the LLM decides when and what to remember
  - retrieve() returns the current core memory block (always in system prompt)
  - Two extra tools are injected into the agent: core_memory_append, core_memory_replace
  - Archival memory (ChromaDB) handles overflow beyond core memory char limit
"""
import json
import uuid
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.tools import tool

from memory.base import BaseMemory

STORE_DIR = Path(".memgpt_store")
CHROMA_PATH = STORE_DIR / "chroma"
CHAR_LIMIT = 2000  # core memory character limit per block

PERSONA = (
    "你是一位專業且富有同理心的健康顧問。"
    "你會記住用戶的健康狀況，並在每次對話中提供個人化的建議。"
    "當你學到關於用戶的重要資訊時，你會主動更新你的記憶。"
)


def _store_file(user_id: str) -> Path:
    STORE_DIR.mkdir(exist_ok=True)
    return STORE_DIR / f"{user_id}.json"


def _load_core(user_id: str) -> dict[str, str]:
    f = _store_file(user_id)
    if not f.exists():
        return {"persona": PERSONA, "human": "（尚無用戶資訊）"}
    return json.loads(f.read_text())


def _dump_core(user_id: str, blocks: dict[str, str]) -> None:
    _store_file(user_id).write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2)
    )


def _get_archival(user_id: str):
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    collection_name = f"memgpt_{user_id}"
    try:
        return client.get_collection(collection_name, embedding_function=embed_fn)
    except Exception:
        return client.create_collection(collection_name, embedding_function=embed_fn)


class MemGPTAdapter(BaseMemory):
    """
    Memory hierarchy:
      Tier 1 (in-context): core_memory blocks → always injected into system prompt
      Tier 2 (external):   archival_memory    → ChromaDB vector search on demand
    """

    def __init__(self, user_id: str = "default"):
        self._user_id = user_id
        self._core = _load_core(user_id)

    def _format_core(self) -> str:
        lines = ["## Core Memory"]
        for label, value in self._core.items():
            lines.append(f"[{label}]\n{value}")
        return "\n\n".join(lines)

    def retrieve(self, user_id: str, query: str) -> str:
        """Return core memory (always shown) + archival search results if query given."""
        self._user_id = user_id
        self._core = _load_core(user_id)
        context = self._format_core()

        if query:
            try:
                col = _get_archival(user_id)
                results = col.query(query_texts=[query], n_results=3)
                docs = results["documents"][0]
                if docs:
                    archival_text = "\n".join(f"- {d}" for d in docs)
                    context += f"\n\n## Archival Memory（相關召回）\n{archival_text}"
            except Exception:
                pass

        return context

    def save(self, user_id: str, messages: list[dict]) -> None:
        # No-op: in MemGPT, memory is managed by the LLM via tool calls.
        # Core memory is persisted immediately when tools are called.
        pass

    def get_tools(self) -> list:
        """Return memory management tools to be added to the agent's tool list."""
        adapter = self

        @tool
        def core_memory_append(label: str, content: str) -> str:
            """Append new information to a core memory block. Use this when you learn
            important new facts about the user (health conditions, medications, preferences).
            label must be 'human' or 'persona'. Core memory is always visible in your context."""
            if label not in adapter._core:
                return f"Error: unknown label '{label}'. Use 'human' or 'persona'."
            current = adapter._core[label]
            updated = (current + "\n" + content).strip()
            if len(updated) > CHAR_LIMIT:
                return (
                    f"Error: core memory block '{label}' would exceed {CHAR_LIMIT} chars. "
                    "Use archival_memory_insert instead."
                )
            adapter._core[label] = updated
            _dump_core(adapter._user_id, adapter._core)
            return f"Core memory [{label}] updated. ({len(updated)}/{CHAR_LIMIT} chars)"

        @tool
        def core_memory_replace(label: str, old_content: str, new_content: str) -> str:
            """Replace existing text in a core memory block. Use this to correct or update
            outdated information about the user.
            label must be 'human' or 'persona'."""
            if label not in adapter._core:
                return f"Error: unknown label '{label}'. Use 'human' or 'persona'."
            if old_content not in adapter._core[label]:
                return f"Error: text not found in block '{label}'."
            updated = adapter._core[label].replace(old_content, new_content, 1)
            if len(updated) > CHAR_LIMIT:
                return f"Error: replacement would exceed {CHAR_LIMIT} chars."
            adapter._core[label] = updated
            _dump_core(adapter._user_id, adapter._core)
            return f"Core memory [{label}] replaced. ({len(updated)}/{CHAR_LIMIT} chars)"

        @tool
        def archival_memory_insert(content: str) -> str:
            """Store information in archival memory (long-term, searchable via vector search).
            Use this for detailed notes, events, or information that doesn't fit in core memory."""
            col = _get_archival(adapter._user_id)
            col.add(ids=[str(uuid.uuid4())], documents=[content])
            return f"Stored in archival memory: '{content[:80]}...'"

        @tool
        def archival_memory_search(query: str) -> str:
            """Search archival memory for relevant past information using semantic search.
            Use this when you need to recall details that may have been stored previously."""
            col = _get_archival(adapter._user_id)
            results = col.query(query_texts=[query], n_results=3)
            docs = results["documents"][0]
            if not docs:
                return "No relevant archival memories found."
            return "\n".join(f"- {d}" for d in docs)

        return [core_memory_append, core_memory_replace,
                archival_memory_insert, archival_memory_search]
