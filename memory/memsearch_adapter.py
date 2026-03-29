"""memsearch adapter: Markdown-first memory inspired by OpenClaw.

Each user's memory is stored as .md files under .memsearch_store/<user_id>/
and indexed into a per-user Milvus Lite .db file.

Architecture:
  save()     → write conversation to .md file → re-index
  retrieve() → hybrid vector + BM25 search
"""
import asyncio
from datetime import datetime
from pathlib import Path

from memsearch import MemSearch

from memory.base import BaseMemory

STORE_DIR = Path(".memsearch_store")
TOP_K = 5


def _user_dir(user_id: str) -> Path:
    d = STORE_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _milvus_uri(user_id: str) -> str:
    return str(STORE_DIR / user_id / "milvus.db")


def _get_mem(user_id: str) -> MemSearch:
    return MemSearch(
        paths=[str(_user_dir(user_id))],
        embedding_provider="openai",
        milvus_uri=_milvus_uri(user_id),
        collection=f"memsearch_{user_id}",
    )


class MemsearchAdapter(BaseMemory):
    def save(self, user_id: str, messages: list[dict]) -> None:
        human_msgs = [m for m in messages if m.get("role") == "human"]
        ai_msgs = [m for m in messages if m.get("role") in ("ai", "assistant")]
        if not human_msgs or not ai_msgs:
            return

        # pair up turns and write as markdown
        turns = list(zip(human_msgs, ai_msgs))
        lines = [f"# Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        for human, ai in turns:
            lines.append(f"**User:** {human['content']}\n")
            lines.append(f"**Assistant:** {ai['content']}\n\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = _user_dir(user_id) / f"{timestamp}.md"
        md_file.write_text("".join(lines), encoding="utf-8")

        async def _index():
            mem = _get_mem(user_id)
            await mem.index()

        asyncio.run(_index())

    def retrieve(self, user_id: str, query: str) -> str:
        if not query:
            return ""

        user_dir = _user_dir(user_id)
        if not any(user_dir.glob("*.md")):
            return ""

        async def _search():
            mem = _get_mem(user_id)
            await mem.index()
            return await mem.search(query, top_k=TOP_K)

        results = asyncio.run(_search())
        if not results:
            return ""

        lines = [r["content"].strip() for r in results if r.get("content")]
        return "\n".join(f"- {line}" for line in lines)
