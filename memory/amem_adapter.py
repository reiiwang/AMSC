"""A-MEM adapter: Agentic memory inspired by Zettelkasten note-taking method.

Design based on:
  - "A-MEM: Agentic Memory for LLM Agents" (Xu et al., NeurIPS 2025)
    https://arxiv.org/abs/2502.12110
  - agiresearch/A-mem GitHub: https://github.com/agiresearch/A-mem

Key design difference from other adapters:
  - Each memory is a structured NOTE with keywords, tags, context, and connections
  - Adding a new note triggers LLM to analyze existing memories and build links
  - Linked memories have their context UPDATED to reflect the new connection (memory evolution)
  - Retrieval uses vector search + graph traversal along links

Persistence:
  AgenticMemorySystem uses an in-memory ChromaDB client (no persistence by default).
  We persist notes as JSON under .amem_store/<user_id>.json and re-load them into
  a fresh AgenticMemorySystem on startup — similar to LangMem's JSON reload pattern.
  Note: this skips re-running the LLM evolution on reload (memories are loaded as-is).
"""
import json
from datetime import datetime
from pathlib import Path

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote

from memory.base import BaseMemory

STORE_DIR = Path(".amem_store")
TOP_K = 5


def _user_file(user_id: str) -> Path:
    STORE_DIR.mkdir(exist_ok=True)
    return STORE_DIR / f"{user_id}.json"


def _load_notes(user_id: str) -> list[dict]:
    f = _user_file(user_id)
    if not f.exists():
        return []
    return json.loads(f.read_text())


def _dump_notes(user_id: str, notes: list[dict]) -> None:
    _user_file(user_id).write_text(
        json.dumps(notes, ensure_ascii=False, indent=2)
    )


def _note_to_dict(note: MemoryNote) -> dict:
    return {
        "id": note.id,
        "content": note.content,
        "context": note.context,
        "keywords": note.keywords,
        "tags": note.tags,
        "category": note.category,
        "links": note.links,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "retrieval_count": note.retrieval_count,
        "evolution_history": note.evolution_history,
    }


class AMemAdapter(BaseMemory):
    """
    Memory notes are structured with: content, context, keywords, tags, connections.
    New notes trigger LLM to build links and evolve existing notes' context.
    """

    def __init__(self):
        self._systems: dict[str, AgenticMemorySystem] = {}

    def _get_system(self, user_id: str) -> AgenticMemorySystem:
        """Get or create an AgenticMemorySystem for this user, loading persisted notes."""
        if user_id not in self._systems:
            system = AgenticMemorySystem(
                model_name="all-MiniLM-L6-v2",
                llm_backend="openai",
                llm_model="gpt-4o-mini",
            )
            # Reload persisted notes directly into system (skip re-running LLM evolution)
            for note_dict in _load_notes(user_id):
                note = MemoryNote(
                    content=note_dict["content"],
                    id=note_dict["id"],
                    keywords=note_dict.get("keywords", []),
                    links=note_dict.get("links", []),
                    context=note_dict.get("context", ""),
                    category=note_dict.get("category", "Uncategorized"),
                    tags=note_dict.get("tags", []),
                    timestamp=note_dict.get("timestamp", ""),
                    last_accessed=note_dict.get("last_accessed", ""),
                    retrieval_count=note_dict.get("retrieval_count", 0),
                    evolution_history=note_dict.get("evolution_history", []),
                )
                system.memories[note.id] = note
                # Re-embed into ChromaDB
                system.retriever.add_document(
                    document=note.content,
                    metadata={
                        "content": note.content,
                        "context": note.context,
                        "keywords": note.keywords,
                        "tags": note.tags,
                        "category": note.category,
                        "timestamp": note.timestamp,
                    },
                    doc_id=note.id,
                )
            self._systems[user_id] = system
        return self._systems[user_id]

    def save(self, user_id: str, messages: list[dict]) -> None:
        """Convert conversation turn into A-MEM notes."""
        # Combine human + ai turns into a single content string per exchange
        human_msgs = [m for m in messages if m.get("role") == "human"]
        ai_msgs = [m for m in messages if m.get("role") in ("ai", "assistant")]
        if not human_msgs or not ai_msgs:
            return

        system = self._get_system(user_id)
        persisted = _load_notes(user_id)
        existing_ids = {n["id"] for n in persisted}

        for human, ai in zip(human_msgs, ai_msgs):
            content = (
                f"用戶：{human['content']}\n"
                f"顧問：{ai['content']}"
            )
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            # Call analyze_content first to extract keywords/tags/context via LLM
            analysis = system.analyze_content(content)
            # add_note triggers: link analysis + memory evolution (process_memory)
            note_id = system.add_note(
                content=content,
                time=timestamp,
                keywords=analysis.get("keywords", []),
                context=analysis.get("context", "General"),
                tags=analysis.get("tags", []),
            )

            if note_id and note_id not in existing_ids:
                note = system.memories.get(note_id)
                if note:
                    persisted.append(_note_to_dict(note))
                    existing_ids.add(note_id)

        # Persist updated notes (including evolved context of existing notes)
        # Re-dump all notes since evolution may have updated older entries
        all_notes = [_note_to_dict(n) for n in system.memories.values()]
        _dump_notes(user_id, all_notes)

    def retrieve(self, user_id: str, query: str) -> str:
        if not query:
            return ""
        system = self._get_system(user_id)
        if not system.memories:
            return ""

        results = system.search_agentic(query, k=TOP_K)
        if not results:
            return ""

        lines = []
        for r in results:
            content = r.get("content", "").strip()
            keywords = r.get("keywords", [])
            context = r.get("context", "")
            if not content:
                continue
            line = f"- {content}"
            if keywords:
                kw_str = ", ".join(keywords[:5])
                line += f"  [關鍵字: {kw_str}]"
            if context and context != "General":
                line += f"  [情境: {context}]"
            lines.append(line)

        return "\n".join(lines)
