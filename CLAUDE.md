# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
make install          # install dependencies via uv sync
make index            # one-time: embed knowledge_base.json into ChromaDB (.chroma/)
make test             # run base agent smoke test
make test-langmem     # run LangMem adapter smoke test
make test-mem0        # run Mem0 adapter smoke test
make test-all         # run all three smoke tests

make chat-dummy       # interactive chat — no memory
make chat-langmem     # interactive chat — LangMem memory
make chat-mem0        # interactive chat — Mem0 memory

make graph-mermaid    # print LangGraph Mermaid diagram
make graph-png        # render graph.png and open it

make clean-all        # remove all local stores (.chroma/, .langmem_store/, .mem0_store/)
```

Required env vars (copy `.env.example` → `.env`):
- `OPENAI_API_KEY` — used for LLM (gpt-4o-mini) and all embeddings
- `ANTHROPIC_API_KEY` — reserved for LLM-as-judge (Phase 4, not yet implemented)

## Architecture

This is a learning project comparing two agent memory frameworks (**LangMem** vs **Mem0**) using a fixed LangGraph agent as the base. The scenario is a personal health advisor chatbot.

### Graph flow (`agent/graph.py`)

```
START → retrieve_memory → agent → [tool_calls?]
                                     ├─ YES → tools → agent (loop, max 3×)
                                     └─ NO  → save_memory → END
```

- `retrieve_memory`: fetches long-term memories from the active adapter using the latest human message as query
- `agent`: calls `gpt-4o-mini` with two context layers — recent 5 messages (short-term) + memory context injected into system prompt (long-term)
- `tools`: LangGraph `ToolNode` executes `search_knowledge_base` (RAG) or `get_user_health_profile`
- `save_memory`: persists the full conversation to the active memory adapter

The graph is built by `build_graph(memory: BaseMemory, user_id: str)`. Swapping the memory adapter is the only change needed to compare frameworks.

### Memory adapters (`memory/`)

All adapters implement `BaseMemory` (`base.py`) with two methods: `save(user_id, messages)` and `retrieve(user_id, query) -> str`.

| Adapter | Framework | Persistence |
|---------|-----------|-------------|
| `DummyMemory` | no-op | none |
| `LangMemAdapter` | LangMem + LangGraph InMemoryStore | `.langmem_store/<user_id>.json` |
| `Mem0Adapter` | mem0ai | `.mem0_store/` (ChromaDB + SQLite) |

LangMem requires `asyncio.run()` since `create_memory_manager` is async-first. On reload, JSON memories are loaded back into `InMemoryStore` — no re-embedding on startup.

### RAG layer (`rag/`)

- `make index` runs `rag/indexer.py` once to embed `data/knowledge_base.json` (12 health knowledge docs) into ChromaDB at `.chroma/` using `text-embedding-3-small`
- `rag/retriever.py` opens a `PersistentClient` on each call (stateless) and returns top-3 results
- This is separate from Mem0's store (`.mem0_store/`) — they use different ChromaDB paths and collections

### Known gotcha: ToolMessage ordering

`trim_messages` with `strategy="last"` can split a tool-call sequence, leaving a `ToolMessage` without its preceding `AIMessage(tool_calls=...)`. OpenAI rejects this with a 400 error. Fixed in `call_agent` by stripping leading tool messages after trimming.

### Notes

- `notes/agent_design.md` — full agent architecture, node logic, design decisions
- `notes/langmem_and_mem0_integration.md` — API exploration, data formats, gotchas for both frameworks
