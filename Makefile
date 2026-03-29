.PHONY: install index test test-langmem test-mem0 test-all \
        graph-mermaid graph-png \
        clean-chroma clean-langmem clean-mem0 clean-all \
        chat-dummy chat-langmem chat-mem0 \
        inspect-langmem inspect-mem0

# ── 環境 ──────────────────────────────────────────────
install:
	uv sync

# ── RAG 知識庫 ────────────────────────────────────────
# 第一次使用，或 knowledge_base.json 有更新時執行
index:
	uv run python -m rag.indexer

# ── 測試 ──────────────────────────────────────────────
test:
	uv run python test_graph.py

test-langmem:
	uv run python test_langmem.py

test-mem0:
	uv run python test_mem0.py

test-all: test test-langmem test-mem0

# ── Graph 視覺化 ──────────────────────────────────────
graph-mermaid:
	uv run python -c "\
from agent.graph import build_graph; \
graph = build_graph(); \
print(graph.get_graph().draw_mermaid())"

graph-png:
	uv run python -c "\
from agent.graph import build_graph; \
graph = build_graph(); \
open('graph.png', 'wb').write(graph.get_graph().draw_mermaid_png())"
	open graph.png

# ── 互動對話（切換 memory adapter）──────────────────
chat-dummy:
	uv run python scripts/chat_loop.py dummy

chat-langmem:
	uv run python scripts/chat_loop.py langmem

chat-mem0:
	uv run python scripts/chat_loop.py mem0

# ── Memory 查詢 ──────────────────────────────────────
# 用法: make inspect-langmem USER=user_001
#       make inspect-mem0    USER=user_001
USER ?= dev

inspect-langmem:
	uv run python scripts/inspect_memory.py langmem $(USER)

inspect-mem0:
	uv run python scripts/inspect_memory.py mem0 $(USER)

# ── 清除 store ────────────────────────────────────────
clean-chroma:
	rm -rf .chroma/

clean-langmem:
	rm -rf .langmem_store/

clean-mem0:
	rm -rf .mem0_store/

clean-all: clean-chroma clean-langmem clean-mem0
	@echo "All stores cleared."
