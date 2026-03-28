.PHONY: install index test graph-mermaid graph-png

install:
	uv sync

index:
	uv run python -m rag.indexer

test:
	uv run python test_graph.py

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
