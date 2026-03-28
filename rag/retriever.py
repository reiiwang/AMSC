"""Load the persisted ChromaDB collection and expose a retrieval function."""
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

CHROMA_PATH = Path(__file__).parent.parent / ".chroma"
COLLECTION_NAME = "health_knowledge"
TOP_K = 3


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(query: str, top_k: int = TOP_K) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    lines = [f"[{m['category']}] {m['title']}\n{doc}" for doc, m in zip(docs, metas)]
    return "\n\n---\n\n".join(lines)
