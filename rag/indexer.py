"""One-time indexing script: embed knowledge_base.json into ChromaDB."""
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

CHROMA_PATH = Path(__file__).parent.parent / ".chroma"
COLLECTION_NAME = "health_knowledge"
KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"


def build_index() -> None:
    docs = json.loads(KB_PATH.read_text())

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # drop and recreate to allow re-indexing cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=[f"{doc['title']}\n{doc['content']}" for doc in docs],
        metadatas=[{"category": doc["category"], "title": doc["title"]} for doc in docs],
    )

    print(f"Indexed {len(docs)} documents into {CHROMA_PATH}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    build_index()
