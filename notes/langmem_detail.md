# LangMem 整合筆記

## 概覽

LangMem 是 LangChain 推出的長期記憶 SDK，負責從對話中**萃取、更新、檢索記憶**，本身不帶 LLM 或 Embedding，需要外接。

-----

## 核心元件

|元件                           |說明                                  |
|-----------------------------|------------------------------------|
|`create_memory_manager`      |純萃取層，不綁 store，手動控制存取                |
|`create_memory_store_manager`|整合 BaseStore，自動搜尋、萃取、寫入             |
|`create_manage_memory_tool`  |給 agent 用的工具，讓 agent 主動存記憶（hot path）|
|`create_search_memory_tool`  |給 agent 用的工具，讓 agent 主動搜記憶（hot path）|

-----

## Store 選擇

|Store                                 |用途        |持久化   |
|--------------------------------------|----------|------|
|`InMemoryStore`                       |開發 / 測試   |❌ 重啟消失|
|`AsyncPostgresStore` / `PostgresStore`|Production|✅     |


> 官方建議 production 使用 `AsyncPostgresStore`。

-----

## `embed` 參數格式

`InMemoryStore` 和 `PostgresStore` 的 `index` 裡的 `embed` 支援三種格式：

### 1. 字串（`"provider:model"`）

底層透過 LangChain `init_embeddings` 解析。

```python
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
```

### 2. LangChain `Embeddings` 物件（推薦）

任何繼承 `langchain_core.embeddings.Embeddings` 的物件都可以直接傳入。

```python
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": emb,  # 直接傳物件
    }
)
```

vLLM 包成 LangChain 格式後同樣適用：

```python
store = InMemoryStore(
    index={
        "dims": 1024,         # 依你的 model 調整
        "embed": your_vllm_embeddings,
    }
)
```

### 3. 自訂 callable

簽名為 `(list[str]) -> list[list[float]]`。

```python
def embed(texts: list[str]) -> list[list[float]]:
    return your_model.encode(texts)

store = InMemoryStore(
    index={"dims": 1024, "embed": embed}
)
```

> 若原本是 async，包成 sync function 再傳入。

### OpenAI Embedding 維度對照

|model                 |dims|
|----------------------|----|
|text-embedding-3-small|1536|
|text-embedding-3-large|3072|
|text-embedding-ada-002|1536|

-----

## LLM 參數格式

`create_memory_manager` / `create_memory_store_manager` 的第一個參數同樣支援字串或 LangChain 物件：

```python
# 字串
manager = create_memory_store_manager("openai:gpt-4o-mini", ...)

# LangChain BaseChatModel 物件
manager = create_memory_store_manager(your_lc_llm, ...)
```

-----

## ChromaDB 的限制

ChromaDB 是 LangChain 的 `VectorStore`，**不是** LangGraph 的 `BaseStore`，無法直接傳給 LangMem 的 store 層。直接傳會報錯：

```
Input should be an instance of BaseStore
```

**替代方案**：用 LangMem 萃取記憶，再手動寫入 Chroma：

```python
# LangMem 只負責萃取
memories = await extractor.ainvoke({"messages": messages})

# 自己寫進 Chroma
chroma_store.add_texts(
    texts=[str(mem) for _, mem in memories],
    metadatas=[{"user_id": user_id}],
)
```

-----

## PostgreSQL 設定

### Docker（使用 pgvector image）

```bash
# pgvector/pgvector 而非普通 postgres，因為需要 vector 擴充套件
docker run -d \
  --name langmem-postgres \
  -e POSTGRES_USER=langmem \
  -e POSTGRES_PASSWORD=langmem \
  -e POSTGRES_DB=langmem \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### docker-compose（推薦，有 volume 持久化）

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: langmem
      POSTGRES_PASSWORD: langmem
      POSTGRES_DB: langmem
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Python 使用

```python
from langgraph.store.postgres import AsyncPostgresStore

async with AsyncPostgresStore.from_conn_string(
    "postgresql://langmem:langmem@localhost:5432/langmem",
    index={
        "dims": 1536,
        "embed": emb,   # LangChain Embeddings 物件
    }
) as store:
    await store.setup()  # 第一次執行，建立 schema
```

-----

## 推薦架構：獨立 memory.py

Agent 只呼叫 function，不直接碰 LangMem / Store 細節。

```python
# memory.py
from langchain_openai import OpenAIEmbeddings
from langmem import create_memory_store_manager
from langgraph.store.postgres import AsyncPostgresStore

emb = OpenAIEmbeddings(model="text-embedding-3-small")

store = AsyncPostgresStore.from_conn_string(
    "postgresql://langmem:langmem@localhost:5432/langmem",
    index={"dims": 1536, "embed": emb}
)

manager = create_memory_store_manager(
    your_lc_llm,
    namespace=("memories", "{user_id}"),
)

async def save_memory(messages: list, user_id: str):
    config = {"configurable": {"user_id": user_id}}
    await manager.ainvoke({"messages": messages}, config=config)

async def search_memory(query: str, user_id: str, limit: int = 5) -> list:
    config = {"configurable": {"user_id": user_id}}
    results = await manager.asearch(query=query, config=config, limit=limit)
    return [r.value for r in results]
```

```python
# agent.py — 只 import function，不知道 LangMem 細節
from memory import save_memory, search_memory

async def agent_node(state, config):
    user_id = config["configurable"]["user_id"]
    memories = await search_memory(state["messages"][-1].content, user_id)
    # 注入 system prompt，呼叫 LLM ...

async def after_turn(messages, user_id):
    await save_memory(messages, user_id)
```

-----

## 現有 code（InMemoryStore + JSON）的運作方式

你分享的 `LangMemAdapter` 採用「自製持久化」方案：

```
InMemoryStore（記憶體，供 LangMem 操作）
     ↕ 首次存取 lazy load / 每次存後 dump
JSON 檔案（磁碟，.langmem_store/{user_id}.json）
```

|優點               |缺點                            |
|-----------------|------------------------------|
|不需要額外服務          |JSON 不適合大量資料                  |
|架構簡單             |並發寫同一 user 有 race condition 風險|
|完整用到 LangMem 萃取邏輯|重啟需重新 load 進記憶體               |

換成 `AsyncPostgresStore` 後，`_ensure_loaded`、`_load`、`_dump`、`STORE_DIR` 等持久化邏輯全部可以刪除。

-----

## memsearch（Milvus / Zilliz 出品）

### 概覽

memsearch 是 Zilliz（Milvus 母公司）開源的 agent 記憶庫，從 OpenClaw 的記憶系統獨立抽出，MIT 授權。

### 與 LangMem 的核心差異

|      |LangMem                   |memsearch                            |
|------|--------------------------|-------------------------------------|
|記憶格式  |JSON in DB（不透明）           |**Markdown 檔案**（人可讀、可 git）           |
|向量 DB |InMemoryStore / PostgreSQL|**Milvus**（Lite / Standalone / Cloud）|
|記憶萃取  |LLM 自動萃取結構化記憶             |LLM 摘要對話 → 寫成 `.md`                  |
|搜尋方式  |純 dense vector            |**Hybrid（dense + BM25 + RRF）**       |
|整合方式  |依賴 LangGraph BaseStore    |**完全獨立，任何 framework 都能用**            |
|主要設計對象|通用 agent                  |以 coding agent 為主                    |

### 安裝

```bash
pip install "memsearch[onnx]"   # 本地 ONNX embedding，不需 API key
# 或
pip install memsearch            # 使用 OpenAI embedding
```

### Embedding Provider 選項

|Provider|安裝                 |預設 model                          |
|--------|-------------------|----------------------------------|
|ONNX（預設）|`memsearch[onnx]`  |`bge-m3-onnx-int8`（CPU，不需 API key）|
|OpenAI  |`memsearch`（已包含）   |`text-embedding-3-small`          |
|Google  |`memsearch[google]`|`gemini-embedding-001`            |
|Voyage  |`memsearch[voyage]`|`voyage-3-lite`                   |
|Ollama  |`memsearch[ollama]`|`nomic-embed-text`                |
|Local   |`memsearch[local]` |`all-MiniLM-L6-v2`                |

### Milvus 後端選擇

|情境        |milvus_uri                   |說明                                  |
|----------|-----------------------------|------------------------------------|
|POC / 本地開發|不傳（預設）                       |Milvus Lite，本地 `.db` 檔，**不需 Docker**|
|自架多 agent |`http://localhost:19530`     |Milvus Standalone                   |
|Production|`https://xxx.zillizcloud.com`|Zilliz Cloud，全託管                    |


> ⚠️ Milvus Lite 僅支援 Linux / macOS，Windows 不支援。

### Python API

```python
from memsearch import MemSearch

mem = MemSearch(
    paths=["./memory"],
    embedding_provider="openai",   # 或 "local"（不需 API key）
    collection="memsearch_chunks", # 不同 user 用不同 collection 隔離
    # milvus_uri 不傳 → 預設 Milvus Lite
)

await mem.index()                                   # index markdown 檔
results = await mem.search("Redis config", top_k=3) # hybrid 語意搜尋
await mem.compact(llm_provider="openai")            # LLM 壓縮舊記憶
mem.close()
```

### 推薦架構：獨立 memory.py（memsearch 版）

```python
# memory.py
import datetime
from pathlib import Path
from memsearch import MemSearch

MEMORY_DIR = Path(".memsearch/memory")

def _get_mem(user_id: str) -> MemSearch:
    return MemSearch(
        paths=[str(MEMORY_DIR / user_id)],
        embedding_provider="openai",
        collection=f"memory_{user_id}",  # user 隔離
        # milvus_uri 不傳 → Milvus Lite
    )

async def save_memory(messages: list[dict], user_id: str) -> None:
    """把對話 append 進 markdown，再 index 進 Milvus"""
    today = datetime.date.today().isoformat()
    md_path = MEMORY_DIR / user_id / f"{today}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## Session {datetime.datetime.now().isoformat()}\n")
        for m in messages:
            f.write(f"**{m.get('role','')}**: {m.get('content','')}\n\n")

    mem = _get_mem(user_id)
    await mem.index()
    mem.close()

async def search_memory(query: str, user_id: str, top_k: int = 5) -> str:
    mem = _get_mem(user_id)
    results = await mem.search(query, top_k=top_k)
    mem.close()
    return "\n".join(f"- {r['content'].strip()}" for r in results if r.get("content"))
```

```python
# agent.py — 呼叫方式與 LangMem 版完全相同
from memory import save_memory, search_memory

async def agent_node(state, config):
    user_id = config["configurable"]["user_id"]
    memories = await search_memory(state["messages"][-1].content, user_id)
    # 注入 system prompt，呼叫 LLM ...

async def after_turn(messages, user_id):
    await save_memory(messages, user_id)
```

### MemSearch 建構子參數

|參數                  |預設值                     |說明                                    |
|--------------------|------------------------|--------------------------------------|
|`paths`             |`[]`                    |要 index 的 markdown 目錄或檔案              |
|`embedding_provider`|`"openai"`              |embedding 後端                          |
|`embedding_model`   |`None`                  |覆寫預設 model                            |
|`embedding_base_url`|`None`                  |OpenAI-compatible 自訂 endpoint（vLLM 可用）|
|`embedding_api_key` |`None`                  |API key                               |
|`milvus_uri`        |`~/.memsearch/milvus.db`|Milvus 連線 URI                         |
|`collection`        |`"memsearch_chunks"`    |Milvus collection 名稱                  |
|`max_chunk_size`    |`1500`                  |每個 chunk 最大字元數                        |


> 使用自訂 vLLM embedding endpoint：設定 `embedding_base_url` 指向你的 vLLM server。