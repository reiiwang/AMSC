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



---

memsearch 的 embedding 系統設計在 embeddings/__init__.py 中，採用 EmbeddingProvider protocol 加上 lazy-loading factory，內建 provider 包含：openai、google、voyage、ollama、local、onnx。
當你用 vLLM 包成 OpenAI-Compatible 的 embedding server（/v1/embeddings），可以用 openai provider 指向自訂 endpoint：

```
# ~/.memsearch/config.toml
[embedding]
provider = "openai"
model = "your-model-name"      # vLLM serve 的模型名稱
api_base = "http://localhost:8000/v1"
api_key = "EMPTY"              # vLLM 不需要真實 key
```
or
```python
from memsearch import MemSearch
import os

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

mem = MemSearch(
    paths=["./memory"],
    embedding_provider="openai",
)
```
memsearch v0.2.3 的 changelog 裡有一條修正：fix: force encoding_format=float for OpenAI-compatible endpoints。 這代表舊版可能對 vLLM 有相容性問題（base64 encoding 報錯），請確保使用 v0.2.3 以上版本。


memsearch 的 compact.py 負責 LLM-powered 的 chunk summarization，支援 OpenAI / Anthropic / Gemini。 它並沒有內建 LangChain LLM interface。
但有幾種繞道方式：
方案 1：如果 LangChain LLM 也是 OpenAI-Compatible（例如用 ChatOpenAI 指向本地 vLLM），memsearch 的 compact 功能其實只需要一個能接受 prompt 的 HTTP 端點，所以可以：
```
[llm]
provider = "openai"
model = "your-llm-model"
api_base = "http://localhost:8000/v1"
api_key = "EMPTY"
```

方案 2：用 Python API 自行處理 LLM 部分
memsearch 的 LLM 只在 compact（記憶壓縮）時用到。如果你只是用 index() 和 search()，完全不需要 LLM。你可以自己拿 LangChain LLM 來處理 compact 邏輯：
```
from memsearch import MemSearch
from langchain_openai import ChatOpenAI  # 或任何 LangChain LLM

# memsearch 只負責 embedding + search
mem = MemSearch(paths=["./memory"], embedding_provider="openai")
results = await mem.search("your query")

# LangChain LLM 自己處理 summarization
llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", model="...")
summary = llm.invoke(f"Summarize these memories: {results}")
```


```
# embedding: 指向 vLLM embedding server
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8000/v1"

# LLM: compact 用的 chat model（可以是同一個或另一個 endpoint）
# memsearch 的 compact 預設讀 OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
export OPENAI_API_KEY="EMPTY"
```

```
import asyncio
from datetime import date
from pathlib import Path
from openai import OpenAI
from memsearch import MemSearch

MEMORY_DIR = "./memory"

# LLM client — 指向你的 vLLM 或 LangChain LLM server
llm = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",  # LLM endpoint（可與 embedding 不同 port）
)

# MemSearch — embedding 走 OPENAI_API_BASE 環境變數
mem = MemSearch(
    paths=[MEMORY_DIR],
    embedding_provider="openai",      # 使用 openai provider
    embedding_model="your-embed-model-name",  # vLLM serve 的模型名稱
)

def ingest(content: str):
    """寫入記憶到 markdown，然後 index"""
    p = Path(MEMORY_DIR) / f"{date.today()}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"\n{content}\n")

async def main():
    # ── Ingest ──────────────────────────────────────────────
    ingest("## 專案決策\n- 使用 PostgreSQL 作為主資料庫\n- Redis 做 cache layer")
    ingest("## 團隊\n- Alice 負責前端\n- Bob 負責後端 API")
    
    await mem.index()  # chunk → embed → upsert to Milvus
    print("✅ Indexed")

    # ── Search ──────────────────────────────────────────────
    query = "我們用什麼資料庫？"
    results = await mem.search(query, top_k=3)
    
    context = "\n".join(f"- {r['content'][:300]}" for r in results)
    print(f"\n🔍 搜尋結果：\n{context}")

    # ── 用 LLM 回答（LangChain / vLLM 都適用）────────────────
    resp = llm.chat.completions.create(
        model="your-llm-model-name",
        messages=[
            {"role": "system", "content": f"你有以下記憶：\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    print(f"\n🤖 回答：{resp.choices[0].message.content}")

asyncio.run(main())
```


在專案根目錄建立 .memsearch.toml：
```
# .memsearch.toml

[embedding]
provider = "openai"
model = "your-embed-model-name"   # 對應 vLLM serve 的模型名稱
# api_base 由環境變數 OPENAI_API_BASE 讀入
# 若要寫死在 config：（需 v0.1.15+ 的 configurable OpenAI endpoint 功能）
# api_base = "http://localhost:8000/v1"

[milvus]
uri = "~/.memsearch/milvus.db"    # 預設 Milvus Lite（本機單檔）
```
然後 Python 端就不需要額外設定：
```
mem = MemSearch(paths=["./memory"])  # 自動讀 .memsearch.toml
```

LLM 是 LangChain 物件












