

LangMem 並非完全獨立的記憶系統——它是擴展 LangGraph 內建 Store 的函式庫，底層儲存仍依賴 LangGraph 的 BaseStore。但你不需要用 LangGraph 建構 agent，可以把它當純粹的記憶工具使用。
https://vectorize.io/articles/langchain-memory-alternatives


1. 安裝
```bash
pip install langmem langgraph
export ANTHROPIC_API_KEY="sk-..."
```

2. 最簡單的獨立用法：create_memory_manager
create_memory_manager 讓你自己控制儲存和更新；create_memory_store_manager 則直接整合 BaseStore 處理搜尋、upsert 和刪除。兩者都透過提示 LLM 使用平行 tool calling 來提取新記憶、更新舊記憶。
https://langchain-ai.github.io/langmem/guides/extract_semantic_memories/
用 create_memory_manager 最單純，不需要 agent 或 graph：
```python
import asyncio
from langmem import create_memory_manager

manager = create_memory_manager("anthropic:claude-3-5-sonnet-latest")

conversation = [
    {"role": "user", "content": "我喜歡深色模式"},
    {"role": "assistant", "content": "好的，我記住了"},
]

# 從對話中提取記憶
memories = asyncio.run(manager(conversation))
print(memories[0][1])
# 輸出: "User prefers dark mode for all applications"
```

3. 搭配自訂 Schema（結構化記憶）
```python
from pydantic import BaseModel
from langmem import create_memory_manager

class PreferenceMemory(BaseModel):
    """儲存使用者偏好"""
    category: str
    preference: str
    context: str

manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[PreferenceMemory],
)
```

4. 帶儲存的完整獨立範例
使用 InMemoryStore 搭配 create_manage_memory_tool 和 create_search_memory_tool 可以建立有記憶能力的系統，不一定要用 create_react_agent，也可以加進現有的 agent 或自行打造記憶系統。
https://langchain-ai.github.io/langmem/
```python
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# 建立向量儲存
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# 建立記憶管理器，綁定 namespace
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories", "user_123"),
)

# 處理對話並自動存入 store
conversation = [
    {"role": "user", "content": "我叫小明，我喜歡 Python"},
    {"role": "assistant", "content": "好的！"},
]
manager.invoke({"messages": conversation})

# 搜尋記憶
results = store.search(("memories", "user_123"), query="程式語言")
print(results)
```

5. 持久化儲存（生產環境）
預設使用 InMemoryStore 時，記憶不會在 process 重啟後保留。若要真正的長期記憶，需要換成持久化的儲存方案，例如 PostgreSQL：
https://medium.com/@astropomeai/langmem-long-term-memory-for-ai-agents-366d7256ddce
```python
from langgraph.store.postgres import AsyncPostgresStore

store = await AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@host:5432/db"
)
```

---

一、InMemoryStore 接自訂 Embedding
InMemoryStore 的 embed 參數可以直接傳入 init_embeddings() 回傳的物件，而不只是字串格式。 
https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/persistence.md
這表示任何符合 LangChain Embeddings 介面的物件都能直接塞進去。

方法一：用 langchain-openai 包 vLLM
vLLM 提供 OpenAI 相容的 API，可以直接用 langchain_openai.OpenAIEmbeddings 指向它：
```python
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

# vLLM 啟動時指定 embedding 模型：
# vllm serve BAAI/bge-m3 --port 8000

vllm_embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url="http://localhost:8000/v1",
    api_key="dummy",           # vLLM 不驗 key，但欄位不能空
    check_embedding_ctx_length=False,
)

store = InMemoryStore(
    index={
        "dims": 1024,          # 依模型實際維度填
        "embed": vllm_embeddings,
    }
)
```

方法二：用 init_embeddings + 自訂 callable
embed 欄位可以是 embedding provider 字串，也可以是指向自訂函式的路徑（callable）。 
https://blog.langchain.com/semantic-search-for-langgraph-memory/
```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# 或者包一個 callable function
def my_embed_fn(texts: list[str]) -> list[list[float]]:
    # 呼叫 vLLM HTTP API 或任意邏輯
    ...
    return embeddings  # list of float vectors

store = InMemoryStore(
    index={
        "dims": 1024,
        "embed": my_embed_fn,
    }
)
```

二、create_memory_store_manager 接 LangChain LLM 物件
create_memory_store_manager 的 model 參數明確支援 str | BaseChatModel——可以傳字串，也可以傳 LangChain 的 BaseChatModel 實例。 
https://langchain-ai.github.io/langmem/reference/memory/

完整範例（vLLM LLM + vLLM Embedding）

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# 1. LLM：vLLM 包成 ChatOpenAI
vllm_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0,
)

# 2. Embedding：vLLM 包成 OpenAIEmbeddings
vllm_embed = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url="http://localhost:8001/v1",  # 可以同一個 port
    api_key="dummy",
    check_embedding_ctx_length=False,
)

# 3. Store
store = InMemoryStore(
    index={
        "dims": 1024,
        "embed": vllm_embed,
    }
)

# 4. Memory Manager 直接傳 BaseChatModel 實例
manager = create_memory_store_manager(
    vllm_llm,                        # ← 傳物件，不是字串
    namespace=("memories", "user_1"),
)
```

注意事項
vLLM 要支援 embedding 需要加 --runner pooling（或依模型不同）：
```bash
vllm serve BAAI/bge-m3 --runner pooling --port 8001
```
維度要對：dims 必須跟模型實際輸出維度一致，否則儲存時會報錯。可以用 vllm_embed.embed_query("test") 先量一下長度。
query_model 也可以換：如果要 memory manager 用不同模型搜尋，同樣傳 BaseChatModel：
```python
manager = create_memory_store_manager(
    vllm_llm,
    query_model=ChatOpenAI(model="smaller-model", ...),
    namespace=("memories", "user_1"),
)
```



- Ingestion 流程圖
```
對話內容
   ↓
[LLM] 分析對話，決定要記什麼 → 輸出純文字記憶（e.g. "用戶喜歡深色模式"）
   ↓
[Embedding] 把這段文字轉成向量
   ↓
[Store] 同時存 文字 + 向量
```
所以 ingestion 時 LLM 和 Embedding 都會用到，只是先後順序不同：LLM 先思考要存什麼，Embedding 再把結果向量化。

- InMemoryStore 存的是什麼
同時存兩份：原始文字（text）＋向量（embedding）。
```
Store 裡的每一筆記憶：
{
  "value": "用戶喜歡深色模式",    ← 原始文字
  "embedding": [0.12, -0.34, ...] ← 對應的向量（內部）
}
```
搜尋時用向量計算相似度，但回傳給你的是原始文字，向量是隱藏在內部的索引結構。
這也是為什麼 InMemoryStore 初始化時要同時給 dims（向量維度）和 embed（embedding 函式）——少了任一個就無法做相似度搜尋，只能做 key-value 的精確查找。



DataBase

不只 PostgreSQL，LangGraph 的 BaseStore 有幾種後端可以選：

- 官方支援的 Store

| Store | 用途 | 支援向量搜尋 |
|-------|------|------------|
| `InMemoryStore` | 開發/測試，重啟即消失 | ✅ |
| `AsyncPostgresStore` / `PostgresStore` | 生產環境推薦 | ✅ |
| `AsyncSqliteStore` / `SqliteStore` | 輕量本地持久化 | ✅ (較新版本) |

```python
# SQLite（本地檔案，不需要架資料庫）
from langgraph.store.sqlite import SqliteStore

store = SqliteStore(
    "memories.db",
    index={
        "dims": 1024,
        "embed": vllm_embed,
    }
)
```


- 自訂 Store

因為 LangMem 只依賴 `BaseStore` 介面，理論上**任何實作 BaseStore 的物件都可以接**。只要實作這幾個方法：

```python
from langgraph.store.base import BaseStore

class MyCustomStore(BaseStore):
    def get(self, namespace, key): ...
    def put(self, namespace, key, value): ...
    def search(self, namespace, query, limit): ...
    def delete(self, namespace, key): ...
```

這樣就能接 Redis、Qdrant、Chroma、Weaviate 等任意後端。



- Example of using SQLite
https://github.com/langchain-ai/langgraph/issues/5150

```python
# 安裝依賴
# pip install langmem langgraph langgraph-checkpoint-sqlite langchain-openai
```

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager

# ── 設定 API Key ──────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk-..."

# ── 1. Embedding 模型 ─────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # dims=1536
)

# ── 2. SQLite Store（持久化到本地檔案）────────────────
with SqliteStore.from_conn_string(
    "memories.db",
    index={
        "dims": 1536,
        "embed": embeddings,
    }
) as store:

    # ── 3. LLM ───────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── 4. Memory Manager ────────────────────────────
    manager = create_memory_store_manager(
        llm,
        namespace=("memories", "user_001"),
        store=store,
    )

    # ── 5. 寫入記憶（ingestion）──────────────────────
    conversation = [
        {"role": "user",      "content": "我叫小明，我喜歡用 Python 寫程式"},
        {"role": "assistant", "content": "好的，我記住了！"},
        {"role": "user",      "content": "我不喜歡 JavaScript"},
        {"role": "assistant", "content": "了解，我也記下來了。"},
    ]
    manager.invoke({"messages": conversation})
    print("✅ 記憶已寫入")

    # ── 6. 查詢記憶（retrieval）──────────────────────
    results = store.search(
        ("memories", "user_001"),
        query="使用者的程式語言偏好",
        limit=5,
    )
    print("\n🔍 搜尋結果：")
    for r in results:
        print(f"  [{r.score:.2f}] {r.value}")

    # ── 7. 第二輪對話，驗證記憶持久化 ────────────────
    conversation2 = [
        {"role": "user",      "content": "我最近開始學 Rust"},
        {"role": "assistant", "content": "Rust 很棒！"},
    ]
    manager.invoke({"messages": conversation2})

    all_memories = store.search(
        ("memories", "user_001"),
        query="程式語言",
        limit=10,
    )
    print("\n📚 所有記憶：")
    for r in all_memories:
        print(f"  {r.value}")
```
store=store 要傳給 manager：create_memory_store_manager 在 with 區塊外無法取得 store context，所以明確傳入比較保險。
SqliteStore 需要用 context manager（with 語法）來確保資料庫連線正確開關。如果要在函式外長期持有，可以手動 store.__enter__() / store.__exit__()。
重啟後記憶依然在：因為存的是 memories.db 檔案，下次執行只要用同一個路徑，之前的記憶都還在。如果要全部清空，直接刪掉 .db 檔即可。
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager

load_dotenv()

DB_PATH = "memories.db"
NAMESPACE = ("memories", "user_001")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

INDEX_CONFIG = {
    "dims": 1536,
    "embed": embeddings,
}


def get_store() -> SqliteStore:
    store = SqliteStore.from_conn_string(DB_PATH, index=INDEX_CONFIG)
    store.__enter__()
    store.setup()
    return store


def ingest(messages: list):
    store = get_store()
    try:
        manager = create_memory_store_manager(
            llm,
            namespace=NAMESPACE,
            store=store,
        )
        manager.invoke({"messages": messages})
        print("✅ ingestion 完成")
    finally:
        store.__exit__(None, None, None)


def search(query: str, limit: int = 5):
    store = get_store()
    try:
        results = store.search(NAMESPACE, query=query, limit=limit)
        return results
    finally:
        store.__exit__(None, None, None)


# ── 使用 ──────────────────────────────────────────────

ingest([
    {"role": "user",      "content": "我叫小明，喜歡用 Python 寫程式"},
    {"role": "assistant", "content": "好的，我記住了！"},
])

results = search("程式語言偏好")
for r in results:
    print(f"[{r.score:.2f}] {r.value}")
```
SqliteStore 是檔案型資料庫，多個連線同時讀沒問題，但同時寫入要小心 SQLite 本身的 write lock 限制。如果 ingestion 和 search 會並發執行，建議之後換 PostgreSQL。
