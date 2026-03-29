# LangMem & Mem0 整合筆記

記錄整合過程中的 API 探索、踩坑、資料格式與設計決策。

---

## 一、LangMem

### 安裝版本

```
langmem==0.0.30
```

### 探索過程

#### 可用的頂層 exports

```python
import langmem
dir(langmem)
# ['Prompt', 'ReflectionExecutor',
#  'create_manage_memory_tool',
#  'create_memory_manager',
#  'create_memory_searcher',
#  'create_memory_store_manager',
#  'create_multi_prompt_optimizer',
#  'create_prompt_optimizer',
#  'create_search_memory_tool',
#  'create_thread_extractor',
#  'errors', 'knowledge', 'prompts', 'reflection', 'utils']
```

主要使用的兩個：
- `create_memory_manager`：從對話中抽取並管理記憶
- `create_manage_memory_tool` / `create_search_memory_tool`：讓 agent 自己管理記憶的工具（tool-use 模式，本專案未使用）

---

### create_memory_manager

#### 簽名

```python
create_memory_manager(
    model: str | BaseChatModel,
    *,
    schemas: Sequence[S] = (Memory,),
    instructions: str = "...",   # 預設是英文的 memory manager prompt
    enable_inserts: bool = True,
    enable_updates: bool = True,
    enable_deletes: bool = False,
) -> Runnable[MemoryState, list[ExtractedMemory]]
```

#### 輸入格式（MemoryState）

```python
# langmem.knowledge.extraction.MemoryState
{
    "messages": list[AnyMessage],   # LangChain message 物件，非 dict
    "existing": list[tuple[str, Memory]],  # (id, Memory物件) 的 list
    "max_steps": int,  # 預設內建
}
```

**重要**：`existing` 的格式是 `list[tuple[str, Memory]]`，不是 `list[str]`。
第一次踩坑：傳 `list[str]` 會 silently 接受但行為不對；傳 dict 會 error。

#### 輸出格式（ExtractedMemory）

```python
# langmem.knowledge.extraction.ExtractedMemory
# 是一個 namedtuple，有兩個欄位：
mem.id       # str，UUID 格式
mem.content  # Memory pydantic 物件，或 str（視 schemas 設定）
```

**重要**：`mem.content` 預設是 `Memory` pydantic 物件，不是純字串。
第二次踩坑：直接把 `mem.content` 存進 LangGraph store 會拋 `TypeError: Object of type Memory is not JSON serializable`。

正確取法：

```python
content = mem.content
if hasattr(content, "content"):
    content = content.content   # Memory.content 才是字串
elif not isinstance(content, str):
    content = str(content)
```

#### Memory pydantic schema

```python
from langmem.knowledge.extraction import Memory
Memory(content="用戶服用氨氯地平")
# Memory 只有一個欄位 content: str
```

#### 呼叫方式

LangMem manager 是 async-first，需用 `ainvoke`：

```python
async def _run():
    return await manager.ainvoke({
        "messages": lc_messages,
        "existing": existing_memories,  # list[tuple[str, Memory]]
    })

extracted = asyncio.run(_run())
```

---

### LangGraph InMemoryStore（作為 LangMem 的向量後端）

#### 初始化（帶 embedding）

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "dims": 1536,          # embedding 維度，需與模型一致
        "embed": embed_fn,     # callable: list[str] -> list[list[float]]
    }
)
```

`embed_fn` 格式：

```python
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

def embed(texts: list[str]) -> list[list[float]]:
    return embedder.embed_documents(texts)
```

#### Namespace 設計

LangGraph store 用 tuple 做 namespace 隔離，類似目錄結構：

```python
namespace = ("memories", "user_001")
```

不同 user 用不同 namespace，互不干擾。

#### 主要方法

```python
# 寫入
store.put(
    namespace,          # tuple[str, ...]
    key="some-uuid",    # str，唯一 ID
    value={"content": "記憶內容"},  # dict，需 JSON serializable
)

# 讀取（語意搜尋）
results = store.search(
    namespace,
    query="查詢字串",   # 會自動 embed 後做 cosine search
    limit=5,
)
# results: list[SearchItem]
# item.key     -> str (存入時的 key)
# item.value   -> dict (存入時的 value)
# item.score   -> float (相似度，若有 embedding)

# 讀取全部（不做語意搜尋）
all_items = store.search(namespace, limit=50)  # 不傳 query
```

#### 持久化問題

`InMemoryStore` **重啟後資料消失**。解決方案：用 JSON 檔案手動持久化。

```
.langmem_store/
  user_001.json   -> {"<uuid>": "記憶文字", ...}
  user_002.json
```

啟動時從 JSON 讀回，`put` 進 store（此時不重新 embed，search 時才 embed query）：

```python
def _ensure_loaded(user_id):
    for mem_id, content in load_json(user_id).items():
        store.put(namespace, key=mem_id, value={"content": content})
```

**注意**：reload 時 value 已存進去，但 vector index 是空的。InMemoryStore 會在第一次 `search` 時對 value 做 embedding（lazy embedding）。所以 reload 後第一次 search 會有 API 呼叫。

---

### LangMem 整體架構小結

```
對話結束
    ↓
create_memory_manager.ainvoke(messages, existing)
    ↓ 抽取 + 整合
list[ExtractedMemory]  (id, Memory物件)
    ↓ 取 .content.content
InMemoryStore.put(namespace, key=id, value={"content": str})
    ↓ 同時
JSON 檔案持久化  .langmem_store/<user_id>.json

查詢時
    ↓
InMemoryStore.search(namespace, query=str, limit=N)
    ↓ embed query → cosine search
list[SearchItem] → 取 item.value["content"]
```

---

## 一之二、LangMem 記憶類型探索（Semantic / Episodic / Procedural）

### 探索指令

```python
# 確認有哪些 memory schema 可用
from langmem.knowledge import extraction
dir(extraction)
# 找到的相關項目：Memory, SummarizeThread, ExtractedMemory

# 查看 Memory schema
from langmem.knowledge.extraction import Memory, SummarizeThread
Memory.model_fields.keys()
# dict_keys(['content'])   ← 只有一個欄位，純字串

SummarizeThread.model_fields.keys()
# dict_keys(['title', 'summary'])

# 嘗試 import EpisodicMemory → 失敗
from langmem.knowledge.extraction import EpisodicMemory
# ImportError: cannot import name 'EpisodicMemory'
```

### 結論：三種類型是 prompt 概念，不是獨立 class

LangMem 0.0.30 中，Semantic / Episodic / Procedural **並非三個不同的 class 或模組**。
它們是 `create_memory_manager` 預設 `instructions` prompt 中的概念分類：

> *"maintaining a core store of **semantic**, **procedural**, and **episodic** memory"*

也就是說，三種記憶類型全部存成同一個 `Memory(content: str)`，由 **LLM 的 prompt 決定抽取什麼性質的內容**。

### 三種類型的實際含義

| 類型 | 含義 | 目前 code 中的體現 |
|------|------|-------------------|
| **Semantic** | 用戶的事實（誰、什麼病、吃什麼藥）| `create_memory_manager` 預設涵蓋，自訂 `instructions` 有引導抽取 |
| **Episodic** | 特定事件記憶（「上週量血壓 155/95」）| 部分涵蓋，但目前 `instructions` 偏向 semantic |
| **Procedural** | agent 應如何行動的知識（行為模式、偏好）| **目前未實作**，需要 `create_prompt_optimizer` |

### Procedural Memory 的正確做法

Procedural memory 不是存在 store 裡，而是用 `create_prompt_optimizer` 根據對話歷史**直接修改 system prompt**：

```python
from langmem import create_prompt_optimizer

optimizer = create_prompt_optimizer("openai:gpt-4o-mini")
# 傳入對話歷史 → 回傳更新後的 system prompt 字串
```

這是 LangMem 與 Mem0 差異最大的地方之一：LangMem 可以讓 agent 從對話中學習並自我改進 prompt，Mem0 沒有這個機制。

---

## 二、Mem0

### 安裝版本

```
mem0ai==1.0.8
```

### 探索過程

#### 可用的頂層 exports

```python
import mem0
dir(mem0)
# ['AsyncMemory', 'AsyncMemoryClient',
#  'Memory', 'MemoryClient', ...]
```

本地使用選 `Memory`（sync），雲端 managed service 用 `MemoryClient`。

#### Memory 預設設定

```python
from mem0 import Memory
m = Memory()
# 預設：
# vector_store: qdrant（path=/tmp/qdrant）
# embedder: openai（text-embedding-ada-002）
# llm: openai（gpt-4o）
# history_db_path: ~/.mem0/history.db
```

#### 可用 vector store backends

探索 `mem0/configs/vector_stores/` 目錄，支援：

```
qdrant（預設）、chroma、faiss、pgvector、
pinecone、weaviate、milvus、mongodb、
elasticsearch、redis、azure_ai_search ... 等 20+ 種
```

本專案選 ChromaDB，與 RAG 的 `.chroma/` 分開存放在 `.mem0_store/`。

---

### ChromaDB 設定

#### ChromaDbConfig 簽名

```python
from mem0.configs.vector_stores.chroma import ChromaDbConfig

ChromaDbConfig(
    collection_name: str = "mem0",
    client: Optional[Client] = None,    # 傳入已有的 chromadb client
    path: Optional[str] = None,         # 本地持久化路徑
    host: Optional[str] = None,         # 遠端 chroma server
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
)
```

本地持久化只需設定 `path`：

```python
{
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_health_memories",
            "path": ".mem0_store",   # 與 RAG 的 .chroma/ 分開
        },
    }
}
```

---

### Memory.from_config() 完整設定格式

```python
config = {
    "vector_store": {
        "provider": "chroma",       # 或 "qdrant", "faiss" 等
        "config": {
            "collection_name": "mem0_health_memories",
            "path": ".mem0_store",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",  # 預設是 ada-002
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",  # 預設是 gpt-4o，用 mini 省費用
        },
    },
    "history_db_path": ".mem0_store/history.db",  # SQLite，記錄操作歷史
}

m = Memory.from_config(config)
```

---

### 主要方法

#### add()

```python
m.add(
    messages,            # list[dict]，格式見下
    user_id="user_001",  # 用 user_id 隔離不同用戶的記憶
    # 也支援：agent_id, run_id, metadata
)
```

messages 格式（**注意：不是 LangChain message 物件**）：

```python
[
    {"role": "user", "content": "我在吃氨氯地平。"},
    {"role": "assistant", "content": "了解，這是降血壓藥。"},
]
# role 只接受 "user" 或 "assistant"，不接受 "human" 或 "ai"
```

**踩坑**：LangGraph 的 message type 是 `"human"` / `"ai"`，需要手動轉換：

```python
if role == "human":
    role = "user"
elif role == "ai":
    role = "assistant"
```

#### search()

```python
results = m.search(
    query="用戶的藥物",
    user_id="user_001",
    limit=5,
)
```

回傳格式：

```python
{
    "results": [
        {
            "id": "uuid-string",
            "memory": "在吃氨氯地平治療高血壓，每天早上一顆",
            "user_id": "user_001",
            "hash": "...",
            "created_at": "2026-03-29T...",
            "updated_at": "2026-03-29T...",
            "score": 0.87,   # 相似度分數
        },
        ...
    ]
}
```

取記憶內容：`results["results"][0]["memory"]`

#### 其他方法

```python
m.get(memory_id)             # 取單筆記憶
m.get_all(user_id="user_001") # 取該用戶所有記憶
m.update(memory_id, data)    # 更新單筆
m.delete(memory_id)          # 刪除單筆
m.delete_all(user_id="...")  # 刪除用戶所有記憶
m.history(memory_id)         # 查看某筆記憶的變更歷史
m.reset()                    # 清空所有記憶
```

---

### Mem0 儲存結構

```
.mem0_store/
  <uuid>/           # ChromaDB collection 資料夾
  chroma.sqlite3    # ChromaDB 索引
  history.db        # SQLite，記錄每次 add/update/delete 操作
```

`history.db` 是 mem0 特有的，記錄記憶的版本變更，可用 `m.history(memory_id)` 查詢。這是 LangMem 沒有的功能。

---

### Mem0 整體架構小結

```
對話結束
    ↓
Memory.add(messages, user_id=...)
    ↓ 內部：LLM 抽取事實 + embedding + 去重
ChromaDB (.mem0_store/)  +  SQLite history.db

查詢時
    ↓
Memory.search(query, user_id=..., limit=N)
    ↓ embed query → cosine search
{"results": [{"memory": str, "score": float, ...}]}
```

---

## 三、LangMem vs Mem0 對比

| 面向 | LangMem | Mem0 |
|------|---------|------|
| 設計哲學 | LangGraph 原生，記憶是 graph 的一部分 | 獨立服務，framework-agnostic |
| 記憶抽取 | `create_memory_manager`（async Runnable） | `Memory.add()`（sync/async 都有） |
| 儲存後端 | LangGraph `BaseStore`（需自帶，預設 in-memory）| 內建 20+ vector store 選項 |
| 本地持久化 | 需自己實作（本專案用 JSON 檔） | 內建（Qdrant local / Chroma path 模式） |
| 記憶格式 | 純字串（`Memory.content`） | 字串 + metadata（id, score, timestamps） |
| 版本歷史 | 無 | 有（`history.db` + `m.history(id)`） |
| API 呼叫風格 | async first（`ainvoke`） | sync first（`add`, `search`） |
| Message 格式 | LangChain message 物件 | dict with `"user"/"assistant"` role |
| Embedding 控制 | 完全自定義 | 設定 provider/model，較封裝 |

---

## 四、共用的 BaseMemory 介面

```python
class BaseMemory(ABC):
    @abstractmethod
    def save(self, user_id: str, messages: list[dict]) -> None:
        """messages 格式：[{"role": "human"/"ai", "content": "..."}]"""
        ...

    @abstractmethod
    def retrieve(self, user_id: str, query: str) -> str:
        """回傳格式：多行字串，每行一條記憶（"- 記憶內容"）"""
        ...
```

兩個 adapter 都遵循這個介面，agent graph 只依賴 `BaseMemory`，切換時只需換 adapter 不動其他邏輯。
