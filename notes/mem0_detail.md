# mem0 架構完整拆解筆記（v1.0.11）

> 版本：mem0ai 1.0.11  
> 原始碼位置：`mem0/`  
> 對應 Paper：arXiv 2504.19413（2025-04）

---

## 目錄

1. [v1.0.11 vs v0.1.11 主要差異](#1-v1011-vs-v0111-主要差異)
2. [目錄結構](#2-目錄結構)
3. [入口點與公開 API](#3-入口點與公開-api)
4. [Memory 主類別](#4-memory-主類別)
5. [操作流程：逐步拆解](#5-操作流程逐步拆解)
6. [記憶萃取 Pipeline（最核心）](#6-記憶萃取-pipeline最核心)
7. [新功能：Procedural Memory](#7-新功能procedural-memory)
8. [新功能：Agent Memory Extraction](#8-新功能agent-memory-extraction)
9. [新功能：Reranker](#9-新功能reranker)
10. [進階 Metadata Filter](#10-進階-metadata-filter)
11. [Exception 系統](#11-exception-系統)
12. [所有 Prompts（完整內容）](#12-所有-prompts完整內容)
13. [設定系統](#13-設定系統)
14. [關鍵資料結構](#14-關鍵資料結構)
15. [常見修改點速查](#15-常見修改點速查)
16. [關於 Paper 中的 Summary 功能](#16-關於-paper-中的-summary-功能)

---

## 1. v1.0.11 vs v0.1.11 主要差異

| 功能 | v0.1.11 | v1.0.11 |
|------|---------|---------|
| 預設版本 | `version="v1.0"` | `version="v1.1"`（已是預設） |
| Reranker | ❌ | ✅ 新增，5 種實作 |
| Procedural Memory | ❌ | ✅ Agent 執行歷史摘要 |
| Agent Memory Extraction | ❌ | ✅ 專屬 Prompt 抽取 agent 特徵 |
| Exception 系統 | 只有 APIError | ✅ 完整型別階層 |
| 進階 Filter | 只有 key=value | ✅ eq/ne/gt/lt/in/contains/AND/OR/NOT |
| infer=False | ❌ | ✅ 直接存入原始訊息，不經 LLM |
| Vision 支援 | ❌ | ✅ 圖片描述後儲存 |
| UUID 幻覺保護 | ❌ | ✅ 暫時用整數 ID 對應真實 UUID |
| 自訂 Prompt | ❌ | ✅ `custom_fact_extraction_prompt` / `custom_update_memory_prompt` |
| Payload 新欄位 | user_id, agent_id, run_id | + `actor_id`, `role`, `memory_type` |
| AsyncMemory | 基本 | ✅ 完整 async/await 實作 |
| 敏感欄位遮蔽 | ❌ | ✅ Telemetry 時自動遮蔽 api_key 等 |

---

## 2. 目錄結構

```
mem0/
├── memory/
│   ├── main.py          ← Memory + AsyncMemory 主類別（最重要）
│   ├── graph_memory.py  ← Neo4j 知識圖譜
│   ├── storage.py       ← SQLite 歷史記錄
│   ├── utils.py         ← parse_messages, 工具函式
│   ├── base.py          ← MemoryBase 抽象介面
│   ├── setup.py         ← ~/.mem0/ 初始化
│   └── telemetry.py     ← PostHog 使用統計
│
├── llms/
├── embeddings/
├── vector_stores/
├── graphs/
│
├── reranker/            ← 新增！
│   ├── base.py          ← BaseReranker 抽象介面
│   ├── cohere_reranker.py
│   ├── huggingface_reranker.py
│   ├── llm_reranker.py
│   ├── sentence_transformer_reranker.py
│   └── zero_entropy_reranker.py
│
├── configs/
│   ├── base.py          ← MemoryConfig, MemoryItem, AzureConfig
│   ├── enums.py         ← MemoryType enum（新增）
│   ├── prompts.py       ← 所有 Prompt 集中於此（新增）
│   ├── rerankers/       ← RerankerConfig（新增）
│   ├── llms/
│   ├── embeddings/
│   └── vector_stores/
│
├── utils/
│   └── factory.py       ← 所有 Factory（新增 RerankerFactory, GraphStoreFactory）
│
├── exceptions.py        ← 新增！完整 Exception 型別
├── client/
└── proxy/
```

---

## 3. 入口點與公開 API

### 本機使用

```python
from mem0 import Memory

memory = Memory()  # 預設使用 v1.1, openai, qdrant

memory = Memory.from_config({
    "llm": {"provider": "openai", "config": {"model": "gpt-4o"}},
    "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/mem0"}},
    "embedder": {"provider": "openai"},
    "reranker": {"provider": "cohere", "config": {"api_key": "..."}},
    "custom_fact_extraction_prompt": "...",   # 新：自訂抽取 prompt
    "custom_update_memory_prompt": "...",     # 新：自訂更新 prompt
})

# Context manager 支援（新）
with Memory() as memory:
    memory.add(...)
```

### 所有公開方法

| 方法 | 新增參數 | 說明 |
|------|---------|------|
| `add(messages, ...)` | `infer`, `memory_type`, `prompt` | 新增/更新記憶 |
| `get(memory_id)` | - | 取得單筆（新增 actor_id, role 欄位） |
| `get_all(...)` | `filters` | 列出所有（新增進階 filter） |
| `search(query, ...)` | `threshold`, `rerank` | 搜尋（新增閾值與 rerank） |
| `update(memory_id, data)` | - | 更新記憶 |
| `delete(memory_id)` | - | 刪除 |
| `delete_all(...)` | - | 批次刪除 |
| `history(memory_id)` | - | 歷史記錄 |
| `reset()` | - | 清空 |
| `close()` | - | 新：釋放資源（SQLite 連線） |

---

## 4. Memory 主類別

**檔案：** `memory/main.py`

### 初始化流程

```python
Memory(config: MemoryConfig = MemoryConfig())
```

```
1. 讀取 custom_fact_extraction_prompt, custom_update_memory_prompt
2. EmbedderFactory.create()   → self.embedding_model
3. VectorStoreFactory.create() → self.vector_store
4. LlmFactory.create()        → self.llm
5. SQLiteManager()            → self.db
6. RerankerFactory.create()   → self.reranker（如果有設定）
7. GraphStoreFactory.create() → self.graph（如果有設定）
8. 建立 Telemetry vector store（分開的 collection）
9. setup_config()
```

### 重要的 helper：`_build_filters_and_metadata()`

新版把 filter 建構抽成獨立函式（module-level），邏輯：

```python
def _build_filters_and_metadata(
    user_id, agent_id, run_id, actor_id=None,
    input_metadata=None, input_filters=None
) -> (base_metadata_template, effective_query_filters):
```

- `base_metadata_template`：寫入記憶時用的 metadata
- `effective_query_filters`：查詢時用的 filter
- 若三個 ID 都沒提供 → 丟 `Mem0ValidationError`
- `actor_id` 只加入 query filter，不加入 metadata

---

## 5. 操作流程：逐步拆解

### 5.1 add()

```
輸入正規化：
  str      → [{"role": "user", "content": str}]
  dict     → [dict]
  list     → 原樣

特殊路徑：
  memory_type == "procedural_memory" 且有 agent_id
      → _create_procedural_memory()（走 LLM 摘要，不走事實抽取）

Vision 處理（如果 enable_vision=True）：
  parse_vision_messages() → 圖片 URL 轉成文字描述

並行執行（ThreadPoolExecutor）：
  Thread 1: _add_to_vector_store(messages, metadata, filters, infer)
  Thread 2: _add_to_graph(messages, filters)
  → concurrent.futures.wait() 等待兩者完成

回傳：
  {"results": [...], "relations": [...]}  ← enable_graph=True
  {"results": [...]}                       ← 否則
```

### 5.2 `_add_to_vector_store()` 詳細流程

#### 路徑 A：infer=False（新功能）

```
對每條非 system 訊息：
  1. 提取 role, actor_id（name 欄位）
  2. embed(content)
  3. _create_memory(content, embeddings, metadata + role)
  4. 加入 returned_memories
```

直接存入，**不呼叫 LLM**。

#### 路徑 B：infer=True（預設）

```
1. parse_messages() → "user: ...\nassistant: ...\n"

2. 選擇 Prompt：
   自訂 custom_fact_extraction_prompt？
     → 用自訂的
   否則，_should_use_agent_memory_extraction()?
     → True: AGENT_MEMORY_EXTRACTION_PROMPT（agent_id 存在 + 有 assistant 訊息）
     → False: USER_MEMORY_EXTRACTION_PROMPT

3. ensure_json_instruction() 確保 prompt 含 "json" 字樣
   （OpenAI json_object 模式的要求）

4. [LLM Call #1] 事實抽取
   response_format={"type": "json_object"}
   → {"facts": ["fact1", ...]}
   → remove_code_blocks() + extract_json() 處理模型輸出亂加 markdown

5. normalize_facts() 正規化
   （小模型可能回 {"fact": "..."} 而非字串，這裡修正）

6. 對每條 fact：
   embed(fact) → vector
   vector_store.search(vector, filters, limit=5) → 舊記憶
   收集到 retrieved_old_memory

7. 去重（以 id 為 key）

8. UUID 幻覺保護：
   把 retrieved_old_memory 的 id 暫時換成整數 0,1,2...
   temp_uuid_mapping = {0: "real-uuid-1", 1: "real-uuid-2", ...}

9. [LLM Call #2] 更新決策
   get_update_memory_messages(retrieved_old_memory, new_facts, custom_prompt)
   response_format={"type": "json_object"}
   → {"memory": [{id, text, event, old_memory?}, ...]}

10. 執行操作：
    ADD    → _create_memory()
    UPDATE → _update_memory()（從 temp_uuid_mapping 還原真實 UUID）
    DELETE → _delete_memory()
    NONE   → 如果有 agent_id 或 run_id，更新 session ID 到現有記憶
```

### 5.3 search()

```
1. _build_filters_and_metadata() 建立 effective_filters
2. 進階 filter 偵測：_has_advanced_operators()？
   → True: _process_metadata_filters() 轉換運算子
3. 並行執行：
   Thread 1: _search_vector_store(query, filters, limit, threshold)
   Thread 2: graph.search()（如果 enable_graph）
4. Reranking（如果 rerank=True 且 self.reranker 存在）：
   self.reranker.rerank(query, memories, limit)
   → 失敗時 fallback 回原始結果
5. 回傳 {"results": [...], "relations": [...]}
```

### 5.4 `_search_vector_store()`

```python
query_embedding = embed(query)
results = vector_store.search(query, query_embedding, limit, filters)
# threshold 過濾
if threshold:
    results = [r for r in results if r.score >= threshold]
# 格式化成 MemoryItem
```

### 5.5 _create_memory()

```python
# 嵌入向量（優先使用快取）
if data in existing_embeddings_dict:
    embeddings = existing_embeddings_dict[data]
else:
    embeddings = embed(data)

memory_id = uuid4()
metadata["data"] = data
metadata["hash"] = md5(data)
metadata["created_at"] = now UTC
metadata["updated_at"] = created_at

vector_store.insert([embeddings], [memory_id], [metadata])
db.add_history(memory_id, None, data, "ADD", ...)
return memory_id
```

### 5.6 _update_memory()

```python
existing_memory = vector_store.get(memory_id)
prev_value = existing_memory.payload["data"]

metadata["data"] = new_data
metadata["hash"] = md5(new_data)
metadata["created_at"] = 保留原本的（UTC 正規化）
metadata["updated_at"] = now UTC

# 保留舊的 session ID（user_id/agent_id/run_id/actor_id/role）若新 metadata 沒提供

vector_store.update(memory_id, new_embeddings, new_metadata)
db.add_history(memory_id, prev_value, new_data, "UPDATE", ...)
```

### 5.7 _delete_memory()

```python
existing_memory = vector_store.get(memory_id)  # 若未傳入
prev_value = existing_memory.payload["data"]
created_at = 正規化原本的

vector_store.delete(memory_id)
db.add_history(memory_id, prev_value, None, "DELETE", is_deleted=1, ...)
```

---

## 6. 記憶萃取 Pipeline（最核心）

詳見 [第 12 節的完整 Prompt 內容](#12-所有-prompts完整內容)

### 決策邏輯重點

| 情境 | 決策 |
|------|------|
| 舊記憶無，新 fact 有 | ADD |
| 舊記憶 "喜歡打板球"，新 "喜歡跟朋友打板球" | UPDATE 為更詳細版 |
| 舊記憶 "喜歡乳酪披薩"，新 "愛吃乳酪披薩" | NONE（語意相同） |
| 舊記憶 "喜歡乳酪披薩"，新 "不喜歡乳酪披薩" | DELETE 舊的 |
| NONE 但有新 agent_id/run_id | 更新 session ID 到現有記憶（新行為） |

---

## 7. 新功能：Procedural Memory

**用途：** 儲存 Agent 的執行歷史，不做事實抽取，做「完整行為摘要」

**觸發條件：**
```python
memory.add(messages, agent_id="agent-1", memory_type="procedural_memory")
```

**流程：**
```
messages + PROCEDURAL_MEMORY_SYSTEM_PROMPT
    ↓
LLM 生成完整執行摘要（保留所有 action/result 細節）
    ↓
embed(摘要)
    ↓
_create_memory(摘要, metadata + {"memory_type": "procedural_memory"})
```

**與一般記憶的差異：**
- 不拆分成獨立 facts，保留完整序列
- Prompt 強調「不得改寫 action result，原文照存」
- 適合 browser agent、tool-use agent 的執行記錄

---

## 8. 新功能：Agent Memory Extraction

當同時滿足：
1. `agent_id` 存在
2. messages 裡有 `role == "assistant"` 的訊息

→ 自動切換為 `AGENT_MEMORY_EXTRACTION_PROMPT`

**差異：**

| | USER_MEMORY_EXTRACTION_PROMPT | AGENT_MEMORY_EXTRACTION_PROMPT |
|---|---|---|
| 抽取對象 | user 訊息 | assistant 訊息 |
| 抽取內容 | 個人偏好、計畫、事實 | assistant 的能力、性格、偏好 |
| 範例輸出 | "名字叫 John" | "Admires software engineering" |

---

## 9. 新功能：Reranker

**抽象介面：** `reranker/base.py`

```python
class BaseReranker(ABC):
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        # 回傳：加上 rerank_score 欄位的 documents
```

### 支援的 Provider

| Provider | 說明 |
|----------|------|
| `cohere` | Cohere Rerank API |
| `huggingface` | HuggingFace cross-encoder |
| `sentence_transformer` | 本機 SentenceTransformer cross-encoder |
| `llm` | 用 LLM 評分重排 |
| `zero_entropy` | 基於熵的排序 |

### 設定方式

```python
{
    "reranker": {
        "provider": "cohere",
        "config": {
            "api_key": "...",
            "model": "rerank-english-v3.0",
            "top_n": 10
        }
    }
}
```

### 在 search() 中的行為

```python
# rerank=True 預設啟用
memory.search("query", user_id="u1", rerank=True)

# 可以關閉
memory.search("query", user_id="u1", rerank=False)
```

失敗時自動 fallback，不拋例外。

### 新增 Reranker Provider

1. 繼承 `BaseReranker`，實作 `rerank()`
2. 在 `utils/factory.py` → `RerankerFactory.provider_to_class` 加入對應
3. 在 `configs/rerankers/` 加入 Config class

---

## 10. 進階 Metadata Filter

### 基本運算子

```python
# 精確比對（舊方法，仍可用）
memory.search(query, user_id="u1", filters={"role": "user"})

# 新運算子
filters = {
    "score": {"gt": 0.8},           # score > 0.8
    "score": {"gte": 0.8},          # score >= 0.8
    "score": {"lt": 0.5},           # score < 0.5
    "score": {"lte": 0.5},          # score <= 0.5
    "role": {"eq": "user"},         # role == "user"
    "role": {"ne": "system"},       # role != "system"
    "role": {"in": ["user", "assistant"]},   # role in [...]
    "role": {"nin": ["system"]},             # role not in [...]
    "memory": {"contains": "pizza"},         # 包含
    "memory": {"icontains": "pizza"},        # 不分大小寫包含
    "role": "*",                             # 萬用字元（任何值）
}
```

### 邏輯運算子

```python
filters = {
    "AND": [
        {"role": "user"},
        {"score": {"gt": 0.5}}
    ]
}

filters = {
    "OR": [
        {"role": "user"},
        {"role": "assistant"}
    ]
}

filters = {
    "NOT": [{"role": "system"}]
}
```

---

## 11. Exception 系統

**檔案：** `exceptions.py`

### 繼承結構

```
MemoryError (base)
├── AuthenticationError   → HTTP 401/403
├── RateLimitError        → HTTP 429
├── ValidationError       → HTTP 400/422
├── MemoryNotFoundError   → HTTP 404
├── NetworkError          → HTTP 408/502/503/504
├── ConfigurationError    → 設定錯誤
├── MemoryQuotaExceededError → HTTP 413
├── MemoryCorruptionError
├── VectorSearchError
├── CacheError
├── VectorStoreError      → OSS 向量操作失敗
├── GraphStoreError       → OSS 圖譜操作失敗
├── EmbeddingError        → OSS embedding 失敗
├── LLMError              → OSS LLM 呼叫失敗
├── DatabaseError         → OSS SQLite 失敗
└── DependencyError       → 套件未安裝
```

### Exception 格式

每個 exception 都有：

```python
e.message       # 人類可讀的錯誤描述
e.error_code    # 程式碼（如 "VALIDATION_001"）
e.details       # dict，額外的錯誤上下文
e.suggestion    # 建議使用者怎麼修
e.debug_info    # 技術除錯資訊
```

### 程式碼中的使用範例

```python
# 若三個 ID 都沒提供
raise Mem0ValidationError(
    message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
    error_code="VALIDATION_001",
    details={"provided_ids": {...}},
    suggestion="Please provide at least one identifier to scope the memory operation."
)

# memory_type 不合法
raise Mem0ValidationError(
    message=f"Invalid 'memory_type'...",
    error_code="VALIDATION_002",
    ...
)
```

---

## 12. 所有 Prompts（完整內容）

**檔案：** `configs/prompts.py`

### Prompt 1：FACT_RETRIEVAL_PROMPT（舊版，向下相容）

```
You are a Personal Information Organizer...
（與 v0.1.11 相同，保留用於 get_fact_retrieval_messages_legacy()）
```

### Prompt 2：USER_MEMORY_EXTRACTION_PROMPT（新版預設）

```
You are a Personal Information Organizer...

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES.
# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:
1. Store Personal Preferences
2. Maintain Important Personal Details
3. Track Plans and Intentions
4. Remember Activity and Service Preferences
5. Monitor Health and Wellness Preferences
6. Store Professional Details
7. Miscellaneous Information Management

Few-shot examples（user+assistant 對話格式）：
Input: User: "Hi." / Assistant: "Hello!..."
Output: {"facts": []}

Input: User: "Hi, my name is John. I am a software engineer." / Assistant: "Nice to meet you..."
Output: {"facts": ["Name is John", "Is a Software engineer"]}

Rules:
- Today's date is {datetime.now()}
- Only from user messages（強調 2 次）
- Return JSON with "facts" key
- Detect user's language, record facts in same language
```

**與舊版的差異：**
- Few-shot 格式包含 assistant 回覆（讓模型知道要忽略它）
- 加入「語言偵測並用同語言記錄」
- 強調 2 次只從 user 訊息抽取（含 PENALIZED 警告）

### Prompt 3：AGENT_MEMORY_EXTRACTION_PROMPT

```
You are an Assistant Information Organizer...

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES.

Types of Information to Remember:
1. Assistant's Preferences
2. Assistant's Capabilities
3. Assistant's Hypothetical Plans or Activities
4. Assistant's Personality Traits
5. Assistant's Approach to Tasks
6. Assistant's Knowledge Areas
7. Miscellaneous Information

Few-shot examples：
Input: User: "Hi, I am looking for a restaurant..." / Assistant: "Sure, I can help..."
Output: {"facts": []}  ← assistant 沒有揭露個人資訊

Input: User: "Hi, my name is John. I am a software engineer."
       Assistant: "Nice to meet you, John! My name is Alex and I admire software engineering."
Output: {"facts": ["Admires software engineering", "Name is Alex"]}
```

### Prompt 4：DEFAULT_UPDATE_MEMORY_PROMPT

完整決策規則（對應 ADD/UPDATE/DELETE/NONE 四種操作）：

```
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add, (2) update, (3) delete, (4) no change.

Guidelines:
1. ADD: New info not in memory → generate new ID
2. UPDATE: Info present but needs update → keep same ID
   - "喜歡打板球" + "喜歡跟朋友打板球" → UPDATE
   - "喜歡乳酪披薩" + "愛吃乳酪披薩" → NONE（same meaning）
3. DELETE: Info contradicts existing → delete
4. NONE: Already present or irrelevant

Output format:
{
  "memory": [
    {
      "id": "<existing or new ID>",
      "text": "<content>",
      "event": "ADD|UPDATE|DELETE|NONE",
      "old_memory": "<required if UPDATE>"
    }
  ]
}
```

### Prompt 5：PROCEDURAL_MEMORY_SYSTEM_PROMPT

```
You are a memory summarization system that records and preserves
the complete interaction history between a human and an AI agent.

Structure:
- Overview: Task Objective, Progress Status
- Sequential Agent Actions（numbered）:
  Each step must include:
  1. Agent Action（exact description）
  2. Action Result（verbatim, UNMODIFIED）
  3. Embedded Metadata:
     - Key Findings
     - Navigation History
     - Errors & Challenges
     - Current Context

Guidelines:
1. Preserve Every Output（do NOT paraphrase）
2. Chronological Order
3. Detail and Precision（URLs, indexes, error messages）
4. Output Only the Summary（no preamble）
```

### Prompt 6：MEMORY_ANSWER_PROMPT（用於 chat 功能）

```
You are an expert at answering questions based on the provided memories.
Extract relevant information from the memories based on the question.
If no relevant information, accept the question and provide a general response.
```

---

## 13. 設定系統

**主設定：** `configs/base.py` → `MemoryConfig`

```python
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig      # 預設 qdrant
    llm: LlmConfig                       # 預設 openai
    embedder: EmbedderConfig             # 預設 openai
    history_db_path: str                 # 預設 ~/.mem0/history.db
    graph_store: GraphStoreConfig        # 預設空（不啟用）
    reranker: Optional[RerankerConfig]   # 新！預設 None
    version: str = "v1.1"               # 新：預設已是 v1.1
    custom_fact_extraction_prompt: Optional[str]  # 新！
    custom_update_memory_prompt: Optional[str]    # 新！
```

### 完整設定範例（含所有新功能）

```python
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0,
            "api_key": "sk-...",
            "max_tokens": 3000,
            "enable_vision": False,          # 新：啟用圖片理解
            "vision_details": "auto",        # 新：圖片解析精度
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "path": "/tmp/mem0"
        }
    },
    "reranker": {                            # 新！
        "provider": "cohere",
        "config": {
            "api_key": "...",
            "model": "rerank-english-v3.0"
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "..."
        }
    },
    "custom_fact_extraction_prompt": "...", # 新！
    "custom_update_memory_prompt": "...",   # 新！
    "history_db_path": "~/.mem0/history.db",
    "version": "v1.1"
}
```

### MemoryType Enum（新增）

```python
class MemoryType(Enum):
    SEMANTIC    = "semantic_memory"
    EPISODIC    = "episodic_memory"
    PROCEDURAL  = "procedural_memory"
```

目前 `add()` 只處理 `PROCEDURAL`，SEMANTIC/EPISODIC 是保留的命名。

---

## 14. 關鍵資料結構

### MemoryItem（回傳給使用者）

```python
class MemoryItem(BaseModel):
    id: str
    memory: str
    hash: Optional[str]
    metadata: Optional[Dict[str, Any]]
    score: Optional[float]          # search 才有
    created_at: Optional[str]       # UTC ISO 8601（新版統一 UTC）
    updated_at: Optional[str]
```

`get()` 和 `get_all()` 還會額外加入（promoted fields）：

```python
{
    "user_id": "...",    # 如果 payload 有
    "agent_id": "...",   # 如果 payload 有
    "run_id": "...",     # 如果 payload 有
    "actor_id": "...",   # 新！如果 payload 有
    "role": "user",      # 新！如果 payload 有
}
```

### Payload 結構（Vector Store 儲存）

```python
{
    "data": "記憶文字",
    "hash": "md5",
    "created_at": "2025-04-14T00:00:00+00:00",  # UTC
    "updated_at": "2025-04-14T00:00:00+00:00",  # UTC
    "user_id": "...",
    "agent_id": "...",       # 選用
    "run_id": "...",         # 選用
    "actor_id": "...",       # 新！來自 messages[i].name
    "role": "user",          # 新！infer=False 時記錄
    "memory_type": "procedural_memory",  # 新！Procedural Memory 才有
    # 自訂 metadata...
}
```

### add() 回傳格式

```python
# v1.1（預設）
{
    "results": [
        {"id": "uuid", "memory": "text", "event": "ADD"},
        {"id": "uuid", "memory": "text", "event": "UPDATE", "previous_memory": "old text"},
        {"id": "uuid", "memory": "text", "event": "DELETE"},
    ],
    "relations": [...]   # 只有 enable_graph=True 時才有
}
```

---

## 15. 常見修改點速查

| 想修改什麼 | 去哪裡改 |
|-----------|---------|
| 事實抽取的 Prompt | `configs/prompts.py` → `USER_MEMORY_EXTRACTION_PROMPT` |
| Agent 記憶抽取的 Prompt | `configs/prompts.py` → `AGENT_MEMORY_EXTRACTION_PROMPT` |
| 更新決策的 Prompt | `configs/prompts.py` → `DEFAULT_UPDATE_MEMORY_PROMPT` |
| Procedural Memory 摘要 Prompt | `configs/prompts.py` → `PROCEDURAL_MEMORY_SYSTEM_PROMPT` |
| 搜尋幾筆舊記憶（每條 fact） | `memory/main.py:574` → `limit=5`（`_add_to_vector_store` 中） |
| 搜尋閾值 | `memory.search(threshold=0.7)` 或 `_search_vector_store` 中 |
| 判斷是否用 agent memory | `memory/main.py` → `_should_use_agent_memory_extraction()` |
| 新增 Vector Store | 繼承 `VectorStoreBase`，加入 `VectorStoreFactory` |
| 新增 LLM Provider | 繼承 `LLMBase`，加入 `LlmFactory` |
| 新增 Embedding Provider | 繼承 `EmbeddingBase`，加入 `EmbedderFactory` |
| 新增 Reranker | 繼承 `BaseReranker`，加入 `RerankerFactory` |
| 修改 Payload 儲存結構 | `memory/main.py` → `_create_memory()` / `_update_memory()` |
| 修改歷史記錄格式 | `memory/storage.py` |
| 修改 Graph 抽取邏輯 | `memory/graph_memory.py` |
| 自訂 Prompt（不改 code） | `MemoryConfig(custom_fact_extraction_prompt="...")` |
| 新增 Exception 型別 | `exceptions.py` |

---

## 16. 關於 Paper 中的 Summary 功能

**論文（arXiv 2504.19413）** 描述的架構有：
1. **Conversation Summary Storage** - 每次抽取時把摘要一起餵給 LLM
2. **非同步 Summary Generation Module** - 背景程序定期刷新摘要

**v1.0.11 的狀況：**

這兩個功能在 open-source 版本中**仍未實作**。但 `PROCEDURAL_MEMORY_SYSTEM_PROMPT` 是最接近的：它的功能是「對 agent 執行歷史生成摘要」，用於 Procedural Memory 路徑。

**差異：**

| | Paper 的 Summary | Procedural Memory（現有） |
|--|--|--|
| 觸發時機 | 非同步背景定期執行 | 手動呼叫 `add(..., memory_type="procedural_memory")` |
| 摘要用途 | 下次 add() 時餵給 LLM 作為上下文 | 存成一筆記憶，供未來搜尋 |
| 抽取影響 | 直接影響 fact 抽取品質 | 不影響一般記憶抽取 |

**如果你要自己加 summary 功能，修改點：**
1. `memory/storage.py` - 加一個 `conversation_summary` 表
2. `memory/main.py` → `_add_to_vector_store()` - 在 LLM Call #1 前查詢並注入摘要
3. `configs/prompts.py` - 更新 `USER_MEMORY_EXTRACTION_PROMPT` 加入摘要佔位符
4. 加一個背景執行的 summary refresh 機制（可用 threading.Timer 或 celery）
