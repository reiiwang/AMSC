# Agent 設計與邏輯筆記

記錄目前 LangGraph agent 的完整架構、每個元件的職責、設計決策與資料流。

---

## 一、整體架構

```
用戶輸入
    ↓
[LangGraph Graph]
    ├─ retrieve_memory    ← 從 memory layer 取跨對話長期記憶
    ↓
    ├─ agent (LLM)        ← 決策：直接回答 or 呼叫工具
    ↓         ↓
    │    [tool_calls?]
    │         ↓ YES（最多 3 次）
    ├─ tools              ← 執行工具，結果回傳給 agent
    │         ↓
    │    loop back to agent
    │
    ↓ NO（或達上限）
    ├─ save_memory        ← 將對話存入 memory layer
    ↓
  END → 回傳給用戶
```

---

## 二、State 設計

**檔案**：`agent/state.py`

```python
class AgentState(MessagesState):
    memory_context: str
```

- 繼承 LangGraph 的 `MessagesState`，自帶 `messages: list[AnyMessage]`
- 新增 `memory_context: str`：在 `retrieve_memory` 節點寫入，`call_agent` 節點讀取注入 system prompt
- `messages` 使用 `add_messages` reducer，每次 append 而非覆蓋

---

## 三、Graph 節點詳解

**檔案**：`agent/graph.py`

### build_graph(memory, user_id)

接受一個 `BaseMemory` 物件和 `user_id`，回傳編譯好的 LangGraph CompiledGraph。
這個設計讓切換 memory adapter 時只需傳入不同物件，graph 結構不變。

---

### 節點 1：retrieve_memory

```python
def retrieve_memory(state: AgentState) -> dict:
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    context = memory.retrieve(user_id, last_user_msg)
    return {"memory_context": context}
```

- 取 messages 中最後一條 human message 作為 query
- 呼叫 memory adapter 的 `retrieve()`，回傳相關的長期記憶字串
- 寫入 state 的 `memory_context`

---

### 節點 2：call_agent

```python
def call_agent(state: AgentState) -> dict:
    system_content = SYSTEM_PROMPT
    if state.get("memory_context"):
        system_content += f"\n\n## 用戶長期記憶\n{state['memory_context']}"

    recent_messages = trim_messages(
        state["messages"],
        max_tokens=HISTORY_LIMIT,   # = 5
        token_counter=len,
        strategy="last",
        include_system=False,
    )

    response = llm.invoke([SystemMessage(content=system_content)] + recent_messages)
    return {"messages": [response]}
```

**兩層 context 的區分**：
- **短期 context（chat history）**：`trim_messages` 只保留最後 5 條訊息，超過的捨棄
- **長期 context（memory）**：`memory_context` 注入 system prompt，不受 5 條限制影響

**LLM 設定**：

```python
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)
```

`bind_tools` 讓 LLM 可以輸出 tool_calls，model 自行決定是否呼叫工具。

---

### 節點 3：tools（ToolNode）

```python
tool_node = ToolNode(TOOLS)
```

LangGraph prebuilt 的 `ToolNode`，自動：
1. 讀取 LLM 輸出的 `tool_calls`
2. 依名稱找到對應函數並執行
3. 把結果包成 `ToolMessage` 加入 messages

---

### 節點 4：save_memory

```python
def save_memory(state: AgentState) -> dict:
    serialized = [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]
    memory.save(user_id, serialized)
    return {}
```

- 把 state 中所有 messages 序列化成 `{"role": ..., "content": ...}` 格式
- 傳給 memory adapter 的 `save()`
- 每次對話結束都觸發，讓記憶持續累積

**注意**：`m.type` 的值是 LangGraph 的 message type（`"human"`, `"ai"`, `"tool"`），各 memory adapter 內部會做轉換。

---

### 條件邊：should_use_tools

```python
def should_use_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    tool_call_count = sum(1 for m in state["messages"] if m.type == "tool")
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        if tool_call_count < MAX_TOOL_ITERATIONS:  # = 3
            return "tools"
    return "save_memory"
```

判斷條件：
1. 最後一條 message 是否有 `tool_calls`（LLM 決定要用工具）
2. 目前已執行的工具輪次是否未達上限（防止無限迴圈）

上限 `MAX_TOOL_ITERATIONS = 3`：同一輪對話最多 3 次工具呼叫。

---

## 四、工具設計

**檔案**：`agent/tools.py`

### Tool 1：search_knowledge_base

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the health knowledge base for information about medications,
    diseases, lab results, or lifestyle topics."""
    return rag_retrieve(query)
```

- 觸發條件（由 docstring 引導 LLM）：問到藥物、疾病、檢驗、生活習慣等專業知識
- 底層呼叫 `rag/retriever.py` 的 `retrieve()`，回 ChromaDB vector search 結果
- 回傳格式：多筆文件以 `---` 分隔，每筆含 `[category] title\ncontent`

### Tool 2：get_user_health_profile

```python
@tool
def get_user_health_profile(user_id: str) -> str:
    """Retrieve the health profile and past conversation history for a given user."""
    # 從 mock_conversations.json 取用戶資料
```

- 觸發條件：需要了解用戶的基本健康背景
- 從 `data/mock_conversations.json` 讀取靜態 profile（姓名、年齡、已知病症）
- 回傳格式：純字串描述

**設計注意**：`user_id` 由 LLM 自行決定傳入什麼值。在測試中如果 user_id 與 mock 資料不符會回傳 "No profile found"。

---

## 五、RAG 層設計

**檔案**：`rag/indexer.py`, `rag/retriever.py`

### 知識庫來源

`data/knowledge_base.json`：12 筆健康知識文件，涵蓋：
- 藥物（氨氯地平、Sumatriptan、Statin、鼻用類固醇）
- 疾病管理（高血壓、糖尿病、偏頭痛、腰痛、過敏性鼻炎）
- 檢驗解讀（血脂、血壓量測）
- 生活（睡眠與慢性病）

每筆格式：

```json
{
  "id": "kb_001",
  "category": "藥物",
  "title": "氨氯地平（Amlodipine）",
  "content": "..."
}
```

### 索引（一次性）

```python
# rag/indexer.py
collection.add(
    ids=[doc["id"] for doc in docs],
    documents=[f"{doc['title']}\n{doc['content']}" for doc in docs],
    metadatas=[{"category": doc["category"], "title": doc["title"]} for doc in docs],
)
```

- 用 `make index` 執行
- drop + recreate collection（全量重建）
- Embedding model：`text-embedding-3-small`（1536 dims）
- 存入 `.chroma/`

### 查詢

```python
# rag/retriever.py
results = collection.query(query_texts=[query], n_results=TOP_K)  # TOP_K = 3
```

- 每次查詢都開新 client 連線（stateless，適合 tool 使用情境）
- 回傳 top 3 相關文件

---

## 六、Memory Layer 設計

**檔案**：`memory/base.py`

```python
class BaseMemory(ABC):
    def save(self, user_id: str, messages: list[dict]) -> None: ...
    def retrieve(self, user_id: str, query: str) -> str: ...
```

三個實作：

| Adapter | 說明 | 持久化位置 |
|---------|------|-----------|
| `DummyMemory` | no-op，用於測試 | 無 |
| `LangMemAdapter` | LangMem + InMemoryStore | `.langmem_store/<user_id>.json` |
| `Mem0Adapter` | mem0ai + ChromaDB | `.mem0_store/` |

---

## 七、關鍵設計決策

### 1. chat history vs long-term memory 分開處理
- **chat history**：`trim_messages` 只保留最近 5 條，控制 context window
- **long-term memory**：透過 memory adapter semantic search，注入 system prompt
- 兩者在 prompt 中位置不同：memory 在 system，history 在 messages

### 2. tool call 上限
- `MAX_TOOL_ITERATIONS = 3`：防止 LLM 陷入工具呼叫迴圈
- 計算方式：數 `state["messages"]` 中 type 為 `"tool"` 的訊息數量

### 3. memory adapter 作為 closure
- `memory` 和 `user_id` 透過 closure 傳入各 node function
- Graph 本身無狀態，memory 邏輯完全在 adapter 內
- 切換 adapter 只需在 `build_graph()` 傳入不同物件

### 4. save_memory 在每輪結束觸發
- 每次對話（包括工具呼叫完成後）都會執行 `save_memory`
- 傳入的是完整 message history，由 adapter 決定抽取邏輯

---

## 八、訊息類型對照表

LangGraph message type 與各層系統的對應：

| LangGraph `m.type` | LangChain class | LangMem | Mem0 |
|--------------------|----------------|---------|------|
| `"human"` | `HumanMessage` | `HumanMessage` | `"user"` |
| `"ai"` | `AIMessage` | `AIMessage` | `"assistant"` |
| `"tool"` | `ToolMessage` | 忽略 | 忽略 |
| `"system"` | `SystemMessage` | 忽略 | 忽略 |
