# MemGPT 設計哲學筆記

## 資料來源聲明

本筆記與對應實作（`memory/memgpt_adapter.py`）的設計依據：

1. **原始論文概念摘要**（透過 leoniemonigatti.com 的論文整理頁面取得）
   - 論文原名：*MemGPT: Towards LLMs as Operating Systems*
   - 作者：Packer et al.（UC Berkeley）
   - 來源頁：https://www.leoniemonigatti.com/papers/memgpt.html

2. **Letta（MemGPT 延伸框架）的實作細節**
   - 來源文章：*Stateful AI Agents: A Deep Dive into Letta (MemGPT) Memory Models*
   - URL：https://medium.com/@piyush.jhamb4u/stateful-ai-agents-a-deep-dive-into-letta-memgpt-memory-models-a2ffc01a7ea1

3. **Letta 官方文件與 GitHub**
   - https://docs.letta.com/concepts/memgpt/
   - https://github.com/letta-ai/letta

> ⚠️ **重要說明**：`memgpt_adapter.py` 是依據上述資料對 MemGPT 設計哲學的**自行實作**，
> 並非使用 `letta` 官方套件。官方套件需要 Docker server，不符合本專案純本地 Python 的架構。
> 實作忠於 MemGPT 的核心概念，但細節（如 token 計算、heartbeat 機制）做了簡化。

---

## MemGPT 的核心設計哲學：LLM as Operating System

MemGPT 的出發點是：LLM 的 context window 是有限的（如同 CPU 的 RAM），
但人類的記憶需求是無限的（如同磁碟）。

解法：**借鑑 OS 的虛擬記憶體（Virtual Memory）機制——在 RAM 和磁碟之間做 paging**。

```
傳統 OS：
  RAM（快，有限）  ←→  磁碟（慢，無限）

MemGPT：
  Context Window（快，有限）  ←→  External Storage（慢，無限）
```

最關鍵的差異：**記憶管理的主體是 LLM 自己**，它透過 tool call 決定什麼時候要存、要取、要覆蓋。

---

## 記憶三層架構

### Tier 1：Main Context（在 context window 裡）

#### 1a. Core Memory（核心記憶區）
- 永遠出現在 system prompt 裡
- 有字元上限（如 2000 字元）
- 存放最重要的持久資訊：用戶基本資料、agent persona
- **LLM 可透過 tool call 主動修改**

結構：
```
[persona block]   ← agent 的身份與行為準則
[human block]     ← 用戶的重要資訊（姓名、病史、偏好）
```

每個 block 包含：
- `label`：識別名稱（"persona" / "human"）
- `value`：實際內容字串
- `char_limit`：字元上限

#### 1b. Conversation History（對話歷史）
- FIFO queue，context window 裡保留最近的對話
- 當歷史過長時，最舊的被 evict 到 Recall Storage

---

### Tier 2：External Storage（在 context window 外）

#### 2a. Recall Storage（對話回顧庫）
- 完整的歷史對話，用 text search 查詢
- 當 agent 需要「回憶之前說過的話」時，呼叫 `conversation_search` 取回

#### 2b. Archival Storage（封存知識庫）
- 無限容量，用 vector embedding search 查詢
- Core Memory 放不下的資訊可以 offload 到這裡
- agent 透過 `archival_memory_insert` / `archival_memory_search` 存取

---

## 六個記憶管理工具（Agent 自己呼叫）

| Tool | 作用 |
|------|------|
| `core_memory_append(label, content)` | 在指定 block 後面附加新內容 |
| `core_memory_replace(label, old, new)` | 替換 block 中的特定字串 |
| `archival_memory_insert(content)` | 寫入封存知識庫 |
| `archival_memory_search(query)` | 從封存知識庫語意搜尋 |
| `conversation_search(query)` | 從對話歷史 text search |
| `send_message(message)` | 回覆用戶（唯一能輸出給用戶的工具）|

重點：**agent 不直接輸出文字，所有回應都透過 `send_message` tool**。
這讓 agent 可以在「回應前先更新記憶」，或「一次執行多個 memory 操作」。

---

## Paging 機制（Heartbeat）

當 context window 快滿時：
1. Agent 呼叫 memory 工具把重要資訊存到 Archival
2. 系統 evict 最舊的對話到 Recall Storage
3. Agent 可以設定 heartbeat，讓自己在下一輪繼續執行（不回應用戶）

這讓 agent 可以做「多步記憶整理」而不中斷對話。

---

## 與其他框架的根本差異

| | LangMem / Mem0 / memsearch | **MemGPT** |
|---|---|---|
| 記憶管理主體 | 框架程式碼（你寫的邏輯）| **LLM 自己**（tool call）|
| 觸發時機 | 對話結束後外部呼叫 | **LLM 自己決定何時存取** |
| Core Memory | 無 | 永遠在 system prompt，有字元上限 |
| 記憶超出處理 | 新的覆蓋舊的 | LLM 決定要 offload 還是 replace |
| 架構哲學 | memory as service | **memory as agent capability** |

---

## 本專案實作說明（`memgpt_adapter.py`）

### 忠實實作的部分
- Core Memory block（`human` + `persona`），有字元上限，注入 system prompt
- `core_memory_append` 和 `core_memory_replace` 作為 agent 可呼叫的 tool
- Archival Memory（ChromaDB vector search）作為 overflow
- LLM 自主決定何時更新記憶

### 簡化/省略的部分
- Recall Storage（conversation_search）：本專案已有 chat history，不重複實作
- Heartbeat 機制：簡化為單輪 tool call loop
- Token 精確計算：用字元數代替
- 完整的 context eviction 策略

### 與 BaseMemory 介面的調整
MemGPT 的邏輯是「agent 主動管記憶」，與 `save()/retrieve()` 的被動介面有根本差異。
解決方式：
- `retrieve()`：回傳當前 core memory 內容，注入 system prompt
- `save()`：no-op（記憶由 agent 透過 tool 自己管）
- 新增兩個 tool 加入 `TOOLS` list，讓 graph 的 agent 節點可以呼叫
