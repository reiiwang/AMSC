# Memory Framework 深度比較

> 比較對象：**DummyMemory**（baseline）、**LangMem**、**Mem0**、**memsearch**（OpenClaw）、**MemGPT**（手刻）
>
> 參考來源：
> - LangMem 官方文件：https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
> - Mem0 DeepWiki：https://deepwiki.com/mem0ai/mem0
> - Mem0 論文：https://arxiv.org/html/2504.19413v1
> - memsearch GitHub：https://github.com/zilliztech/memsearch
> - memsearch Milvus Blog：https://milvus.io/blog/we-extracted-openclaws-memory-system-and-opensourced-it-memsearch.md
> - MemGPT 論文：https://arxiv.org/abs/2310.08560（Packer et al., 2023）
> - Letta 官方文件：https://docs.letta.com/concepts/memgpt/

---

## 1. Ingestion 觸發時機

| 框架 | 觸發方式 | 觸發時機 | 說明 |
|------|----------|----------|------|
| **Dummy** | 無 | 從不 | 純 no-op |
| **LangMem** | 程式碼控制（hot path 或 background） | 每輪對話結束後（或背景非同步） | 呼叫 `memory_manager.invoke()` 時觸發；框架支援兩種模式：Hot Path（同步，加 latency）和 Background（非同步，不阻塞主流程） |
| **Mem0** | 程式碼控制（hot path） | 每次呼叫 `memory.add()` 時 | 固定在對話結束後呼叫，無法自動觸發；每次 `add()` 都是同步阻塞操作 |
| **memsearch** | 程式碼控制（hot path）或檔案 watcher | 每輪對話結束後（或 `watch` 模式下檔案改變即觸發） | `save()` 寫 .md 檔後呼叫 `await mem.index()`；也支援 `memsearch watch` 以 1500ms debounce 監控目錄 |
| **MemGPT** | **LLM 自決**（tool call） | LLM 判斷對話中出現需要記憶的資訊時，主動呼叫記憶工具 | `save()` 是 no-op；記憶在 `core_memory_append` / `archival_memory_insert` 被呼叫時即時寫入；LLM 決定何時、記什麼 |

---

## 2. Ingestion 頻率

| 框架 | 頻率 | 補充說明 |
|------|------|----------|
| **Dummy** | 從不 | — |
| **LangMem** | 每輪對話 1 次（hot path）或批次（background） | Background 模式可 debounce，將多輪對話合併後批次處理，效率較高 |
| **Mem0** | 每輪對話 1 次 | 每次 `add()` 都執行完整 extraction + dedup pipeline；無批次選項 |
| **memsearch** | 每輪對話 1 次（hot path）或 debounce 1500ms（watch mode） | Indexing 時對未改變內容做 SHA-256 dedup，不重新 embed；但 index 操作本身每次都執行 |
| **MemGPT** | 按需（LLM 決定，每輪 0-N 次） | LLM 可能在一輪對話中呼叫多次記憶工具，也可能一次都不呼叫 |

---

## 3. Preprocessing（前處理）

| 框架 | 前處理步驟 | 說明 |
|------|------------|------|
| **Dummy** | 無 | — |
| **LangMem** | LLM 萃取（無顯式分塊） | 直接將完整對話 messages 送給 LLM，由 LLM 判斷哪些值得記憶；無 chunking |
| **Mem0** | Message 正規化 → LLM Fact Extraction | 先將訊息格式化（role 轉換），再呼叫 LLM 從對話中抽取離散 facts；不做 chunking |
| **memsearch** | Markdown 生成 → heading/paragraph chunking | `save()` 將對話轉成有結構的 .md 檔；index 時按 heading 和段落邊界切割成語意 chunk |
| **MemGPT** | 無（LLM 直接決定要存什麼） | LLM 從上下文中判斷並自行撰寫要存入 core memory 的內容；無自動 preprocessing |

---

## 4. Extraction / Embedding 機制

| 框架 | Extraction 方式 | Embedding 模型 | Embedding 時機 |
|------|-----------------|----------------|----------------|
| **Dummy** | 無 | 無 | 無 |
| **LangMem** | LLM tool calling（parallel）萃取記憶 | text-embedding-3-small（可換） | 每次 `store.put()` 時即時 embed；`_ensure_loaded()` reload 時不重新 embed（讀 JSON） |
| **Mem0** | **2 次 LLM call**：①抽取 facts ②決定 ADD/UPDATE/DELETE | text-embedding-3-small（可換） | 每個 fact 被抽取後即時 embed |
| **memsearch** | 無 LLM extraction；直接對原始對話文字分塊 | OpenAI text-embedding-3-small（可換；本地模式用 ONNX bge-m3） | index 時批次 embed；SHA-256 dedup 跳過未改變 chunk |
| **MemGPT** | LLM 自行撰寫記憶內容（tool call 參數） | text-embedding-3-small（archival memory 用） | `archival_memory_insert` 呼叫時即時 embed；core memory 不做 embedding（直接注入文字） |

---

## 5. 更新 / 去重 / 刪除機制

| 框架 | 去重（Dedup） | 更新（Update） | 刪除（Delete） |
|------|--------------|----------------|----------------|
| **Dummy** | 無 | 無 | 無 |
| **LangMem** | LLM 比對 existing memories，避免重複插入；`enable_inserts/updates/deletes` 控制允許哪些操作 | ✅ LLM 判斷矛盾或補充資訊時更新（preserve original UUID） | 可選（`enable_deletes=True`）；本專案關閉 |
| **Mem0** | 第二次 LLM call 比對 existing memories 決定 ADD/UPDATE/DELETE；similarity > 0.85 觸發 merge | ✅ LLM 判斷後更新 vector + history.db | ✅ 自動（LLM 判斷舊資訊過時）；有 history.db audit trail 可追蹤刪除紀錄 |
| **memsearch** | **SHA-256 hash dedup**（content-level）：相同內容永遠不重新 embed；非 LLM dedup | ❌ 無 semantic update；相同檔案 hash 相同則跳過；不同輪對話寫不同 .md 檔 | ❌ 無自動刪除；手動刪除 .md 檔後重新 index 即可 |
| **MemGPT** | 無自動 dedup；LLM 可呼叫 `core_memory_replace` 主動修正 | ✅ LLM 主動呼叫 `core_memory_replace` 修改特定文字 | ❌ 無（本實作未做）；core block 可整段改寫；archival 無刪除 tool |

---

## 6. Storage 設計與 DB 選擇

| 框架 | 儲存架構 | DB / 格式 | 持久化位置 |
|------|----------|-----------|------------|
| **Dummy** | 無 | — | — |
| **LangMem** | 單層向量 store | **InMemoryStore**（langGraph 內建）+ **JSON 檔**持久化；可換成任何 LangGraph BaseStore | `.langmem_store/<user_id>.json` |
| **Mem0** | 三層儲存：① 向量 ② 圖 ③ 歷史 | **向量**：ChromaDB / Qdrant / 24+ 種可選；**圖**：Neo4j / Kuzu / Memgraph（可選）；**歷史**：SQLite（`history.db`）| `.mem0_store/`（ChromaDB + SQLite） |
| **memsearch** | 雙層：① Markdown 原始檔（truth）② 向量索引（derived） | **向量**：Milvus Lite（`.db` 單檔）；**原始**：.md 檔案系統 | `.memsearch_store/<user_id>/` |
| **MemGPT** | 雙層：① Core Memory（JSON 文字）② Archival Memory（向量） | **Core**：JSON 檔；**Archival**：ChromaDB | `.memgpt_store/<user_id>.json` + `.memgpt_store/chroma/` |

---

## 7. Search 機制

| 框架 | Search 類型 | 步驟 | Reranking |
|------|------------|------|-----------|
| **Dummy** | 無 | — | — |
| **LangMem** | **純向量搜尋** | ① embed query → ② cosine similarity vs InMemoryStore → ③ top-K 返回 | ❌ 無 |
| **Mem0** | **向量搜尋 + 可選圖查詢** | ① embed query → ② vector similarity search → ③（可選）graph entity lookup → ④ 合併結果，附 score → ⑤ 可選 reranker | ✅ 可選（需設定） |
| **memsearch** | **Hybrid：Dense + BM25 + RRF** | ① dense embed query → ② BM25 keyword search → ③ RRF (Reciprocal Rank Fusion) 合併排名 → ④ top-K 返回 chunk + source attribution | ✅ RRF 本身即為 reranking |
| **MemGPT** | **Core**：直接注入（無搜尋）；**Archival**：向量搜尋 | Core Memory 永遠完整出現在 system prompt；Archival：① embed query → ② ChromaDB cosine search → ③ top-3 | ❌ 無 |

---

## 8. Token / Cost / Latency

| 框架 | 額外 LLM calls（per turn） | Embedding calls | 主要成本來源 | Latency 影響 |
|------|---------------------------|-----------------|--------------|--------------|
| **Dummy** | 0 | 0 | 無 | 無 |
| **LangMem** | **1 次**（memory manager extraction） | 1 次 query embed + N 次 insert embed | Memory manager LLM call（通常 200-500 tokens） | 中等；hot path 會阻塞回應 |
| **Mem0** | **2 次**（① fact extraction ② ADD/UPDATE/DELETE 決策） | 1 次 query embed + M 次 fact embed | 兩次 LLM call 最貴；fact 數量越多成本越高 | 高；兩次串行 LLM call |
| **memsearch** | **0**（無 LLM extraction） | index 時批次（dedup 跳過未改變）+ 1 次 query embed | 僅 embedding；幾乎無 LLM 成本 | 低；但 `await mem.index()` 每次都執行（有 I/O）|
| **MemGPT** | **0-N 次**（LLM 自決是否呼叫工具） | archival insert/search 各 1 次（按需） | Core memory 文字注入 system prompt（固定 token overhead）；LLM tool call 按需計費 | 取決於 LLM 決定呼叫幾次工具；core memory 越大 system prompt 越貴 |

---

## 9. 記憶管理主體（總覽）

| 框架 | 誰管記憶？ | 機制類型 |
|------|-----------|----------|
| **Dummy** | 無人 | — |
| **LangMem** | **框架程式碼**（LLM 為工具） | LLM 被呼叫來「萃取」，但觸發和儲存邏輯由程式碼控制 |
| **Mem0** | **框架程式碼**（LLM 為工具） | 兩次 LLM call 做 extraction + decision；但 pipeline 由框架驅動 |
| **memsearch** | **框架程式碼**（無 LLM） | 純程式：寫 Markdown → chunking → embedding；LLM 完全不介入記憶管理 |
| **MemGPT** | **LLM 自己**（工具呼叫） | 記憶的觸發、內容、時機完全由 LLM 判斷；框架只提供工具介面 |

---

## 10. 綜合 Tradeoff 比較

| 維度 | LangMem | Mem0 | memsearch | MemGPT |
|------|---------|------|-----------|--------|
| **記憶品質** | 高（LLM 萃取，支援更新） | 高（LLM 萃取 + dedup + graph） | 中（原文 chunk，無語意壓縮） | 取決於 LLM 判斷力 |
| **成本** | 中（1 LLM call/turn） | 高（2 LLM calls/turn） | 低（0 LLM calls） | 可變（0-N calls + system prompt token） |
| **去重能力** | 中（LLM 判斷，非精確） | 強（vector similarity + LLM） | 強但局限（SHA-256，content-level only） | 無自動 dedup |
| **可解釋性** | 中（知道 memory 內容，不知道為何萃取） | 中（有 history.db audit trail） | 高（Markdown 直接可讀） | 高（core memory 永遠可見） |
| **搜尋精度** | 中（純向量） | 高（向量 + 可選圖） | 高（hybrid BM25 + dense + RRF） | 中（core 直接注入；archival 純向量） |
| **框架耦合** | 高（依賴 LangGraph InMemoryStore） | 低（framework-agnostic） | 低（獨立 library） | 低（獨立實作） |
| **適合場景** | LangGraph 生態、需要 Procedural Memory | 複雜關係圖譜、需要 entity dedup | 文件型知識、markdown 工作流、低成本 | 展示 LLM 自主性、研究型 agent |

---

## 本專案實作對照

| 框架 | 觸發（本專案） | 儲存（本專案） | 備註 |
|------|----------------|----------------|------|
| LangMem | `save_memory` 節點，每輪對話後同步呼叫 | `.langmem_store/<user_id>.json` | 非同步 `asyncio.run()` 包裹；enable_deletes=False |
| Mem0 | `save_memory` 節點，每輪對話後同步呼叫 | `.mem0_store/`（ChromaDB + SQLite） | Role 轉換 human→user, ai→assistant |
| memsearch | `save_memory` 節點，每輪對話後同步呼叫（含 index） | `.memsearch_store/<user_id>/*.md` + Milvus Lite | retrieve 也重新 index（確保新檔被納入） |
| MemGPT | LLM tool call（隨時，0-N 次/turn） | `.memgpt_store/<user_id>.json` + `.memgpt_store/chroma/` | `save()` no-op；4 個工具注入 agent |
