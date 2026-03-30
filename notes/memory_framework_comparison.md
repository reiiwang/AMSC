# Memory Framework 深度比較

> 比較對象：**DummyMemory**（baseline）、**LangMem**、**Mem0**、**OpenClaw（原版）**、**memsearch（本專案實作）**、**MemGPT**（手刻）、**A-MEM**（研究參考）
>
> ⚠️ **OpenClaw vs memsearch 的重要區別**：
> OpenClaw 原版是 LLM-managed memory（LLM 自己決定寫什麼到 Markdown），與 MemGPT 哲學相近。
> memsearch 是 Zilliz 抽出 OpenClaw **搜尋層**開源的 library，不限制誰來寫 Markdown。
> 本專案的 `MemsearchAdapter` 選擇讓程式碼直接寫對話 Markdown（0 LLM call），**偏離了 OpenClaw 原版設計**。
>
> 參考來源：
> - LangMem 官方文件：https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
> - Mem0 DeepWiki：https://deepwiki.com/mem0ai/mem0
> - Mem0 論文：https://arxiv.org/html/2504.19413v1
> - OpenClaw Memory 文件：https://docs.openclaw.ai/concepts/memory
> - memsearch GitHub：https://github.com/zilliztech/memsearch
> - memsearch Milvus Blog：https://milvus.io/blog/we-extracted-openclaws-memory-system-and-opensourced-it-memsearch.md
> - MemGPT 論文：https://arxiv.org/abs/2310.08560（Packer et al., 2023）
> - Letta 官方文件：https://docs.letta.com/concepts/memgpt/
> - A-MEM 論文：https://arxiv.org/abs/2502.12110（Xu et al., 2025, NeurIPS）
> - A-MEM GitHub：https://github.com/agiresearch/A-mem

---

## 1. Ingestion 觸發時機

| 框架 | 觸發方式 | 觸發時機 | 說明 |
|------|----------|----------|------|
| **Dummy** | 無 | 從不 | 純 no-op |
| **LangMem** | 程式碼控制（hot path 或 background） | 每輪對話結束後（或背景非同步） | 呼叫 `memory_manager.invoke()` 時觸發；框架支援兩種模式：Hot Path（同步，加 latency）和 Background（非同步，不阻塞主流程） |
| **Mem0** | 程式碼控制（hot path） | 每次呼叫 `memory.add()` 時 | 固定在對話結束後呼叫，無法自動觸發；每次 `add()` 都是同步阻塞操作 |
| **OpenClaw（原版）** | **LLM 自決**（主動寫檔） | LLM 判斷對話中有值得記憶的資訊時主動觸發 | 類似 MemGPT；LLM 主動將資訊寫入 `memory/YYYY-MM-DD.md` 或 `MEMORY.md`；context 壓縮前有 automatic memory flush |
| **memsearch（本專案）** | 程式碼控制（hot path）或檔案 watcher | 每輪對話結束後（或 `watch` 模式下檔案改變即觸發） | `save()` 寫 .md 檔後呼叫 `await mem.index()`；也支援 `memsearch watch` 以 1500ms debounce 監控目錄。**注意：本專案讓程式碼寫 Markdown，非 OpenClaw 原版設計** |
| **MemGPT** | **LLM 自決**（tool call） | LLM 判斷對話中出現需要記憶的資訊時，主動呼叫記憶工具 | `save()` 是 no-op；記憶在 `core_memory_append` / `archival_memory_insert` 被呼叫時即時寫入；LLM 決定何時、記什麼 |
| **A-MEM** | 程式碼控制（hot path） | 每次呼叫 `add_note()` 時 | 添加新記憶時，LLM 自動分析歷史記憶建立連結並更新相關記憶的 metadata；多次 LLM call 發生在 `add_note()` 內部 |

---

## 2. Ingestion 頻率

| 框架 | 頻率 | 補充說明 |
|------|------|----------|
| **Dummy** | 從不 | — |
| **LangMem** | 每輪對話 1 次（hot path）或批次（background） | Background 模式可 debounce，將多輪對話合併後批次處理，效率較高 |
| **Mem0** | 每輪對話 1 次 | 每次 `add()` 都執行完整 extraction + dedup pipeline；無批次選項 |
| **OpenClaw（原版）** | 按需（LLM 決定，每輪 0-N 次）；flush 在 context 壓縮前 | 與 MemGPT 類似；LLM 可能不寫、也可能寫多條 |
| **memsearch（本專案）** | 每輪對話 1 次（hot path）或 debounce 1500ms（watch mode） | Indexing 時對未改變內容做 SHA-256 dedup，不重新 embed；但 index 操作本身每次都執行 |
| **MemGPT** | 按需（LLM 決定，每輪 0-N 次） | LLM 可能在一輪對話中呼叫多次記憶工具，也可能一次都不呼叫 |
| **A-MEM** | 每輪對話 1 次 | 每次 `add_note()` 內部執行多次 LLM call（note 生成 + 連結分析 + 相關記憶更新）；無批次選項 |

---

## 3. Preprocessing（前處理）

| 框架 | 前處理步驟 | 說明 |
|------|------------|------|
| **Dummy** | 無 | — |
| **LangMem** | LLM 萃取（無顯式分塊） | 直接將完整對話 messages 送給 LLM，由 LLM 判斷哪些值得記憶；無 chunking |
| **Mem0** | Message 正規化 → LLM Fact Extraction | 先將訊息格式化（role 轉換），再呼叫 LLM 從對話中抽取離散 facts；不做 chunking |
| **OpenClaw（原版）** | LLM 自行撰寫（已是精煉後的事實） | LLM 直接產出「值得記憶的摘要」，不記錄原始對話；兩層：daily log vs curated MEMORY.md |
| **memsearch（本專案）** | Markdown 生成 → heading/paragraph chunking | `save()` 將**原始對話**轉成 .md 檔；index 時按 heading 和段落邊界切割成語意 chunk |
| **MemGPT** | 無（LLM 直接決定要存什麼） | LLM 從上下文中判斷並自行撰寫要存入 core memory 的內容；無自動 preprocessing |
| **A-MEM** | LLM 生成結構化 Note（Zettelkasten 風格） | LLM 將原始內容轉換成含 keywords、tags、contextual description、connections 的結構化記憶 note；不做顯式 chunking |

---

## 4. Extraction / Embedding 機制

| 框架 | Extraction 方式 | Embedding 模型 | Embedding 時機 |
|------|-----------------|----------------|----------------|
| **Dummy** | 無 | 無 | 無 |
| **LangMem** | LLM tool calling（parallel）萃取記憶 | text-embedding-3-small（可換） | 每次 `store.put()` 時即時 embed；`_ensure_loaded()` reload 時不重新 embed（讀 JSON） |
| **Mem0** | **2 次 LLM call**：①抽取 facts ②決定 ADD/UPDATE/DELETE | text-embedding-3-small（可換） | 每個 fact 被抽取後即時 embed |
| **OpenClaw（原版）** | LLM 自行萃取（寫 Markdown 的過程即是 extraction） | OpenAI（預設，可換） | LLM 寫完 Markdown 後由 memsearch 觸發 index |
| **memsearch（本專案）** | 無 LLM extraction；直接對原始對話文字分塊 | OpenAI text-embedding-3-small（可換；本地模式用 ONNX bge-m3） | index 時批次 embed；SHA-256 dedup 跳過未改變 chunk |
| **MemGPT** | LLM 自行撰寫記憶內容（tool call 參數） | text-embedding-3-small（archival memory 用） | `archival_memory_insert` 呼叫時即時 embed；core memory 不做 embedding（直接注入文字） |
| **A-MEM** | **多次 LLM call**：① 生成 note 結構（keywords/tags/context）② 分析歷史記憶建立連結 ③ 更新被連結記憶的 context | all-MiniLM-L6-v2（預設，可換 OpenAI） | `add_note()` 時即時 embed；每個結構化 note 獨立 embed |

---

## 5. 更新 / 去重 / 刪除機制

| 框架 | 去重（Dedup） | 更新（Update） | 刪除（Delete） |
|------|--------------|----------------|----------------|
| **Dummy** | 無 | 無 | 無 |
| **LangMem** | LLM 比對 existing memories，避免重複插入；`enable_inserts/updates/deletes` 控制允許哪些操作 | ✅ LLM 判斷矛盾或補充資訊時更新（preserve original UUID） | 可選（`enable_deletes=True`）；本專案關閉 |
| **Mem0** | 第二次 LLM call 比對 existing memories 決定 ADD/UPDATE/DELETE；similarity > 0.85 觸發 merge | ✅ LLM 判斷後更新 vector + history.db | ✅ 自動（LLM 判斷舊資訊過時）；有 history.db audit trail 可追蹤刪除紀錄 |
| **OpenClaw（原版）** | LLM 判斷後選擇寫入 MEMORY.md 覆蓋舊資訊（semantic dedup by LLM） | ✅ LLM 直接改寫 MEMORY.md 中的舊內容 | ✅ LLM 可刪除 MEMORY.md 中的過時條目 |
| **memsearch（本專案）** | **SHA-256 hash dedup**（content-level）：相同內容永遠不重新 embed；非 LLM dedup | ❌ 無 semantic update；相同檔案 hash 相同則跳過；不同輪對話寫不同 .md 檔 | ❌ 無自動刪除；手動刪除 .md 檔後重新 index 即可 |
| **MemGPT** | 無自動 dedup；LLM 可呼叫 `core_memory_replace` 主動修正 | ✅ LLM 主動呼叫 `core_memory_replace` 修改特定文字 | ❌ 無（本實作未做）；core block 可整段改寫；archival 無刪除 tool |
| **A-MEM** | **語意 dedup via linking**：新記憶加入時 LLM 分析相似舊記憶，相似者更新 context 而非重複插入 | ✅ 新記憶觸發相關歷史記憶的 contextual description 自動更新（memory evolution） | ✅ 提供 `delete(memory_id)` API；但無自動觸發刪除 |

---

## 6. Storage 設計與 DB 選擇

| 框架 | 儲存架構 | DB / 格式 | 持久化位置 |
|------|----------|-----------|------------|
| **Dummy** | 無 | — | — |
| **LangMem** | 單層向量 store | **InMemoryStore**（langGraph 內建）+ **JSON 檔**持久化；可換成任何 LangGraph BaseStore | `.langmem_store/<user_id>.json` |
| **Mem0** | 三層儲存：① 向量 ② 圖 ③ 歷史 | **向量**：ChromaDB / Qdrant / 24+ 種可選；**圖**：Neo4j / Kuzu / Memgraph（可選）；**歷史**：SQLite（`history.db`）| `.mem0_store/`（ChromaDB + SQLite） |
| **OpenClaw（原版）** | 雙層：① Markdown 檔（LLM 精煉後的事實）② 向量索引（memsearch） | **向量**：Milvus（或 Lite）；**原始**：.md 檔案系統 | agent workspace 下的 `memory/` 目錄 |
| **memsearch（本專案）** | 雙層：① Markdown 原始對話檔 ② 向量索引（derived） | **向量**：Milvus Lite（`.db` 單檔）；**原始**：.md 檔案系統 | `.memsearch_store/<user_id>/` |
| **MemGPT** | 雙層：① Core Memory（JSON 文字）② Archival Memory（向量） | **Core**：JSON 檔；**Archival**：ChromaDB | `.memgpt_store/<user_id>.json` + `.memgpt_store/chroma/` |
| **A-MEM** | 單層向量 store + 記憶間連結圖（存在 metadata 中） | **向量**：ChromaDB；**連結**：存在每個 note 的 metadata 欄位（非獨立 graph DB） | ChromaDB persistent store（預設路徑可設定） |

---

## 7. Search 機制

| 框架 | Search 類型 | 步驟 | Reranking |
|------|------------|------|-----------|
| **Dummy** | 無 | — | — |
| **LangMem** | **純向量搜尋** | ① embed query → ② cosine similarity vs InMemoryStore → ③ top-K 返回 | ❌ 無 |
| **Mem0** | **向量搜尋 + 可選圖查詢** | ① embed query → ② vector similarity search → ③（可選）graph entity lookup → ④ 合併結果，附 score → ⑤ 可選 reranker | ✅ 可選（需設定） |
| **OpenClaw（原版）** | **Hybrid：Dense + BM25 + RRF**（同 memsearch） | 同 memsearch；但搜到的是 LLM 精煉事實而非原始對話 | ✅ RRF |
| **memsearch（本專案）** | **Hybrid：Dense + BM25 + RRF** | ① dense embed query → ② BM25 keyword search → ③ RRF (Reciprocal Rank Fusion) 合併排名 → ④ top-K 返回 chunk + source attribution | ✅ RRF 本身即為 reranking |
| **MemGPT** | **Core**：直接注入（無搜尋）；**Archival**：向量搜尋 | Core Memory 永遠完整出現在 system prompt；Archival：① embed query → ② ChromaDB cosine search → ③ top-3 | ❌ 無 |
| **A-MEM** | **向量搜尋 + 連結圖擴展** | ① embed query → ② ChromaDB similarity search（`search_agentic`）→ ③ 沿記憶連結展開相關 notes → ④ top-K 返回（含 keywords/tags/context） | ❌ 無明確 reranker；但連結展開本身有語意過濾效果 |

---

## 8. Token / Cost / Latency

| 框架 | 額外 LLM calls（per turn） | Embedding calls | 主要成本來源 | Latency 影響 |
|------|---------------------------|-----------------|--------------|--------------|
| **Dummy** | 0 | 0 | 無 | 無 |
| **LangMem** | **1 次**（memory manager extraction） | 1 次 query embed + N 次 insert embed | Memory manager LLM call（通常 200-500 tokens） | 中等；hot path 會阻塞回應 |
| **Mem0** | **2 次**（① fact extraction ② ADD/UPDATE/DELETE 決策） | 1 次 query embed + M 次 fact embed | 兩次 LLM call 最貴；fact 數量越多成本越高 | 高；兩次串行 LLM call |
| **OpenClaw（原版）** | **1-N 次**（LLM 決定寫什麼） | LLM 寫完後 index | LLM 寫 Markdown 的費用（與 MemGPT 相近） | 取決於 LLM 決定；context flush 時有固定成本 |
| **memsearch（本專案）** | **0**（無 LLM extraction） | index 時批次（dedup 跳過未改變）+ 1 次 query embed | 僅 embedding；幾乎無 LLM 成本 | 低；但 `await mem.index()` 每次都執行（有 I/O）|
| **MemGPT** | **0-N 次**（LLM 自決是否呼叫工具） | archival insert/search 各 1 次（按需） | Core memory 文字注入 system prompt（固定 token overhead）；LLM tool call 按需計費 | 取決於 LLM 決定呼叫幾次工具；core memory 越大 system prompt 越貴 |
| **A-MEM** | **2-3+ 次**（① note 生成 ② 連結分析 ③ 被連結記憶的 context 更新，每條受影響記憶都觸發）| 1 次 query embed + 1 次 insert embed | 最貴：LLM call 數量隨記憶庫增大而增加（連結越多更新越多）；適合品質優先場景 | 高；ingestion 時的 LLM call 數與歷史記憶量正相關 |

---

## 9. 記憶管理主體（總覽）

| 框架 | 誰管記憶？ | 機制類型 |
|------|-----------|----------|
| **Dummy** | 無人 | — |
| **LangMem** | **框架程式碼**（LLM 為工具） | LLM 被呼叫來「萃取」，但觸發和儲存邏輯由程式碼控制 |
| **Mem0** | **框架程式碼**（LLM 為工具） | 兩次 LLM call 做 extraction + decision；但 pipeline 由框架驅動 |
| **OpenClaw（原版）** | **LLM 自己** | LLM 主動判斷並寫入 Markdown；memsearch 負責搜尋層 |
| **memsearch（本專案）** | **框架程式碼**（無 LLM） | 本專案選擇讓程式碼寫 Markdown，偏離 OpenClaw 原版設計；LLM 完全不介入記憶管理 |
| **MemGPT** | **LLM 自己**（工具呼叫） | 記憶的觸發、內容、時機完全由 LLM 判斷；框架只提供工具介面 |
| **A-MEM** | **框架程式碼**（LLM 為核心工具） | `add_note()` 由程式碼呼叫，但 note 內容、連結、演化全由 LLM 執行；LLM 是記憶組織的主角，程式碼控制觸發時機 |

---

## 10. 綜合 Tradeoff 比較

| 維度 | LangMem | Mem0 | OpenClaw（原版） | memsearch（本專案） | MemGPT | A-MEM |
|------|---------|------|-----------------|---------------------|--------|-------|
| **記憶品質** | 高（LLM 萃取，支援更新） | 高（LLM 萃取 + dedup + graph） | 高（LLM 精煉後才寫入） | 中（原文 chunk，無語意壓縮） | 取決於 LLM 判斷力 | **最高**（Zettelkasten 結構 + 連結演化） |
| **成本** | 中（1 LLM call/turn） | 高（2 LLM calls/turn） | 可變（LLM 按需寫） | 低（0 LLM calls） | 可變（0-N calls + system prompt token） | **最高**（2-3+ calls/turn，隨記憶庫成長） |
| **去重能力** | 中（LLM 判斷，非精確） | 強（vector similarity + LLM） | 中（LLM 判斷是否更新 MEMORY.md） | 強但局限（SHA-256，content-level only） | 無自動 dedup | 強（連結機制自然去重，新記憶更新舊記憶而非重複插入） |
| **可解釋性** | 中 | 中（有 history.db audit trail） | 高（Markdown 人類可讀，LLM 精煉後） | 高（Markdown 直接可讀，但是原始對話） | 高（core memory 永遠可見） | 高（每條記憶有 keywords/tags/connections 可讀） |
| **搜尋精度** | 中（純向量） | 高（向量 + 可選圖） | 高（hybrid BM25 + dense + RRF） | 高（hybrid BM25 + dense + RRF） | 中（core 直接注入；archival 純向量） | 高（向量 + 連結圖擴展） |
| **框架耦合** | 高（依賴 LangGraph InMemoryStore） | 低（framework-agnostic） | 低（agent workspace file-based） | 低（獨立 library） | 低（獨立實作） | 低（獨立 library，pip install） |
| **適合場景** | LangGraph 生態、需要 Procedural Memory | 複雜關係圖譜、需要 entity dedup | 需要 LLM 自主管理 + 可讀性高的長期記憶 | 文件型知識、markdown 工作流、低成本 | 展示 LLM 自主性、研究型 agent | 長期複雜對話、需要跨記憶連結推理、品質優先（成本不敏感） |

---

## 本專案實作對照

| 框架 | 觸發（本專案） | 儲存（本專案） | 備註 |
|------|----------------|----------------|------|
| LangMem | `save_memory` 節點，每輪對話後同步呼叫 | `.langmem_store/<user_id>.json` | 非同步 `asyncio.run()` 包裹；enable_deletes=False |
| Mem0 | `save_memory` 節點，每輪對話後同步呼叫 | `.mem0_store/`（ChromaDB + SQLite） | Role 轉換 human→user, ai→assistant |
| memsearch | `save_memory` 節點，每輪對話後同步呼叫（含 index） | `.memsearch_store/<user_id>/*.md` + Milvus Lite | retrieve 也重新 index（確保新檔被納入） |
| MemGPT | LLM tool call（隨時，0-N 次/turn） | `.memgpt_store/<user_id>.json` + `.memgpt_store/chroma/` | `save()` no-op；4 個工具注入 agent |
