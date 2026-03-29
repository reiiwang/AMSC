# Agent Memory 框架比較 — 學習專案

## 目標
用固定的 LangGraph agent，比較不同 memory 框架（LangMem vs Mem0）對跨對話記憶品質的影響，並用 LLM-as-judge 做量化評估。

## 核心概念

**比較對象（變數）：Memory 框架**

| | LangMem | Mem0 |
|---|---|---|
| 定位 | LangGraph 原生 | Framework-agnostic |
| Memory 類型 | Semantic / Episodic / Procedural | 向量 + Graph（知識圖譜）|
| 儲存 | 自管（接 vector store） | 自管或託管服務 |
| 特色 | Procedural memory 可自動優化 system prompt | 實體關係抽取，跨對話連結 |

**固定不變：LangGraph agent 骨架**（同一套 graph，只換 memory adapter）

---

## 應用場景：個人健康顧問 Bot

用戶跨對話記錄症狀、習慣、藥物，agent 需記住這些資訊才能給出個人化回應。

為何適合這個主題：
- 無記憶 baseline：每次忘記用戶背景，明顯對比
- 精確召回測試："我上週說的藥物是什麼？"
- LLM-judge 三個評分維度：連貫性、個人化、事實準確性

---

## 資料夾結構

```
AgentMemory/
├── PLAN.md
├── pyproject.toml
├── .env.example
│
├── data/
│   ├── mock_conversations.json   # 10 用戶 × 5 輪對話
│   └── eval_cases.json           # 20 題需記憶才能回答的問題
│
├── agent/
│   ├── graph.py                  # LangGraph 主 graph（固定）
│   ├── nodes.py                  # agent nodes
│   └── state.py                  # graph state 定義
│
├── memory/
│   ├── base.py                   # 統一 Memory 介面（抽象層）
│   ├── langmem_adapter.py        # LangMem 接入
│   └── mem0_adapter.py           # Mem0 接入
│
├── evaluation/
│   ├── judge.py                  # LLM-as-judge 主邏輯
│   ├── metrics.py                # 評分維度定義
│   └── run_eval.py               # 執行評估 + 輸出比較報告
│
└── notebooks/
    └── comparison_demo.ipynb     # 互動展示兩種框架差異
```

---

## Phase 任務拆分

### Phase 0 — 環境與 Mock 資料
- [x] 初始化 uv 專案，安裝 langgraph / anthropic / langmem / mem0ai
- [x] 設計 `mock_conversations.json`（3 個用戶，各 10 輪對話，健康顧問場景）
- [ ] 設計 `eval_cases.json`（20 題：需要記憶才能正確回答）（暫緩）

### Phase 1 — LangGraph Agent 骨架
- [x] 定義 `state.py`（包含 memory context slot）
- [x] 建 `graph.py`（條件式節點：retrieve_memory → agent ⇄ tools → save_memory）
- [x] 實作 `agent/tools.py`（search_knowledge_base、get_user_health_profile）
- [x] 建 `rag/indexer.py`（OpenAI text-embedding-3-small + ChromaDB 持久化）
- [x] 建 `rag/retriever.py`（只做 retrieval，不重新 embed）
- [x] 建 `data/knowledge_base.json`（12 條健康知識）
- [x] 接 DummyMemory，graph 骨架完成

### Phase 2 — 接入 LangMem
- [x] 實作 `base.py` 抽象介面（`save` / `retrieve` 兩個方法）
- [x] 實作 `langmem_adapter.py`（InMemoryStore + text-embedding-3-small + create_memory_manager）
- [x] 跑 smoke test，驗證 memory 寫入/讀取正常

### Phase 3 — 接入 Mem0
- [x] 實作 `mem0_adapter.py`（同介面）
- [x] 使用 ChromaDB 持久化至 `.mem0_store/`（與 RAG 的 `.chroma/` 分離）
- [x] smoke test 通過

### Phase 3b — 接入 memsearch（OpenClaw 架構）
- [x] 實作 `memsearch_adapter.py`（Markdown-first，hybrid vector + BM25 search）
- [x] 持久化：`.memsearch_store/<user_id>/` 存 .md 檔 + Milvus Lite .db
- [x] smoke test 通過

### Phase 3c — MemGPT 設計哲學實作
- [x] 撰寫設計哲學筆記 `notes/memgpt_design.md`（含資料來源聲明）
- [x] 實作 `memgpt_adapter.py`：Core Memory blocks + Archival Memory（ChromaDB）
- [x] 新增四個記憶管理 tools：`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`
- [x] 更新 `build_graph()` 支援 `get_tools()` 擴充介面
- [x] smoke test 通過

### Phase 4 — LLM-as-Judge 評估
- [ ] `metrics.py`：定義三個評分維度（各 1-5 分）
- [ ] `judge.py`：呼叫 Claude 對每個回答評分，輸出 JSON
- [ ] `run_eval.py`：對兩組 adapter 跑完整評估，輸出比較報告

### Phase 5 — Notebook 展示
- [ ] 側邊比較兩個框架在相同問題上的回答與分數
- [ ] 分析各框架適合的場景與 tradeoff

---

## 評估設計

### 評分維度
| 維度 | 說明 | 滿分 |
|------|------|------|
| 連貫性 | 回答是否與對話歷史一致 | 5 |
| 個人化 | 是否用到用戶的個人資訊 | 5 |
| 事實準確性 | 記憶召回是否正確 | 5 |

### eval_cases 設計原則
- 每題需要跨至少 2 輪對話的記憶才能正確回答
- 包含：事實記憶（藥物名稱）、偏好記憶（飲食習慣）、時序記憶（上週 vs 這週）

---

## 技術選擇說明

| 面向 | 選擇 | 原因 |
|------|------|------|
| Graph framework | LangGraph | 有狀態 graph，易於插拔 memory layer |
| Memory 框架 A | LangMem | LangGraph 原生整合，學習成本低 |
| Memory 框架 B | Mem0 | Graph memory 架構差異最大，對比明顯 |
| LLM | Claude (claude-sonnet-4-6) | 統一用同一模型，確保評估公平 |
| 套件管理 | uv | 快速、現代 |
