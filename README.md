# Agent Memory 框架比較

> 用同一個 LangGraph agent，比較 **LangMem** 與 **Mem0** 兩種 memory 框架對跨對話記憶品質的影響。

---

## 核心問題

當 AI agent 需要記住用戶的歷史資訊時，不同的 memory 框架會有多大的差異？

這個專案把 memory 框架做成可插拔的 adapter，在固定的 agent 架構下替換比較，最後用 LLM-as-judge 量化評估。

---

## Agent 架構

```
用戶輸入
   ↓
retrieve_memory   ── 從 memory 框架取跨對話長期記憶
   ↓
agent (LLM)       ── gpt-4o-mini，帶工具呼叫能力
   ↓         ↓
   │     tool_calls?
   │         ↓ YES（最多 3 次）
   │       tools   ── search_knowledge_base / get_user_health_profile
   │         ↓
   └── loop back
   ↓ NO
save_memory       ── 將對話存入 memory 框架
   ↓
回傳用戶
```

**兩層記憶設計**：
- **短期**（chat history）：最近 5 條訊息，控制 context window
- **長期**（memory framework）：語意搜尋召回，注入 system prompt

---

## 比較對象

| | LangMem | Mem0 | memsearch |
|---|---|---|---|
| 定位 | LangGraph 原生 | Framework-agnostic | OpenClaw 架構，framework-agnostic |
| 記憶類型 | Semantic / Episodic / Procedural | 向量 + 知識圖譜 | Markdown 對話 chunk |
| 搜尋方式 | Vector search | Vector search | **Hybrid（Dense + BM25）** |
| 本地持久化 | `.langmem_store/<user_id>.json` | `.mem0_store/`（ChromaDB + SQLite）| `.memsearch_store/<user_id>/`（.md + Milvus Lite）|
| 特色 | Procedural memory 可優化 system prompt | 實體關係抽取，跨對話連結 | **Markdown 可讀、可 Git 版控** |

---

## 應用場景

**個人健康顧問 Bot** — 記住用戶的症狀、藥物、生活習慣，跨對話給出個人化建議。

場景適合比較的原因：
- 無記憶 baseline 效果明顯（每次忘記用戶病史）
- 可測精確召回（「我上週說的藥物是什麼？」）
- 評分維度清晰：連貫性、個人化、事實準確性

---

## 快速開始

```bash
# 1. 安裝依賴
make install

# 2. 設定 API key
cp .env.example .env
# 填入 OPENAI_API_KEY

# 3. 建立 RAG 知識庫（只需執行一次）
make index

# 4. 開始對話
make chat-dummy        # 無記憶
make chat-langmem      # LangMem
make chat-mem0         # Mem0
make chat-memsearch    # memsearch（OpenClaw）
```

---

## 專案結構

```
AgentMemory/
├── agent/
│   ├── graph.py        # LangGraph graph（固定，不隨 memory 框架改變）
│   ├── state.py        # AgentState（messages + memory_context）
│   └── tools.py        # search_knowledge_base, get_user_health_profile
│
├── memory/
│   ├── base.py         # BaseMemory 抽象介面
│   ├── langmem_adapter.py
│   └── mem0_adapter.py
│
├── rag/
│   ├── indexer.py      # 一次性建立 ChromaDB 索引
│   └── retriever.py    # 查詢用，不重新 embed
│
├── data/
│   ├── knowledge_base.json       # 12 條健康知識文件
│   └── mock_conversations.json   # 3 用戶 × 10 輪模擬對話
│
├── notes/
│   ├── agent_design.md           # Agent 架構詳細說明與踩坑記錄
│   └── langmem_and_mem0_integration.md  # 兩個框架的 API 整合筆記
│
└── scripts/
    └── chat_loop.py    # 互動對話入口
```

---

## Roadmap

| Phase | 狀態 | 內容 |
|-------|------|------|
| 0 | ✅ | 環境建置、mock 資料 |
| 1 | ✅ | LangGraph agent（工具呼叫、RAG、條件式節點）|
| 2 | ✅ | LangMem adapter |
| 3 | ✅ | Mem0 adapter |
| 3b | ✅ | memsearch adapter（OpenClaw，Markdown-first）|
| 4 | 🔲 | LLM-as-judge 評估（連貫性 / 個人化 / 事實準確性）|
| 5 | 🔲 | Notebook 視覺化比較 |

---

## 技術棧

- **LangGraph** — agent orchestration
- **LangMem** — memory framework A
- **Mem0** — memory framework B
- **memsearch** — memory framework C（OpenClaw 架構）
- **ChromaDB** — RAG 向量儲存 + Mem0 後端
- **Milvus Lite** — memsearch 向量後端
- **OpenAI** — gpt-4o-mini（LLM）、text-embedding-3-small（embedding）
- **uv** — 套件管理
