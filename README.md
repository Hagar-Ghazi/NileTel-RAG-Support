<div align="center">

<img src="artifacts\NileTel RAG Image.png" alt="NileTel RAG"/>

# 📡 NileTel RAG — Intelligent Customer Support System

**A production-grade Hybrid Retrieval-Augmented Generation (RAG) system for Arabic/English telecom customer support featuring multi-intent routing semantic + keyword search fusion and automated ticket escalation**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-F5820D?style=flat-square)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Semantic%20Search-4285F4?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📸 Screenshots

<table>
<tr>
<td align="center" width="33%">
<img src="screenshots/swagger_ui.png" alt="Swagger API Docs" width="100%"/>
<br/><sub><b>FastAPI Swagger UI — /query endpoint live test</b></sub>
</td>
<td align="center" width="33%">
<img src="screenshots/api_response.png" alt="API Response" width="100%"/>
<br/><sub><b>Real API response — greeting routed in 3.3 ms</b></sub>
</td>
<td align="center" width="33%">
<img src="screenshots/chat_ui.png" alt="Streamlit Chat UI" width="100%"/>
<br/><sub><b>Arabic RTL customer-facing chat interface</b></sub>
</td>
</tr>
</table>

---

## 🎬 Demo

> **▶ [Watch full demo video](C:\Users\WellCome\Desktop\NileTel RAG Assistant\artifacts\Demo.mp4)**  
> *(Shows greeting routing, knowledge-base Q&A and automatic ticket escalation flow)*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Core Components](#-core-components)
  - [RAG Engine](#1-rag-engine--rag_enginepy)
  - [FastAPI Backend](#2-fastapi-backend--apipy)
  - [Streamlit Frontend](#3-streamlit-frontend--apppy)
- [Pipeline Flow](#-pipeline-flow)
- [Query Routing Logic](#-query-routing-logic)
- [Hybrid Retrieval — FAISS + BM25 + RRF](#-hybrid-retrieval--faiss--bm25--rrf)
- [API Reference](#-api-reference)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Performance](#-performance)

---

## 🌐 Overview

NileTel RAG is a **bilingual (Arabic + English)** intelligent support assistant built for NileTel an Egyptian telecom company it answers customer queries about internet packages, billing and technical issues using a **hybrid retrieval pipeline** that combines semantic vector search with keyword-based BM25 ranking fused via **Reciprocal Rank Fusion (RRF)**

When a customer's issue requires human intervention (full outage, router failure, billing disputes) the system **automatically escalates** by collecting structured customer info through an embedded form and dispatching a support ticket to an **n8n webhook** for CRM/ticketing workflow automation

### Key Capabilities

| Capability | Details |
|---|---|
| 🌍 **Multilingual** | Native Arabic (Egyptian dialect) + English |
| 🧠 **Hybrid RAG** | FAISS semantic search + BM25 keyword, fused with RRF |
| 🔀 **Smart Routing** | 4-intent classifier: `greeting`, `chat`, `ticket`, `out_of_scope` |
| ⚡ **Sub-5ms routing** | Keyword-first lookup, LLM fallback only when ambiguous |
| 📋 **Ticket Escalation** | Structured form → n8n webhook → CRM |
| 💾 **Persistent Index** | MD5-hashed cache; rebuilds only when docs change |
| 🔁 **Multi-intent** | Handles "مرحبا + مشكلة في النت" as greeting + ticket in one query |
| 📦 **REST API** | FastAPI with batch queries, health check, and cache management |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Customer Query (AR / EN)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  QueryRouter   │  ← Keyword lists + LLM fallback
                    │  (Stage 1+2)   │
                    └───────┬────────┘
          ┌─────────────────┼──────────────────┐
          │                 │                  │
    ┌─────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
    │  greeting  │  │    ticket    │  │ out_of_scope │
    │  (canned)  │  │ (form flow)  │  │  (polite ✗)  │
    └────────────┘  └──────────────┘  └──────────────┘
          │
    ┌─────▼──────────────────────────────────────────┐
    │               HybridRetriever                  │
    │                                                │
    │   ┌─────────────────┐   ┌──────────────────┐   │
    │   │  FAISS (semantic)│   │   BM25 (keyword) │  │ 
    │   │  multilingual-e5 │   │   BM25Okapi      │  │
    │   │  cosine sim ≥0.35│   │   token overlap  │  │
    │   └────────┬─────────┘   └───────┬──────────┘  │
    │            └──────────┬───────────┘            │
    │                ┌──────▼───────┐                │
    │                │  RRF Fusion  │  k=60          │
    │                │  top-K docs  │                │
    │                └──────────────┘                │
    └───────────────────────┬────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Generator   │  ← Groq LLaMA 3.1-8B
                    │  (RAG prompt) │     NEEDS_ACTION parser
                    └───────┬───────┘
                            │
              ┌─────────────▼───────────────┐
              │         FastAPI /query      │
              │   { route, answer, sources, │
              │     needs_action, latency } │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
              │     Streamlit Chat UI       │
              │  ↳ if needs_action == YES:  │
              │    Show ticket form         │
              │    POST /ticket → n8n       │
              └─────────────────────────────┘
```

---

## 🔩 Core Components

### 1. RAG Engine — `rag_engine.py`

The heart of the system is a self-contained Python module with five classes working in concert:

#### `RecursiveChunker`
Splits markdown documents using a **priority separator cascade**: `\n\n` → `\n` → `.` → `،` → `؟` → `!` → ` `. This handles mixed Arabic/English documents with wildly different section lengths. Applies configurable overlap (default: 80 chars) to preserve cross-boundary context

```python
chunker = RecursiveChunker(chunk_size=600, chunk_overlap=80)
chunks  = chunker.split(raw_text)
```

#### `PersistenceManager`
Computes an **MD5 hash of all `.md` source files** and compares against a saved manifest. If unchanged loads pre-built index artefacts from disk but If changed rebuilds and saves:

| File | Contents |
|---|---|
| `chunks.json` | All chunk strings + source metadata |
| `embeddings.npy` | Float32 matrix (N × D) |
| `bm25.pkl` | Serialized `BM25Okapi` object |
| `manifest.json` | Source hash for cache invalidation |

#### `QueryRouter`
Two-stage multi-intent router:

**Stage 1 — Keyword matching** (< 1 ms, zero LLM cost):
- Normalizes Arabic text (removes diacritics, maps alef variants, lowercases)
- Splits on sentence boundaries: `،`, `؟`, `!`, `\n`, `كمان`, `وكمان`, `and`, `also`
- Checks each part against curated keyword lists for `greeting`, `ticket`, `out_of_scope`

**Stage 2 — LLM fallback** (only for ambiguous parts):
- Calls Groq LLaMA 3.1 with a single-word classification prompt
- Returns one of: `greeting`, `ticket`, `out_of_scope`, `chat`

**Priority resolution**: `ticket` > `out_of_scope` > `chat`

#### `HybridRetriever`
```
score(doc) = 1/(60 + rank_in_FAISS) + 1/(60 + rank_in_BM25)
```

- **FAISS**: `IndexFlatIP` on L2-normalized vectors → exact cosine similarity. Uses `intfloat/multilingual-e5-large` with `passage:` prefix for indexing and `query:` prefix at search time. Score threshold: 0.35.
- **BM25**: `BM25Okapi` on lowercased tokens. Catches exact product names, error codes, account IDs that embeddings miss.
- **RRF**: k=60 constant softens rank position advantage, making the fusion robust to outlier scores.

#### `Generator`
Builds a structured Arabic-language prompt, calls Groq, and parses the `NEEDS_ACTION: YES/NO` flag from the end of the response. The flag drives the ticket escalation flow without any extra LLM call.

---

### 2. FastAPI Backend — `api.py`

Wraps `NileTelRAG` with a production-ready REST API. The engine is initialized at startup via `@asynccontextmanager lifespan` and stored as a module-level singleton.

#### Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns engine status, dirs, n8n config |
| `GET` | `/info` | Engine metadata — model name, doc count, cache dir |
| `POST` | `/query` | Main chat endpoint — returns route, answer, sources, latency |
| `POST` | `/query/batch` | Run up to 20 queries in one request |
| `POST` | `/ticket` | Submit full customer info → fires n8n webhook |
| `DELETE` | `/cache` | Wipe index cache (restart to rebuild) |

#### Ticket Webhook (`fire_ticket`)
- Generates `TKT-YYYYMMDD-HHMMSS` ticket ID locally
- Posts full structured payload to `N8N_WEBHOOK_URL` with 6s timeout
- **Never raises** — returns `webhook_status: skipped/sent/failed` so the API always responds

---

### 3. Streamlit Frontend — `app.py`

A fully RTL Arabic chat interface with:

- **Header** with gradient brand banner and live status badge
- **Session statistics sidebar** tracking questions, tickets, chat, and out-of-scope counts
- **Example query buttons** that inject into the pipeline directly
- **Ticket collection form** — appears automatically when `needs_action == YES`:
  - Required: name, phone, account number, governorate (25 Egyptian governorates)
  - Problem classification: full outage / slow speed / router fault / billing / other
  - Service type: FTTH / ADSL / Mobile 4G
  - Full validation before submission
- **Source attribution pills** showing which knowledge-base files were cited
- **Latency display** in milliseconds per response

---

## 🔄 Pipeline Flow

```
User sends: "مرحبا، انقطع النت عندي"

1. Router splits → ["مرحبا", "انقطع النت عندي"]
   - "مرحبا" → greeting ✓
   - "انقطع النت" → ticket (keyword: "انقطع النت") ✓
   - primary = ticket, has_greeting = True

2. Returns:
   route        = "ticket"
   answer       = "أهلاً بيك يا فندم! 😊\nتمام يا فندم، هبدأ في رفع التذكرة فوراً..."
   needs_action = "YES"

3. Streamlit receives needs_action == YES
   → Shows ticket form
   → User fills name/phone/account/address
   → POST /ticket → n8n webhook → CRM system
   → Displays ticket ID: TKT-20250507-123456
```

---

## 🧭 Query Routing Logic

```
Query Input
    │
    ├─ Contains greeting keyword? ──────────────────────────────────► has_greeting = True
    │
    ├─ Split into parts at [،؟!\n, كمان, and, also]
    │
    └─ For each part:
           │
           ├─ matches _GREETINGS list?    → label: "greeting"
           ├─ matches _TICKET_TRIGGERS?   → label: "ticket"
           ├─ matches _OUT_OF_SCOPE?      → label: "out_of_scope"
           ├─ len < 8 chars?              → label: "chat" (safe default)
           └─ else → LLM classify (Groq, max_tokens=10, temp=0.0)

Priority merge: ticket > out_of_scope > chat
```

**Ticket trigger examples (Arabic):**
`افتح تذكرة` · `ابعتلي مهندس` · `انقطع النت` · `ضوء أحمر` · `فاتورة غلط` · `مقطوع تماما`

**Out-of-scope examples:**
`فيلم` · `كورة` · `طقس` · `انتخابات` · `مطعم` · `بورصة` · `دكتور`

---

## 🔍 Hybrid Retrieval — FAISS + BM25 + RRF

### Why Hybrid?

| Signal | Strength | Weakness |
|---|---|---|
| **FAISS (semantic)** | Catches paraphrases, cross-language questions | Misses exact product codes, account numbers |
| **BM25 (keyword)** | Exact token match, no embedding cost | Fails on synonyms, Arabic morphology variants |
| **RRF fusion** | Best of both, no score normalization needed | Slightly higher latency |

### RRF Formula

```
score(d) = Σ 1 / (k + rank(d, list_i))

Where:
  k    = 60   (standard constant)
  rank = position in FAISS or BM25 ranked list (1-indexed)
```

Higher `k` reduces the penalty difference between rank 1 and rank 5, making the fusion more balanced rather than winner-take-all.

### Embedding Model

`intfloat/multilingual-e5-large` — 560M parameter model supporting 100+ languages including Arabic. Uses asymmetric prompting:
- **Indexing**: `"passage: {chunk_text}"`
- **Query**: `"query: {user_question}"`

---

## 📡 API Reference

### `POST /query`

**Request:**
```json
{
  "query": "ايه باقات الإنترنت المتاحة؟"
}
```

**Response:**
```json
{
  "query":        "ايه باقات الإنترنت المتاحة؟",
  "route":        "chat",
  "answer":       "يا فندم، باقات NileTel للإنترنت المنزلي تبدأ من...",
  "needs_action": "NO",
  "sources":      ["packages.md", "faq.md"],
  "latency_ms":   312.5
}
```

**Route values:**

| Route | Meaning | `needs_action` |
|---|---|---|
| `greeting` | Casual greeting, no query | `NO` |
| `chat` | Knowledge-base answer | `NO` / `YES` |
| `ticket` | Needs engineer dispatch | `YES` |
| `out_of_scope` | Non-telecom topic | `NO` |

---

### `POST /ticket`

**Request:**
```json
{
  "original_query": "انقطع النت عندي",
  "name":           "أحمد محمد علي",
  "phone":          "01012345678",
  "account_number": "NTL-123456",
  "governorate":    "القاهرة",
  "problem_type":   "انقطاع كامل في الإنترنت",
  "since_when":     "من امبارح الصبح",
  "service_type":   "FTTH (فايبر)",
  "address":        "شارع النيل، ش 5، شقة 12"
}
```

**Response:**
```json
{
  "ticket_id":      "TKT-20250507-143022",
  "webhook_status": "sent",
  "message":        "تم رفع التذكرة بنجاح، فريق الدعم هيتواصل معاك قريباً."
}
```

---

### `POST /query/batch`

```json
{
  "queries": [
    "ايه الـ SLA؟",
    "كيفية الدفع الإلكتروني",
    "مرحبا"
  ]
}
```

Returns `results[]` array + `total_ms`.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com))
- (Optional) `N8N_WEBHOOK_URL` for ticket escalation

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/niletel-rag.git
cd niletel-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn streamlit groq faiss-cpu \
            sentence-transformers rank-bm25 numpy \
            python-dotenv httpx pydantic

# 4. Configure environment
cp .env.example .env
# Edit .env and add your keys (see Configuration section)

# 5. Add your knowledge-base documents
mkdir data
# Place your .md files in data/
# Example: data/packages.md, data/faq.md, data/sla.md

# 6. Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 7. Start the Streamlit UI (new terminal)
streamlit run app.py
```

First run will download the `multilingual-e5-large` model (~2.2 GB) and build the FAISS index. Subsequent starts load from cache in milliseconds.

---

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional — ticket escalation webhook
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/niletel-ticket

# Optional — override defaults
DATA_DIR=data
CACHE_DIR=cache
```

### Engine Tuning Parameters

These can be passed directly to `NileTelRAG()`:

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `intfloat/multilingual-e5-large` | HuggingFace embedding model |
| `groq_model` | `llama-3.1-8b-instant` | Groq chat model |
| `top_k` | `6` | Number of chunks returned by retriever |
| `score_threshold` | `0.35` | Minimum cosine similarity for FAISS results |
| `rrf_k` | `60` | RRF fusion constant |
| `chunk_size` | `600` | Target characters per chunk |
| `chunk_overlap` | `80` | Overlap between consecutive chunks |

---

## 📁 Project Structure

```
niletel-rag/
│
├── rag_engine.py          # Core RAG pipeline (Chunker, Router, Retriever, Generator)
├── api.py                 # FastAPI wrapper with all endpoints
├── app.py                 # Streamlit customer-facing chat UI
│
├── data/                  # Knowledge base place .md files here
│
├── cache/                 # Auto-generated index cache (gitignored)
│   ├── chunks.json
│   ├── embeddings.npy
│   ├── bm25.pkl
│   └── manifest.json
│
├── requirements.txt                 
├── test.py                # to test the code 
├── artifacts              # contains Demo for the system
└── README.md     
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Greeting routing latency | ~3–5 ms |
| Full RAG query (cache warm) | ~280–450 ms |
| Batch throughput | 20 queries / request |
| Embedding model | `multilingual-e5-large` (560M params) |
| LLM | Groq `llama-3.1-8b-instant` |
| Index rebuild (100 chunks) | ~8–12 seconds |
| Cache load (subsequent starts) | < 1 second |

### Cache Invalidation
The index is automatically rebuilt when any `.md` file in `data/` changes. The MD5 hash of all source files is stored in `cache/manifest.json` and checked on every startup.

---

## 🔗 n8n Integration

When a ticket is submitted via `POST /ticket`, the system fires a structured JSON payload to your configured `N8N_WEBHOOK_URL`. This can be connected to:

- **Jira / ServiceNow** — auto-create support tickets
- **WhatsApp Business API** — send ticket confirmation SMS
- **Google Sheets** — log tickets for reporting
- **Internal CRM** — update customer records

The webhook call is non-blocking with a 6-second timeout. If n8n is unreachable, the ticket ID is still generated locally and returned to the customer

---

<div align="center">

Built for **NileTel** Powered by **Groq**, **FAISS**, **BM25** and **Streamlit**

</div>
