"""
NileTel RAG — FastAPI Wrapper
==============================
Endpoints:
    POST /query          → main chat endpoint
    POST /query/batch    → run multiple queries at once
    POST /ticket         → submit ticket with full customer info → n8n
    GET  /health         → liveness check
    GET  /info           → engine metadata
    DELETE /cache        → clear cache

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rag_engine import NileTelRAG

load_dotenv()

# Config 

DATA_DIR        = os.getenv("DATA_DIR",        "data")
CACHE_DIR       = os.getenv("CACHE_DIR",       "cache")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")   # e.g. https://xxxx.ngrok-free.app/webhook/niletel-ticket

# n8n Webhook helper 

def fire_ticket(payload: dict) -> dict:
    """
    Send full ticket payload to n8n webhook.
    Always returns a ticket_id — never raises, so the API never fails.
    """
    ticket_id = "TKT-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    payload["ticket_id"] = ticket_id

    if not N8N_WEBHOOK_URL:
        print("[Webhook] N8N_WEBHOOK_URL not set — skipping")
        return {"ticket_id": ticket_id, "webhook_status": "skipped"}

    try:
        r = httpx.post(N8N_WEBHOOK_URL, json=payload, timeout=6.0)
        r.raise_for_status()
        n8n_data  = r.json() if r.content else {}
        ticket_id = n8n_data.get("ticket_id", ticket_id)
        print(f"[Webhook] ✅ Ticket sent → {ticket_id}")
        return {"ticket_id": ticket_id, "webhook_status": "sent"}
    except Exception as e:
        print(f"[Webhook] ❌ Failed: {e}")
        return {"ticket_id": ticket_id, "webhook_status": "failed"}


# Global engine 

_rag: Optional[NileTelRAG] = None


def get_engine() -> NileTelRAG:
    if _rag is None:
        raise HTTPException(
            status_code =status.HTTP_503_SERVICE_UNAVAILABLE,
            detail ="RAG engine not ready please Check server logs",
        )
    return _rag


# Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    print("\n[API] Starting up loading NileTelRAG engine...")
    try:
        _rag = NileTelRAG(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
        print("[API] Engine ready")
    except Exception as exc:
        print(f"[API] Failed: {exc}")
        _rag = None
    yield
    print("[API] Shutting down.")


# App 

app = FastAPI(
    title       = "NileTel RAG API",
    description = "Hybrid RAG for NileTel customer support with ticket collection flow",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas 

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Customer question in Arabic or English.")


class QueryResponse(BaseModel):
    query:          str
    route:          str           # greeting | ticket | out_of_scope | chat
    answer:         str
    needs_action:   str           # YES | NO
    sources:        list[str]
    latency_ms:     float


class TicketRequest(BaseModel):
    """Full customer info collected by Streamlit before firing the ticket."""
    # Original query that triggered the ticket
    original_query: str = Field(..., description="The query that triggered needs_action=YES")

    # 🔴 Required fields
    name:           str = Field(..., min_length=2, max_length=100,  description="Customer full name")
    phone:          str = Field(..., min_length=8, max_length=20,   description="Mobile number")
    account_number: str = Field(..., min_length=3, max_length=50,   description="Account / line number")
    governorate:    str = Field(..., min_length=2, max_length=100,  description="Governorate / region")

    # 🟡 Important fields
    problem_type:   str = Field(..., description="انقطاع كامل | بطء | عطل راوتر | فاتورة | أخرى")
    since_when:     str = Field(..., description="Since when the problem started")
    service_type:   str = Field(..., description="FTTH | ADSL | Mobile")
    address:        str = Field(..., min_length=5, max_length=300,  description="Detailed address")


class TicketResponse(BaseModel):
    ticket_id:      str
    webhook_status: str           # sent | skipped | failed
    message:        str


class BatchQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=20)


class BatchQueryResponse(BaseModel):
    results:  list[QueryResponse]
    total_ms: float


class HealthResponse(BaseModel):
    status:       str
    engine_ready: bool
    data_dir:     str
    cache_dir:    str
    n8n_configured: bool


class InfoResponse(BaseModel):
    engine_ready:    bool
    data_dir:        str
    cache_dir:       str
    groq_model:      str
    data_dir_exists: bool
    md_file_count:   int
    n8n_configured:  bool


# Endpoints 

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Liveness check")
def health():
    return HealthResponse(
        status         = "ok" if _rag is not None else "degraded",
        engine_ready   = _rag is not None,
        data_dir       = DATA_DIR,
        cache_dir      = CACHE_DIR,
        n8n_configured = bool(N8N_WEBHOOK_URL),
    )


@app.get("/info", response_model=InfoResponse, tags=["System"],
         summary="Engine metadata")
def info():
    data_path = Path(DATA_DIR)
    return InfoResponse(
        engine_ready    = _rag is not None,
        data_dir        = DATA_DIR,
        cache_dir       = CACHE_DIR,
        groq_model      = _rag.groq_model if _rag else "unknown",
        data_dir_exists = data_path.exists(),
        md_file_count   = len(list(data_path.glob("*.md"))) if data_path.exists() else 0,
        n8n_configured  = bool(N8N_WEBHOOK_URL),
    )


@app.post("/query", response_model=QueryResponse, tags=["Chat"],
          summary="Ask the RAG assistant")
def query(body: QueryRequest):
    """
    Main chat endpoint. Returns route + answer.
    If route == 'ticket', the frontend should collect customer info
    and then call POST /ticket to actually fire the webhook.
    """
    engine = get_engine()
    t0     = time.perf_counter()

    try:
        result = engine.query(body.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}")

    return QueryResponse(
        query        = body.query,
        route        = result["route"],
        answer       = result["answer"],
        needs_action = result["needs_action"],
        sources      = result["sources"],
        latency_ms   = round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/ticket", response_model=TicketResponse, tags=["Chat"],
          summary="Submit ticket with customer info to n8n")
def submit_ticket(body: TicketRequest):
    """
    Called after the customer fills in their info in the Streamlit form.
    Fires the full payload to n8n webhook.
    """
    payload = {
        "original_query": body.original_query,
        "name":           body.name,
        "phone":          body.phone,
        "account_number": body.account_number,
        "governorate":    body.governorate,
        "problem_type":   body.problem_type,
        "since_when":     body.since_when,
        "service_type":   body.service_type,
        "address":        body.address,
        "timestamp":      datetime.now().isoformat(),
    }

    result = fire_ticket(payload)

    return TicketResponse(
        ticket_id      = result["ticket_id"],
        webhook_status = result["webhook_status"],
        message        = "تم رفع التذكرة بنجاح، فريق الدعم هيتواصل معاك قريباً." if result["webhook_status"] != "failed"
                         else "تم تسجيل التذكرة محلياً — سيتم الإرسال لاحقاً.",
    )


@app.post("/query/batch", response_model=BatchQueryResponse, tags=["Chat"],
          summary="Run multiple queries at once (max 20)")
def query_batch(body: BatchQueryRequest):
    engine  = get_engine()
    t0      = time.perf_counter()
    results = []

    for q in body.queries:
        qt = time.perf_counter()
        try:
            r = engine.query(q)
        except Exception as exc:
            r = {"route": "error", "answer": str(exc), "needs_action": "NO", "sources": []}
        results.append(QueryResponse(
            query        = q,
            route        = r["route"],
            answer       = r["answer"],
            needs_action = r["needs_action"],
            sources      = r["sources"],
            latency_ms   = round((time.perf_counter() - qt) * 1000, 1),
        ))

    return BatchQueryResponse(results=results,
                              total_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.delete("/cache", tags=["System"], summary="Clear index cache")
def clear_cache():
    cache_path = Path(CACHE_DIR)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        return {"detail": f"Cache deleted. Restart server to rebuild."}
    return {"detail": "Cache folder does not exist."}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
