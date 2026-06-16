"""
FastAPI server — exposes Nexus as a REST API.

Endpoints:
  POST /chat            — send a question, get an answer + sources
  POST /feedback        — submit like / dislike for an interaction
  GET  /feedback/stats  — RLHF dashboard metrics
  GET  /feedback/history— recent interaction history
  POST /ingest          — upload and ingest documents
  GET  /health          — service health check
  GET  /health/gemini   — check Google Gemini API status
  GET  /                — serve the chat frontend
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import shutil

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from builder import build_graph
from store import FeedbackStore
from rlhf import get_rlhf_manager
from models import FeedbackRequest, FeedbackResponse
from config import DATA_DIR, GOOGLE_API_KEY, GEMINI_MODEL, DEBUG


# ═══════════════════════════════════════════════════════════════
#  App Initialisation
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Nexus",
    description=(
        "Agentic Corrective RAG with Hybrid Retrieval "
        "(Dense ChromaDB + Sparse BM25 + RRF Fusion) "
        "and RLHF feedback adaptation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile the LangGraph once at startup
_graph = build_graph()
_feedback_store = FeedbackStore()

# Warm up/Initialize ChromaDB singleton once at startup
from dense import get_vectorstore
try:
    get_vectorstore()
    print("[OK] ChromaDB singleton initialized successfully", flush=True)
except Exception as e:
    print(f"[WARN] Failed to warm up ChromaDB singleton at startup: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════
#  Request / Response Schemas
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    interaction_id: str
    answer: str
    sources: list[dict]
    metadata: dict
    debug: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    graph_nodes: list[str]


class GeminiHealthResponse(BaseModel):
    status: str
    model: str
    available: bool
    error: Optional[str] = None


class SettingsModel(BaseModel):
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k_constant: int
    max_rewrite_attempts: int
    max_hallucination_retries: int
    dense_weight: float
    sparse_weight: float


# ═══════════════════════════════════════════════════════════════
#  Default Graph Input
# ═══════════════════════════════════════════════════════════════

def _default_state(question: str) -> dict:
    """Build a clean initial state dict for graph invocation."""
    return {
        "question": question,
        "rewritten_query": "",
        "generation": "",
        "documents": [],
        "dense_docs": [],
        "sparse_docs": [],
        "fused_docs": [],
        "web_documents": [],
        "query_rewrites": [],
        "rewrite_count": 0,
        "retrieval_retries": 0,
        "relevance_score": 0.0,
        "hallucination_retries": 0,
        "interaction_id": "",
        "route_decision": "",
        "is_grounded": True,
    }


# ═══════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the chat UI."""
    index = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")
    if not os.path.exists(index):
        return JSONResponse({"error": "Frontend not found"}, status_code=404)
    return FileResponse(index)


# ── Title Generation & Sanitisation Helpers ──
import re

FILLER_PREFIXES = [
    "based on the provided data",
    "based on the provided",
    "based on the documents",
    "based on the information",
    "according to the context",
    "according to the documents",
    "the provided data",
    "based on",
]

GREETING_PATTERNS = [
    "hello", "hi there", "hi!", "how can i help",
    "how may i help", "what can i help",
    "greetings", "welcome"
]

def sanitize_title(title_str: str, full_response: str = "") -> str:
    print(f"[sanitize] input: {title_str[:50]}")
    result = title_str
    lower = title_str.lower()
    for prefix in FILLER_PREFIXES:
        if lower.startswith(prefix):
            stripped = title_str[len(prefix):].lstrip(' ,.')
            stripped_words = stripped.split()
            # Fallback if result is empty or single short word
            if (not stripped or 
                len(stripped) < 5 or 
                len(stripped_words) < 2):
                if full_response:
                    words = full_response.split()
                    result = " ".join(words[:7]) + (
                        "..." if len(words) > 7 else ""
                    )
                else:
                    result = title_str
            else:
                result = stripped[:1].upper() + stripped[1:]
            break
    print(f"[sanitize] output: {result[:50]}")
    return result

def generate_title(user_query: str, ai_response: str) -> str:
    response_lower = ai_response.strip().lower()
    is_greeting = any(
        response_lower.startswith(p) 
        for p in GREETING_PATTERNS
    )
    if is_greeting:
        # Use user query as title
        words = user_query.strip().split()
        return " ".join(words[:7]) + ("..." if len(words) > 7 else "")
    
    # Otherwise use first 7 words of AI response
    clean_gen = re.sub(r'[*#`_\-\[\]\(\)]', '', ai_response)
    clean_gen = re.sub(r'\s+', ' ', clean_gen).strip()
    words = clean_gen.split()
    if words:
        title = " ".join(words[:7]) + ("..." if len(words) > 7 else "")
    else:
        title = user_query.strip()[:50]
    title = sanitize_title(title, ai_response)

    title = title.rstrip('.,;:!')
    if title:
        # Title Case but preserve the ellipsis if it exists
        title = " ".join(w.capitalize() for w in title.split())
    return title


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — invokes the full CRAG graph and returns
    the generated answer, source citations, and pipeline metadata.
    """
    try:
        try:
            result = _graph.invoke(_default_state(request.question))
        except Exception as e:
            import uuid
            result = _default_state(request.question)
            result["interaction_id"] = str(uuid.uuid4())
            if "429" in str(e) or "ResourceExhausted" in str(e):
                result["generation"] = "⚠️ Nexus System Status: Gemini API Daily Free-Tier Quota Limit fully exhausted. The system will resume operational status at Midnight Pacific Time."
            else:
                result["generation"] = f"An unexpected processing error occurred: {str(e)}"

        interaction_id = result.get("interaction_id", "")
        generation = result.get("generation", "No answer generated.")
        documents = result.get("documents", [])

        # Server-side persistence filtering & Title Generation
        blocked_words = {
            'hi', 'hello', 'hey', 'test', 'lol',
            'ok', 'rag', 'jesus', 'moses', 'joshua'
        }

        # Auto-generate chat title using the helper function
        title = generate_title(request.question, generation)

        q_cleaned = request.question.strip().lower().rstrip('?.!')

        is_valid_chat = (
            len(request.question.strip()) >= 4
            and len(request.question.split()) > 1
            and q_cleaned not in blocked_words
        )

        title_valid = (
            title
            and len(title) >= 4
            and title.lower().rstrip('?.!') not in blocked_words
        )

        is_persist_eligible = is_valid_chat and title_valid

        # Persist for RLHF if valid and eligible
        if is_persist_eligible:
            _feedback_store.save_interaction(
                interaction_id=interaction_id,
                question=request.question,
                generation=generation,
                documents=documents,
                rewrite_count=result.get("rewrite_count", 0),
                relevance_score=result.get("relevance_score", 0.0),
                title=title,
            )

        # Format sources
        sources = []
        for doc in documents:
            sources.append({
                "content": (
                    doc.page_content[:300] + "…"
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "source": doc.metadata.get("source_file", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "rrf_score": doc.metadata.get("rrf_score", 0),
                "rrf_sources": doc.metadata.get("rrf_sources", []),
                "rlhf_boost": doc.metadata.get("rlhf_boost", None),
            })

        metadata = {
            "rewrite_count": result.get("rewrite_count", 0),
            "relevance_score": round(result.get("relevance_score", 0.0), 3),
            "query_rewrites": result.get("query_rewrites", []),
            "hallucination_retries": result.get("hallucination_retries", 0),
        }

        debug_info = None
        if DEBUG:
            debug_info = {
                "route": result.get("route_decision", ""),
                "grounded": result.get("is_grounded", True)
            }

        return ChatResponse(
            interaction_id=interaction_id,
            answer=generation,
            sources=sources,
            metadata=metadata,
            debug=debug_info
        )
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
            import uuid
            return ChatResponse(
                interaction_id=str(uuid.uuid4()),
                answer="⚠️ Nexus System Status: Gemini API Daily Free-Tier Quota Limit fully exhausted. The system will resume operational status at Midnight Pacific Time.",
                sources=[],
                metadata={
                    "rewrite_count": 0,
                    "relevance_score": 0.0,
                    "query_rewrites": [],
                    "hallucination_retries": 0,
                }
            )
        if "api_key" in error_msg.lower() or "credentials" in error_msg.lower() or "403" in error_msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Google Gemini API key issue.",
                    "hint": "Please ensure GOOGLE_API_KEY is correctly set in your .env file.",
                    "detail": error_msg,
                },
            )
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — uses LangGraph astream_events(v2) to
    capture real LLM token generation events and stream them as SSE.

    Includes a lightweight semantic router: if the cached embedding
    similarity score clears a strict confidence threshold (> 0.85),
    the heavy query-rewriter LLM node is bypassed and a fast-route
    local response is streamed instead.

    Event types:
      • token    — a single generated text chunk
      • sources  — JSON array of source citations
      • metadata — pipeline metadata (rewrites, relevance, etc.)
      • done     — signals end of stream
      • error    — error message
    """
    import json as _json
    import uuid
    import asyncio

    async def event_generator():
        try:
            question = request.question
            interaction_id = str(uuid.uuid4())

            # ── Phase 0: Lightweight semantic routing bypass ──────
            # Check if the query embedding has high-confidence similarity
            # to known cached vectors. If score > 0.85, skip the heavy
            # rewriter node and stream a fast-route RAG answer directly.
            fast_route_result = None
            try:
                from dense import cached_embed_query, get_vectorstore
                normalized_q = question.strip().lower()
                embedding_vector = list(cached_embed_query(normalized_q))
                store = get_vectorstore()
                collection = store._collection
                if collection.count() > 0:
                    # Use similarity_search_with_relevance_scores for threshold check
                    scored_results = store.similarity_search_by_vector_with_relevance_scores(
                        embedding_vector, k=3
                    )
                    if scored_results and scored_results[0][1] > 0.85:
                        print(f"  -> [stream] Semantic fast-route: top score {scored_results[0][1]:.4f} > 0.85 threshold")
                        fast_route_docs = [doc for doc, score in scored_results]
                        for i, doc in enumerate(fast_route_docs):
                            doc.metadata["retrieval_source"] = "dense_fast_route"
                            doc.metadata["dense_rank"] = i

                        # Build a fast-route RAG context and generate directly
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        from langchain_core.prompts import ChatPromptTemplate
                        from langchain_core.output_parsers import StrOutputParser
                        from config import GEMINI_MODEL

                        rag_context = "\n\n".join([d.page_content for d in fast_route_docs])
                        prompt = ChatPromptTemplate.from_template(
                            "You are a precise assistant grounded strictly in retrieved context. "
                            "Never invent facts not present in the context. "
                            "Answer the user query using ONLY the provided data.\n\n"
                            "DATA:\n{rag_context}\n\n"
                            "USER QUESTION: {question}\n\n"
                            "RESPONSE:"
                        )
                        llm = ChatGoogleGenerativeAI(
                            model=GEMINI_MODEL,
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.2,
                            max_retries=1,
                            timeout=30.0,
                            streaming=True,
                        )
                        chain = prompt | llm | StrOutputParser()

                        # Stream tokens from the fast-route chain
                        full_gen = ""
                        async for chunk in chain.astream({"rag_context": rag_context, "question": question}):
                            if chunk:
                                yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"
                                full_gen += chunk

                        fast_route_result = {
                            "generation": full_gen,
                            "documents": fast_route_docs,
                            "interaction_id": interaction_id,
                            "rewrite_count": 0,
                            "relevance_score": 1.0,
                            "query_rewrites": [],
                            "hallucination_retries": 0,
                            "route_decision": "fast_route",
                            "is_grounded": True,
                        }
            except Exception as fast_err:
                print(f"  -> [stream] Fast-route check skipped: {fast_err}")
                fast_route_result = None

            # ── Phase 1: Full pipeline with astream_events (if no fast route) ──
            if fast_route_result is None:
                try:
                    # Use astream_events v2 to capture real LLM token events
                    full_gen = ""
                    result_state = {}

                    async for event in _graph.astream_events(
                        _default_state(question),
                        version="v2"
                    ):
                        kind = event.get("event", "")

                        # Capture real-time LLM token generation events
                        if kind == "on_chat_model_stream":
                            node = event.get("metadata", {}).get("langgraph_node")
                            if node in ("generate", "generate_direct"):
                                chunk = event.get("data", {}).get("chunk", None)
                                if chunk and hasattr(chunk, "content") and chunk.content:
                                    token_text = chunk.content
                                    if isinstance(token_text, list):
                                        parts = []
                                        for part in token_text:
                                            if isinstance(part, dict) and "text" in part:
                                                parts.append(part["text"])
                                            elif isinstance(part, str):
                                                parts.append(part)
                                        token_text = "".join(parts)
                                    elif not isinstance(token_text, str):
                                        token_text = str(token_text)
                                    
                                    if token_text:
                                        yield f"data: {_json.dumps({'type': 'token', 'content': token_text})}\n\n"
                                        full_gen += token_text

                        # Capture final graph state from chain_end events
                        elif kind == "on_chain_end":
                            output = event.get("data", {}).get("output", {})
                            if isinstance(output, dict) and "generation" in output:
                                result_state = output

                    # If astream_events didn't capture tokens (e.g. direct route),
                    # fall back to streaming the generation text from result_state
                    if not full_gen and result_state.get("generation"):
                        generation = result_state["generation"]
                        words = generation.split(" ")
                        buffer = ""
                        chunk_size = 3
                        for i, word in enumerate(words):
                            buffer += (" " if buffer else "") + word
                            if (i + 1) % chunk_size == 0 or i == len(words) - 1:
                                yield f"data: {_json.dumps({'type': 'token', 'content': buffer})}\n\n"
                                buffer = ""
                        full_gen = generation

                    # Build final result dict
                    fast_route_result = {
                        "generation": full_gen or result_state.get("generation", "No answer generated."),
                        "documents": result_state.get("documents", []),
                        "interaction_id": result_state.get("interaction_id", interaction_id),
                        "rewrite_count": result_state.get("rewrite_count", 0),
                        "relevance_score": result_state.get("relevance_score", 0.0),
                        "query_rewrites": result_state.get("query_rewrites", []),
                        "hallucination_retries": result_state.get("hallucination_retries", 0),
                        "route_decision": result_state.get("route_decision", ""),
                        "is_grounded": result_state.get("is_grounded", True),
                    }

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    err_msg = str(e)
                    if "429" in err_msg or "ResourceExhausted" in err_msg:
                        yield f"data: {_json.dumps({'type': 'token', 'content': '⚠️ Nexus System Status: Gemini API Daily Free-Tier Quota Limit fully exhausted. The system will resume operational status at Midnight Pacific Time.'})}\n\n"
                        yield f"data: {_json.dumps({'type': 'done'})}\n\n"
                        return
                    else:
                        yield f"data: {_json.dumps({'type': 'error', 'content': f'Processing error: {err_msg}'})}\n\n"
                        return

            # ── Phase 2: Send sources ────────────────────────────
            result = fast_route_result
            interaction_id = result.get("interaction_id", interaction_id)
            generation = result.get("generation", "")
            documents = result.get("documents", [])

            sources = []
            for doc in documents:
                sources.append({
                    "content": (
                        doc.page_content[:300] + "…"
                        if len(doc.page_content) > 300
                        else doc.page_content
                    ),
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "rrf_score": doc.metadata.get("rrf_score", 0),
                    "rrf_sources": doc.metadata.get("rrf_sources", []),
                    "rlhf_boost": doc.metadata.get("rlhf_boost", None),
                })
            yield f"data: {_json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # ── Phase 3: Send metadata ───────────────────────────
            metadata = {
                "interaction_id": interaction_id,
                "rewrite_count": result.get("rewrite_count", 0),
                "relevance_score": round(result.get("relevance_score", 0.0), 3),
                "query_rewrites": result.get("query_rewrites", []),
                "hallucination_retries": result.get("hallucination_retries", 0),
            }
            if DEBUG:
                metadata["debug"] = {
                    "route": result.get("route_decision", ""),
                    "grounded": result.get("is_grounded", True),
                }
            yield f"data: {_json.dumps({'type': 'metadata', 'metadata': metadata})}\n\n"

            # ── Phase 4: Persist for RLHF ────────────────────────
            title = generate_title(request.question, generation)

            blocked_words = {
                'hi', 'hello', 'hey', 'test', 'lol',
                'ok', 'rag', 'jesus', 'moses', 'joshua'
            }
            q_cleaned = request.question.strip().lower().rstrip('?.!')
            is_valid_chat = (
                len(request.question.strip()) >= 4
                and len(request.question.split()) > 1
                and q_cleaned not in blocked_words
            )
            title_valid = (
                title
                and len(title) >= 4
                and title.lower().rstrip('?.!') not in blocked_words
            )
            if is_valid_chat and title_valid:
                _feedback_store.save_interaction(
                    interaction_id=interaction_id,
                    question=request.question,
                    generation=generation,
                    documents=documents,
                    rewrite_count=result.get("rewrite_count", 0),
                    relevance_score=result.get("relevance_score", 0.0),
                    title=title,
                )

            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    from starlette.responses import StreamingResponse
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Accept a like/dislike signal for a past interaction.
    Triggers RLHF weight adaptation.
    """
    try:
        rlhf = get_rlhf_manager()
        result = rlhf.process_feedback(
            interaction_id=request.interaction_id,
            signal=request.signal.value,
            comment=request.comment,
        )
        return FeedbackResponse(
            success=True,
            feedback_id=result["feedback_id"],
            weights_updated=result["weights_updated"],
            current_weights=result["current_weights"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
async def feedback_stats():
    """Return RLHF dashboard metrics."""
    return get_rlhf_manager().get_stats()


@app.get("/feedback/history")
async def feedback_history(limit: int = 20):
    """Return recent interaction history."""
    return _feedback_store.get_recent_interactions(limit=limit)


@app.post("/feedback/clear")
async def clear_feedback_history():
    """Clear all interaction, feedback, and reward history."""
    try:
        _feedback_store.clear_all_data()
        return {"success": True, "message": "History cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/feedback/interaction/{interaction_id}")
async def delete_interaction(interaction_id: str):
    """Delete a single interaction and its feedback."""
    try:
        _feedback_store.delete_interaction(interaction_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    """Upload files and run the dual-index ingestion pipeline."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        saved = []
        for f in files:
            dest = os.path.join(DATA_DIR, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(f.filename)

        from ingest import ingest_documents
        count = ingest_documents()
        from dense import reset_vectorstore
        reset_vectorstore()

        return {
            "success": True,
            "files": saved,
            "chunks_created": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        graph_nodes=[
            "router", "hybrid_retrieve", "fusion_rrf",
            "grade_documents", "rewrite_query", "web_scrape",
            "generate", "generate_direct", "hallucination_check",
        ],
    )


@app.get("/health/gemini", response_model=GeminiHealthResponse)
async def health_gemini():
    """Check whether Google Gemini API is reachable and has credentials configured."""
    try:
        if not GOOGLE_API_KEY:
            return GeminiHealthResponse(
                status="key_missing",
                model=GEMINI_MODEL,
                available=False,
                error="GOOGLE_API_KEY environment variable is not configured."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            max_retries=0
        )
        await llm.ainvoke("ping")
        return GeminiHealthResponse(
            status="online",
            model=GEMINI_MODEL,
            available=True,
            error=None
        )
    except Exception as e:
        return GeminiHealthResponse(
            status="error",
            model=GEMINI_MODEL,
            available=False,
            error=str(e)
        )


@app.get("/documents")
async def list_documents():
    """List documents in the knowledge base directory."""
    docs = []
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                docs.append({
                    "name": fname,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
    return {"documents": docs, "count": len(docs)}


@app.get("/settings", response_model=SettingsModel)
async def get_settings():
    """Retrieve runtime settings from SQLite."""
    try:
        settings = _feedback_store.get_settings()
        return SettingsModel(
            dense_top_k=int(settings.get("dense_top_k", 10)),
            sparse_top_k=int(settings.get("sparse_top_k", 10)),
            fusion_top_k=int(settings.get("fusion_top_k", 6)),
            rrf_k_constant=int(settings.get("rrf_k_constant", 60)),
            max_rewrite_attempts=int(settings.get("max_rewrite_attempts", 3)),
            max_hallucination_retries=int(settings.get("max_hallucination_retries", 2)),
            dense_weight=float(settings.get("dense_weight", 0.6)),
            sparse_weight=float(settings.get("sparse_weight", 0.4)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/settings")
async def save_settings(settings: SettingsModel):
    """Save new runtime settings into SQLite."""
    try:
        _feedback_store.save_settings(settings.dict())
        # Re-initialize the RLHFManager global instance to pick up new weights
        from rlhf import get_rlhf_manager
        mgr = get_rlhf_manager()
        mgr._dense_w = float(settings.dense_weight)
        mgr._sparse_w = float(settings.sparse_weight)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static files (CSS / JS) ──────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")
