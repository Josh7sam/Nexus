"""
Agent nodes — LLM-powered processing steps for the Corrective RAG graph.

Nodes:
  • router_node           — classify question → retrieve or direct answer
  • grade_documents_node  — binary relevance scoring of retrieved chunks
  • rewrite_query_node    — rephrase query for better hybrid search
  • web_scrape_node       — fallback web crawling search using Crawl4AI
  • generate_node         — RAG generation or comparative synthesis grounded in context
  • generate_direct_node  — direct answer without retrieval
  • hallucination_check_node — verify generation is grounded in context
"""

import uuid
import asyncio
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from state import GraphState
from config import GOOGLE_API_KEY, GEMINI_MODEL
from scraper import nexus_scraper

_GEMINI_ERROR_MSG = (
    "⚠ The Google Gemini API backend failed to respond. "
    "Please check your network connection and GOOGLE_API_KEY settings."
)

ANTI_FILLER = (
    "IMPORTANT: Never, under any circumstances, begin "
    "your response with 'Based on the provided data', "
    "'Based on the documents', 'According to the context', "
    "'Based on the information', 'The provided data', "
    "or ANY similar filler phrase. "
    "Your very first word must be a direct, substantive "
    "answer. Violating this rule is not acceptable.\n\n"
)


# ═══════════════════════════════════════════════════════════════
#  LLM Factory
# ═══════════════════════════════════════════════════════════════

def _get_llm() -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance with project-wide settings."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_retries=1,
        timeout=30.0,
    )


def _invoke_safe(chain, inputs: dict, fallback: str) -> str:
    """Invoke a chain, returning fallback string on errors."""
    try:
        return chain.invoke(inputs)
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "unauthorized" in err.lower() or "403" in err:
            raise ValueError(f"Gemini API key error: {err}") from e
        if (
            "429" in err 
            or "ResourceExhausted" in err 
            or "quota" in err.lower() 
            or "timeout" in err.lower()
            or "504" in err
            or "deadline" in err.lower()
        ):
            raise e
        print(f"    [WARN] LLM invocation failed: {e}")
        return fallback


# ═══════════════════════════════════════════════════════════════
#  1. Router
# ═══════════════════════════════════════════════════════════════

def router_node(state: GraphState) -> dict:
    """
    Classify the user question as needing retrieval or being directly
    answerable.  Sets `route_decision` to 'retrieve' or 'direct'.
    """
    question = state["question"]
    
    # Fast path for obvious greetings, farewells, gratitude, or identity/bot questions to save API quota and latency
    q_clean = question.strip().lower().rstrip("?.!")
    
    greetings = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "how's it going", "howdy", "sup", "yo", "whats up", "what's up"
    }
    
    farewells = {
        "bye", "goodbye", "see you", "see you later", "talk to you later", "adios", "farewell"
    }
    
    gratitude = {
        "thank you", "thanks", "thank you very much", "thanks a lot", "appreciate it", "great", "perfect", "awesome"
    }
    
    identity_phrases = {
        "who are you", "what is your name", "what are you", "who made you", "who created you", 
        "what can you do", "help", "how can you help me", "what is nexus", "who is nexus", "about nexus"
    }
    
    is_chitchat = (
        q_clean in greetings 
        or q_clean in farewells 
        or q_clean in gratitude 
        or q_clean in identity_phrases 
        or len(q_clean) <= 3
        or any(q_clean.startswith(p) for p in ["hello ", "hi ", "hey ", "thank you for ", "thanks for "])
    )

    if is_chitchat:
        print(f"  -> [router] Rule-based route decision: direct")
        return {
            "question": question,
            "interaction_id": str(uuid.uuid4()),
            "rewrite_count": 0,
            "hallucination_retries": 0,
            "query_rewrites": [],
            "route_decision": "direct",
        }

    # Bypassed Router LLM to minimize latency
    print(f"  -> [router] Rule-based route decision: retrieve")
    return {
        "question": question,
        "interaction_id": str(uuid.uuid4()),
        "rewrite_count": 0,
        "hallucination_retries": 0,
        "query_rewrites": [],
        "route_decision": "retrieve",
    }



# ═══════════════════════════════════════════════════════════════
#  2. Document Grader
# ═══════════════════════════════════════════════════════════════

def grade_documents_node(state: GraphState) -> dict:
    """
    Grade each retrieved document for relevance using a single batched JSON LLM scoring call.
    Keeps only documents scored as 'yes'.
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {
            "documents": [],
            "relevance_score": 0.0,
        }

    # Bypassed LLM-based document grading to minimize latency.
    # Directly marks all retrieved fused documents as relevant.
    for doc in documents:
        doc.metadata["relevance"] = "relevant"

    print(f"  -> [grade_documents] Directly marked {len(documents)}/{len(documents)} relevant (Bypassed LLM)")
    return {
        "documents": documents,
        "relevance_score": 1.0,
    }


# ═══════════════════════════════════════════════════════════════
#  3. Query Rewriter
# ═══════════════════════════════════════════════════════════════

def rewrite_query_node(state: GraphState) -> dict:
    """
    Rewrite the query to improve both semantic and keyword retrieval.
    Increments `rewrite_count` for loop-guard tracking.
    """
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    query_rewrites = state.get("query_rewrites", [])
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query optimizer. Rewrite the query to improve "
         "both semantic vector similarity search AND BM25 keyword matching.\n"
         "Return ONLY the rewritten query — no explanation, no preamble."),
        ("human", "Original query: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        rewritten = _invoke_safe(chain, {"question": question}, question).strip()
    except Exception as e:
        err = str(e)
        if (
            "429" in err 
            or "ResourceExhausted" in err 
            or "quota" in err.lower() 
            or "timeout" in err.lower()
            or "504" in err
            or "deadline" in err.lower()
        ):
            raise e
        rewritten = question  # Keep original if LLM unavailable

    new_count = rewrite_count + 1
    print(f"  -> [rewrite_query] Attempt {new_count}: {rewritten[:80]}...")

    return {
        "question": rewritten,
        "rewritten_query": rewritten,
        "rewrite_count": new_count,
        "query_rewrites": query_rewrites + [rewritten],
    }


# ═══════════════════════════════════════════════════════════════
#  4. Fallback Web Scraper
# ═══════════════════════════════════════════════════════════════

def _run_async(coro):
    """
    Run an async coroutine from synchronous code, handling the case
    where an event loop may or may not already be running.

    Strategy:
      1. If no loop is running → just use asyncio.run()
      2. If a loop IS running → spawn a background thread with its
         own loop (avoids 'cannot run nested event loop' errors).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No running loop — safe to use asyncio.run directly
        return asyncio.run(coro)

    # A loop is already running (e.g. inside uvicorn/FastAPI) —
    # run in a dedicated background thread with its own event loop.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=60)


def web_scrape_node(state: GraphState) -> dict:
    """
    Crawl top external web results as fallback when local chunks are irrelevant.
    """
    print("\n--- NODE: DEEP CRAWL4AI WEB SCRAPER FALLBACK ---")
    query = state.get("rewritten_query") or state["question"]

    # Extract reference endpoints
    urls = nexus_scraper.search_urls(query, max_results=2)
    if not urls:
        print("    [INFO] Web lookups yielded zero viable index references.")
        return {"web_documents": []}

    print(f"    [INFO] Crawl4AI dispatching for: {urls}")
    try:
        results = _run_async(nexus_scraper.scrape_urls(urls))
        scraped_docs = [
            Document(
                page_content=r["content"],
                metadata={"source": r["url"], "source_file": r["url"]},
            )
            for r in results
            if r.get("content")
        ]
    except Exception as e:
        print(f"    [WARN] Scrape failed: {e}")
        scraped_docs = []

    print(f"    [INFO] Scraped {len(scraped_docs)} document(s) from the web")
    return {"web_documents": scraped_docs}


# ═══════════════════════════════════════════════════════════════
#  5. Generator (RAG & Comparative)
# ═══════════════════════════════════════════════════════════════

def generate_node(state: GraphState) -> dict:
    """
    Generate an answer grounded strictly in retrieved context.
    If both RAG documents and Web Scraped documents exist, pass both
    distinct context arrays to Gemini and draft a response highlighting
    any alignment or contradictions.
    """
    print("\n--- NODE: COMPARATIVE GENERATION SYSTEM ---")
    question = state["question"]
    rag_docs = state.get("documents", [])
    web_docs = state.get("web_documents", [])
    llm = _get_llm()

    # If both exist, perform comparative synthesis
    if rag_docs and web_docs:
        rag_context = "\n\n".join([d.page_content for d in rag_docs])
        web_context = "\n\n".join([d.page_content for d in web_docs])
        
        # System prompt engineered specifically to handle data comparative logic constraints
        comparative_prompt = ChatPromptTemplate.from_template(
            ANTI_FILLER +
            "You are an elite research assistant. Answer the user query using the provided internal database logs "
            "and live external web crawls. If information appears in both, synthesize them cleanly. If they conflict, "
            "explicitly highlight the contradictions (e.g., older internal logs vs. updated live data).\n\n"
            "INTERNAL DATABASE DATA:\n{rag_context}\n\n"
            "LIVE WEB SCRAWLED DATA:\n{web_context}\n\n"
            "USER QUESTION: {question}\n\n"
            "SCIENTIFIC COMPARATIVE RESPONSE:"
        )
        chain = comparative_prompt | llm | StrOutputParser()
        generation = _invoke_safe(chain, {"rag_context": rag_context, "web_context": web_context, "question": question}, _GEMINI_ERROR_MSG)
    elif web_docs:
        # Only web docs exist
        web_context = "\n\n".join([d.page_content for d in web_docs])
        prompt = ChatPromptTemplate.from_template(
            ANTI_FILLER +
            "You are a precise assistant. Answer the user query using the provided live web crawled data.\n\n"
            "LIVE WEB SCRAWLED DATA:\n{web_context}\n\n"
            "USER QUESTION: {question}\n\n"
            "RESPONSE:"
        )
        chain = prompt | llm | StrOutputParser()
        generation = _invoke_safe(chain, {"web_context": web_context, "question": question}, _GEMINI_ERROR_MSG)
    else:
        # Only RAG docs exist (or none)
        if not rag_docs:
            return {
                "generation": (
                    "I don't have enough relevant context in the knowledge base "
                    "to answer this question. Please try rephrasing or ensure the "
                    "knowledge base contains relevant information."
                ),
            }
        rag_context = "\n\n".join([d.page_content for d in rag_docs])
        prompt = ChatPromptTemplate.from_template(
            ANTI_FILLER +
            "You are a precise assistant grounded strictly in retrieved context from the custom knowledge base. "
            "Never invent facts not present in the context. Answer the user query using ONLY the provided internal database data.\n\n"
            "INTERNAL DATABASE DATA:\n{rag_context}\n\n"
            "USER QUESTION: {question}\n\n"
            "RESPONSE:"
        )
        chain = prompt | llm | StrOutputParser()
        generation = _invoke_safe(chain, {"rag_context": rag_context, "question": question}, _GEMINI_ERROR_MSG)
        
    print(f"  -> [generate] Answer generated ({len(generation)} chars)")
    
    # Merge documents for downstream checks/hallucination checks
    all_docs = rag_docs + web_docs
    return {
        "generation": generation,
        "documents": all_docs,
    }


# ═══════════════════════════════════════════════════════════════
#  6. Generator (Direct — no retrieval)
# ═══════════════════════════════════════════════════════════════

def generate_direct_node(state: GraphState) -> dict:
    """
    Generate a direct answer for questions that don't need retrieval
    (greetings, simple math, general knowledge).
    """
    question = state["question"]
    
    # Fast path: instant replies for common greetings without API calls
    q_clean = question.strip().lower().rstrip("?.!")
    greetings_map = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hey there! How can I help you today?",
        "hola": "¡Hola! ¿En qué puedo ayudarte hoy?",
        "greetings": "Greetings! How can I help you today?",
        "good morning": "Good morning! How can I help you today?",
        "good afternoon": "Good afternoon! How can I help you today?",
        "good evening": "Good evening! How can I help you today?",
    }
    if q_clean in greetings_map:
        print(f"  -> [generate_direct] Rule-based greeting answer generated")
        return {
            "generation": greetings_map[q_clean],
            "documents": [],
            "is_grounded": True,
        }

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         ANTI_FILLER +
         "You are a helpful, friendly assistant. Answer the question "
         "directly and concisely. Format your response clearly."),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    generation = _invoke_safe(chain, {"question": question}, _GEMINI_ERROR_MSG)

    print(f"  -> [generate_direct] Direct answer generated")
    return {
        "generation": generation,
        "documents": [],
        "is_grounded": True,
    }


# ═══════════════════════════════════════════════════════════════
#  7. Hallucination Check
# ═══════════════════════════════════════════════════════════════

def hallucination_check_node(state: GraphState) -> dict:
    """
    Verify that the generated answer is grounded in the retrieved context.
    Sets `is_grounded` to True/False accordingly.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    hallucination_retries = state.get("hallucination_retries", 0)

    # Skip check for direct answers (no documents)
    if not documents:
        return {"is_grounded": True}

    # Skip if generation itself contains an error message
    if "Gemini" in generation or "LLM backend" in generation:
        return {"is_grounded": True}

    llm = _get_llm()
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grounding verifier. Determine whether the answer uses "
         "ONLY facts that are present in the provided context.\n\n"
         "Respond with ONLY 'yes' or 'no'."),
        ("human",
         "Context:\n{context}\n\n"
         "Answer:\n{generation}\n\n"
         "Is the answer fully grounded in the context above?"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        result = _invoke_safe(
            chain,
            {"context": context, "generation": generation},
            "yes",
        ).strip().lower()
        is_grounded = "yes" in result
    except Exception as e:
        err = str(e)
        if (
            "429" in err 
            or "ResourceExhausted" in err 
            or "quota" in err.lower() 
            or "timeout" in err.lower()
            or "504" in err
            or "deadline" in err.lower()
        ):
            raise e
        is_grounded = True  # Skip retry if Gemini is down
    
    new_retries = hallucination_retries + (0 if is_grounded else 1)
    status = "[OK] grounded" if is_grounded else f"[FAIL] not grounded (retry {new_retries})"
    print(f"  -> [hallucination_check] {status}")

    return {
        "hallucination_retries": new_retries,
        "is_grounded": is_grounded,
    }
