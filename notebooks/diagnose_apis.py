import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL

def main():
    print(f"GOOGLE_API_KEY configured: {'Yes' if GOOGLE_API_KEY else 'No'}")
    print(f"GEMINI_MODEL: {GEMINI_MODEL}")
    print(f"GEMINI_EMBEDDING_MODEL: {GEMINI_EMBEDDING_MODEL}")
    
    # 1. Test Chat LLM
    print("\n--- 1. Testing Chat LLM (gemini-2.5-flash) ---")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
        t0 = time.time()
        res = llm.invoke("Say 'hello'")
        content = res.content
        if isinstance(content, list):
            content = " ".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
        print(f"Chat LLM responded in {time.time() - t0:.2f}s: {content.strip()}")
    except Exception as e:
        print(f"Chat LLM failed: {e}")
        
    # 2. Test Embeddings
    print("\n--- 2. Testing Embeddings (gemini-embedding-2) ---")
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        t0 = time.time()
        vector = embeddings.embed_query("What is Retrieval-Augmented Generation (RAG)?")
        print(f"Embeddings generated vector of size {len(vector)} in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Embeddings failed: {e}")
        
    # 3. Test ChromaDB Query
    print("\n--- 3. Testing ChromaDB ---")
    try:
        from vectorstore.dense import get_vectorstore
        t0 = time.time()
        store = get_vectorstore()
        count = store._collection.count()
        print(f"ChromaDB collection count: {count} (checked in {time.time() - t0:.2f}s)")
        
        if count > 0:
            t0 = time.time()
            res = store.similarity_search("What is RAG?", k=2)
            print(f"ChromaDB similarity search returned {len(res)} results in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"ChromaDB query failed: {e}")

if __name__ == "__main__":
    main()
