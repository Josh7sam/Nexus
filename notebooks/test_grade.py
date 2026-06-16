import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from graph.nodes_agent import grade_documents_node
from langchain_core.documents import Document

state = {
    "question": "What is RAG?",
    "documents": [
        Document(page_content="Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM responses.", metadata={"source_file": "doc1.txt"}),
        Document(page_content="Chroma is a vector database.", metadata={"source_file": "doc2.txt"})
    ]
}
print("Calling grade_documents_node...")
res = grade_documents_node(state)
print("Result:", res)
