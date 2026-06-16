import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GOOGLE_API_KEY, GEMINI_MODEL

print("GOOGLE_API_KEY configured:", bool(GOOGLE_API_KEY))
print("GEMINI_MODEL:", GEMINI_MODEL)

question = "What is RAG?"
formatted_docs = "--- Document 0 ---\nRetrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM responses."

print("\n1. Constructing prompt...")
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document relevance grader. Evaluate a list of retrieved documents "
     "and decide if each document contains information relevant to answering the user's question.\n\n"
     "For each document, determine if it is relevant ('yes') or irrelevant ('no').\n"
     "You MUST respond ONLY with a valid JSON object where keys are the document index integers as strings "
     "(e.g., \"0\", \"1\", \"2\") and values are \"yes\" or \"no\".\n"
     "Do not include any explanation, preamble, or markdown formatting blocks (like ```json).\n"
     "Example response format:\n"
     "{{\n  \"0\": \"yes\",\n  \"1\": \"no\",\n  \"2\": \"yes\"\n}}"),
    ("human",
     "Question: {question}\n\n"
     "Documents:\n{documents}"),
])
print("Prompt constructed.")

print("\n2. Initializing ChatGoogleGenerativeAI...")
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
    max_retries=1
)
print("LLM initialized.")

print("\n3. Building chain...")
chain = prompt | llm | StrOutputParser()
print("Chain built.")

print("\n4. Invoking chain (prompt + LLM)...")
try:
    res = chain.invoke({"question": question, "documents": formatted_docs})
    print("\n5. Chain invoked successfully! Response:")
    print(res)
except Exception as e:
    print("\nChain invocation failed with exception:")
    print(e)
