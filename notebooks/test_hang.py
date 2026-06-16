import sys
import os
import time

print("Adding backend to sys.path...")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

print("Importing langchain_google_genai...")
t0 = time.time()
from langchain_google_genai import ChatGoogleGenerativeAI
print(f"Imported langchain_google_genai in {time.time() - t0:.2f}s")

print("Importing graph.state...")
t0 = time.time()
from graph.state import GraphState
print(f"Imported graph.state in {time.time() - t0:.2f}s")

print("Importing services.scraper...")
t0 = time.time()
from services.scraper import nexus_scraper
print(f"Imported services.scraper in {time.time() - t0:.2f}s")

print("Importing config...")
t0 = time.time()
from config import GOOGLE_API_KEY, GEMINI_MODEL
print(f"Imported config in {time.time() - t0:.2f}s")

print("Importing graph.nodes_agent...")
t0 = time.time()
import graph.nodes_agent as nodes_agent
print(f"Imported graph.nodes_agent in {time.time() - t0:.2f}s")

print("Importing graph.nodes_retrieval...")
t0 = time.time()
import graph.nodes_retrieval as nodes_retrieval
print(f"Imported graph.nodes_retrieval in {time.time() - t0:.2f}s")

print("Importing graph.nodes_fusion...")
t0 = time.time()
import graph.nodes_fusion as nodes_fusion
print(f"Imported graph.nodes_fusion in {time.time() - t0:.2f}s")

print("Importing graph.builder...")
t0 = time.time()
from graph.builder import build_graph
print(f"Imported graph.builder in {time.time() - t0:.2f}s")

print("Building graph...")
t0 = time.time()
graph = build_graph()
print(f"Built graph in {time.time() - t0:.2f}s")

print("All tests completed successfully!")
