# NEXUS Backend Service

FastAPI server hosting the agentic Corrective RAG pipeline.

## Structure
- `server.py`: API controller and routes.
- `builder.py`, `edges.py`, `state.py`: LangGraph state machine.
- `nodes_agent.py`, `nodes_retrieval.py`, `nodes_fusion.py`: Agent execution steps.
- `dense.py`, `sparse.py`, `fusion.py`, `ingest.py`: Core hybrid retrieval mechanisms.
- `store.py`, `rlhf.py`: Interaction caching and weight feedback loops.

## Requirements
To install backend dependencies separately:
```bash
pip install -r requirements.txt
```
