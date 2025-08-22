import os
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from pydantic import BaseModel

from RAGDataMite.RAG.LLM.rag_controller import rag_with_validation
from RAGDataMite.RAG.Retrieval.retriever import setup_retriever

app = FastAPI(title="Datamite RAG API", version="0.1.0")

class AskRequest(BaseModel):
    question: str
    k: int = 3
    min_similarity: float = 0.48

class SourceItem(BaseModel):
    name: str | None = None
    score: float | None = None
    url: str | None = None
    snippet: str | None = None
    metadata: Dict[str, Any] | None = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    meta: Dict[str, Any] = {}

def _to_source_item(s: Any) -> Dict[str, Any]:
    """
    Normalize a source into a simple dict for the UI.

    Supports:
      - LangChain Document
      - (Document, score) tuples
      - dicts with possible nested document
    """

    if isinstance(s, Document):
        name = s.metadata.get("name") or s.metadata.get("id") or s.metadata.get("type")
        return {
            "name": name,
            "url": s.metadata.get("url"),
            "snippet": (s.page_content or "")[:300],
            "metadata": s.metadata
        }

    return {
        "name": None,
        "url": None,
        "snippet": str(s)[:300],
        "metadata": {}
    }

# ------------ App state ------------
retriever = None

@app.on_event("startup")
def _startup():
    """
    Load the retriever ONCE when the server boots.
    This avoids reloading the vector store on every request.
    """
    global retriever

    persist_dir = os.getenv("PERSIST_DIR", "RAG/ProcessedDocuments/chroma_db")
    k_default = 3

    try:
        retriever = setup_retriever(persist_directory=persist_dir, k=k_default)
    except Exception as e:
        print(f"[startup] Failed to setup retriever from {persist_dir}: {e}")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[startup] WARNING: ANTHROPIC_API_KEY is not set. /ask will fail when calling the LLM.")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server.")

    try:
        result = rag_with_validation(req.question, min_similarity=req.min_similarity, retriever=retriever)

        raw_sources = result.get("sources", []) or []

        # Converting to a UI friendly schema
        sources = [_to_source_item(s) for s in raw_sources]

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "meta": {"k": req.k, "min_similarity": req.min_similarity}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")