"""HTTP API: POST /ask returns a grounded answer plus the source passages it was built from.

  uvicorn src.api:app            (env: RAG_METHOD, RAG_CHUNK_SIZE, RAG_MODEL, RAG_CORPUS, RAG_K)

`create_app` takes the retriever and generator as arguments so tests can inject lightweight stand-ins.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chunking import Chunk
from src.rag import answer_question


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    n: int
    doc: str
    title: str
    heading: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def create_app(retriever, chunks: list[Chunk], generator) -> FastAPI:
    app = FastAPI(title="Ask the Docs", description="RAG over the Azure Container Apps documentation")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            result = answer_question(req.question, retriever, chunks, generator, k=req.k)
        except Exception as exc:  # surface model/retrieval failures as a clean 502, not a stack trace
            raise HTTPException(status_code=502, detail=f"generation failed: {type(exc).__name__}") from exc
        return AskResponse(answer=result["answer"], sources=[
            Source(n=s["n"], doc=s["doc"], title=s["title"], heading=s["heading"], score=s["score"],
                   snippet=s["text"][:300]) for s in result["sources"]])

    return app


def _default_app() -> FastAPI:
    from src.chunking import load_corpus
    from src.rag import LocalGenerator
    from src.retrievers import build_retrievers
    method = os.getenv("RAG_METHOD", "dense")
    chunks = load_corpus(os.getenv("RAG_CORPUS", "data/corpus"), int(os.getenv("RAG_CHUNK_SIZE", "256")))
    retriever = build_retrievers(chunks, [method])[method]
    return create_app(retriever, chunks, LocalGenerator(os.getenv("RAG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")))


def __getattr__(name):  # lazy: `uvicorn src.api:app` builds the heavy pipeline only when it is requested
    if name == "app":
        return _default_app()
    raise AttributeError(name)
