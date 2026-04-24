"""
VietMoney RAG — FastAPI Streaming Server
Exposes POST /chat/stream for SSE-based real-time chat responses.

Usage:
    python main.py serve          → Start the API server
    uvicorn api_server:app --reload  → Direct uvicorn launch
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from src.vector_store_manager import VectorStoreManager
from src.chat_engine import ChatEngine

# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CACHE_PATH = BASE_DIR / "data" / "processed" / "cache.json"

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Global engine reference (initialized at startup)                  #
# ------------------------------------------------------------------ #
engine: ChatEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once at startup, release on shutdown."""
    global engine

    logger.info("=" * 60)
    logger.info("VietMoney RAG — API Server starting...")
    logger.info("=" * 60)

    # Embedding model
    logger.info("Loading embedding model BAAI/bge-m3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Vector store
    vector_store = VectorStoreManager(persist_dir=CHROMA_DIR, embeddings=embeddings)
    total = vector_store.get_total_count()
    logger.info(f"Vector DB has {total} chunks")

    # LLM — streaming enabled
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        streaming=True,  # Enable streaming support
    )

    # Chat engine
    engine = ChatEngine(
        vector_store=vector_store,
        llm=llm,
        embeddings=embeddings,
        cache_path=CACHE_PATH,
    )

    logger.info("✅ API Server ready!")
    yield
    logger.info("API Server shutting down.")


# ------------------------------------------------------------------ #
#  FastAPI app                                                       #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="VietMoney RAG API",
    description="Streaming RAG chatbot for Vietnam Travel & Finance",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Request / Response models                                         #
# ------------------------------------------------------------------ #
class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    total_chunks: int


# ------------------------------------------------------------------ #
#  Endpoints                                                         #
# ------------------------------------------------------------------ #
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns DB stats."""
    total = engine.vector_store.get_total_count() if engine else 0
    return HealthResponse(status="ok", total_chunks=total)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming chat endpoint.

    Streams JSON events line-by-line:
      data: {"type":"token","content":"..."}
      data: {"type":"sources","content":[...]}
      data: {"type":"done"}
    """

    async def event_generator():
        try:
            async for event_type, content in engine.stream_ask(
                query=req.query,
                history=req.history,
            ):
                payload = json.dumps(
                    {"type": event_type, "content": content},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_payload = json.dumps(
                {"type": "error", "content": str(e)},
                ensure_ascii=False,
            )
            yield f"data: {error_payload}\n\n"
        finally:
            done_payload = json.dumps({"type": "done", "content": ""})
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)