"""
VietMoney RAG — Main CLI Entry Point
Unified entry point: chat (default) or ingest.

Usage:
    python main.py              → Start interactive chatbot
    python main.py chat         → Start interactive chatbot
    python main.py ingest       → Run data ingestion pipeline
"""

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from src.vector_store_manager import VectorStoreManager
from src.chat_engine import ChatEngine
from src.exchange_rate_service import ExchangeRateService

# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CACHE_PATH = BASE_DIR / "data" / "processed" / "cache.json"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Chat mode                                                         #
# ------------------------------------------------------------------ #
def run_chat() -> None:
    """Start the interactive chatbot session."""
    logger.info("=" * 60)
    logger.info("VietMoney Assistant — Multilingual Financial Chatbot")
    logger.info("=" * 60)

    # --- Initialize embedding model ---
    logger.info("Loading embedding model BAAI/bge-m3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Initialize vector store ---
    vector_store = VectorStoreManager(persist_dir=CHROMA_DIR, embeddings=embeddings)
    total = vector_store.get_total_count()
    if total == 0:
        logger.warning(
            "Vector database is empty! Run 'python main.py ingest' first "
            "to ingest documents."
        )

    # --- Initialize LLM ---
    logger.info("Initializing Groq LLM (llama-3.1-8b-instant)...")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        streaming=False,
    )

    # --- Initialize Exchange Rate Service ---
    logger.info("Initializing Exchange Rate Service...")
    rate_service = ExchangeRateService()

    # --- Initialize ChatEngine ---
    engine = ChatEngine(
        vector_store=vector_store,
        llm=llm,
        embeddings=embeddings,
        cache_path=CACHE_PATH,
        exchange_rate_service=rate_service,
    )

    # --- Interactive loop ---
    # Ensure UTF‑8 output on Windows consoles
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "-" * 60)
    print("  VietMoney Assistant — Sẵn sàng trả lời!")
    print("  Gõ 'exit', 'quit', hoặc 'q' để thoát.")
    print("-" * 60 + "\n")

    while True:
        try:
            query = input("You / Bạn hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nTạm biệt! / Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\nTạm biệt! / Goodbye!")
            break

        # Get answer
        start = time.time()
        answer = engine.ask(query)
        elapsed_ms = (time.time() - start) * 1000

        print(f"\n🤖 VietMoney: {answer}")
        print(f"   ⏱ {elapsed_ms:.0f}ms\n")


# ------------------------------------------------------------------ #
#  Ingest mode                                                       #
# ------------------------------------------------------------------ #
def run_ingest() -> None:
    """Run the data ingestion pipeline."""
    from ingest_data import run_pipeline
    run_pipeline()


# ------------------------------------------------------------------ #
#  CLI dispatcher                                                    #
# ------------------------------------------------------------------ #
def main() -> None:
    """Parse CLI args and dispatch to the appropriate mode."""
    args = sys.argv[1:]
    command = args[0].lower() if args else "chat"

    if command == "chat":
        run_chat()
    elif command == "ingest":
        run_ingest()
    elif command == "serve":
        import uvicorn
        logger.info("Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    elif command in ("--help", "-h", "help"):
        print(
            "Usage: python main.py [command]\n\n"
            "Commands:\n"
            "  chat     Start interactive chatbot (default)\n"
            "  ingest   Run data ingestion pipeline\n"
            "  serve    Start the FastAPI streaming API server\n"
            "  help     Show this help message\n"
        )
    else:
        print(f"Unknown command: '{command}'. Use 'python main.py help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTạm biệt! / Goodbye!")
        sys.exit(0)
