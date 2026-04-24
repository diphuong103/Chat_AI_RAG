"""
VietMoney RAG — Ingest Data Pipeline
Entry point for the incremental ingestion pipeline.

Performs a full 7-step process:
  1. Scan all .pdf/.txt files in data/raw/{manual,wiki,web}
  2. Detect & clean up orphan files (removed from disk but still in registry)
  3. Compare MD5 hashes to classify files as NEW / MODIFIED / UNCHANGED
  4. For MODIFIED files: delete old vectors, then reload & split
  5. Embed chunks with BAAI/bge-m3 and upsert into ChromaDB
  6. Update hash_registry.json
  7. Print summary report

⚠ NOTE: If you change the embedding model (e.g. from bge-m3 to another),
  you MUST delete the entire data/chroma_db/ directory AND
  data/processed/hash_registry.json, then re-run this script.
"""

import json
import logging
import sys
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager

# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent
RAW_DIRS = [
    BASE_DIR / "data" / "raw" / "manual",
    BASE_DIR / "data" / "raw" / "wiki",
    BASE_DIR / "data" / "raw" / "web",
]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
HASH_REGISTRY_PATH = PROCESSED_DIR / "hash_registry.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".json"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Registry helpers                                                  #
# ------------------------------------------------------------------ #
def load_registry() -> dict:
    """Load hash_registry.json, creating it if it doesn't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if HASH_REGISTRY_PATH.exists():
        try:
            with open(HASH_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Corrupted registry, starting fresh: {e}")
            return {}
    return {}


def save_registry(registry: dict) -> None:
    """Persist hash_registry.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(HASH_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")


# ------------------------------------------------------------------ #
#  Step 1 — Scan source files                                       #
# ------------------------------------------------------------------ #
def scan_source_files() -> list[Path]:
    """Return all .pdf/.txt files across the three raw directories."""
    files: list[Path] = []
    for raw_dir in RAW_DIRS:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(raw_dir.glob(f"*{ext}"))
    return sorted(files)


# ------------------------------------------------------------------ #
#  Main pipeline                                                     #
# ------------------------------------------------------------------ #
def run_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("VietMoney — Smart Ingestion Pipeline")
    logger.info("=" * 60)

    # --- Initialize components ---
    logger.info("Loading embedding model BAAI/bge-m3 (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
    vector_store = VectorStoreManager(persist_dir=CHROMA_DIR, embeddings=embeddings)
    registry = load_registry()

    # Counters
    count_new = 0
    count_modified = 0
    count_unchanged = 0
    count_orphan = 0

    # ============================
    # Step 1 — Scan source files
    # ============================
    current_files = scan_source_files()
    current_paths_set = {str(f) for f in current_files}
    logger.info(f"Step 1: Found {len(current_files)} file(s) on disk")

    # ============================
    # Step 2 — Detect orphans
    # ============================
    orphan_keys = [k for k in registry if k not in current_paths_set]
    for orphan in orphan_keys:
        logger.info(f"Đã xóa orphan: {Path(orphan).name}")
        vector_store.delete_by_source(orphan)
        del registry[orphan]
        count_orphan += 1

    if orphan_keys:
        save_registry(registry)
        logger.info(f"Step 2: Cleaned up {count_orphan} orphan(s)")
    else:
        logger.info("Step 2: No orphans detected")

    # ============================
    # Steps 3-6 — Process files
    # ============================
    for file_path in current_files:
        file_key = str(file_path)
        current_hash = processor.compute_file_hash(file_path)

        if not current_hash:
            logger.warning(f"Skipping {file_path.name} — hash computation failed")
            continue

        # --- Step 3: Classify ---
        previous_hash = registry.get(file_key)

        if previous_hash is None:
            status = "NEW"
            count_new += 1
        elif previous_hash != current_hash:
            status = "MODIFIED"
            count_modified += 1
        else:
            status = "UNCHANGED"
            count_unchanged += 1
            continue  # skip — nothing to do

        # --- Step 4: Delete old vectors for MODIFIED files ---
        if status == "MODIFIED":
            deleted = vector_store.delete_by_source(file_key)
            logger.info(f"  [{status}] {file_path.name} — deleted {deleted} old chunk(s)")

        # --- Step 4 continued: Load & Split ---
        docs = processor.load(file_path)
        if not docs:
            logger.warning(f"  [{status}] {file_path.name} — no content loaded, skipping")
            continue

        chunks = processor.split(docs)
        if not chunks:
            logger.warning(f"  [{status}] {file_path.name} — no chunks after split, skipping")
            continue

        # Add file_hash to each chunk's metadata
        for chunk in chunks:
            chunk.metadata["file_hash"] = current_hash

        # --- Step 5: Embed & Upsert ---
        upserted = vector_store.upsert(chunks)

        # --- Step 6: Update registry ---
        registry[file_key] = current_hash
        save_registry(registry)

        logger.info(f"  ✓ Đã xử lý: {file_path.name} ({upserted} chunks) [{status}]")

    # ============================
    # Step 7 — Summary report
    # ============================
    total_chunks = vector_store.get_total_count()

    logger.info("=" * 60)
    logger.info("KẾT QUẢ INGESTION:")
    logger.info(f"  - Số file mới:              {count_new}")
    logger.info(f"  - Số file thay đổi:         {count_modified}")
    logger.info(f"  - Số file bỏ qua:           {count_unchanged}")
    logger.info(f"  - Số file orphan đã xóa:    {count_orphan}")
    logger.info(f"  - Tổng số chunk trong DB:   {total_chunks}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
