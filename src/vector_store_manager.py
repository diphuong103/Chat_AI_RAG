"""
VietMoney RAG — VectorStoreManager
Manages ChromaDB collection: upsert, delete, similarity search.
"""

import logging
import os
from pathlib import Path
from typing import List

# Disable ChromaDB telemetry before importing to suppress PostHog capture errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from langchain.schema import Document

logger = logging.getLogger(__name__)

COLLECTION_NAME = "vietmoney"


class VectorStoreManager:
    """Wraps ChromaDB for vector storage with upsert / delete / search."""

    def __init__(self, persist_dir: Path, embeddings):
        """Initialize ChromaDB PersistentClient and get-or-create the collection.

        Args:
            persist_dir: Path to the ChromaDB persistence directory.
            embeddings: A LangChain-compatible embedding model instance
                        (must have .embed_documents and .embed_query methods).
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — collection '{COLLECTION_NAME}' "
            f"({self.get_total_count()} vectors)"
        )

    # ------------------------------------------------------------------ #
    #  Upsert chunks (add new or update existing by chunk_id)            #
    # ------------------------------------------------------------------ #
    def upsert(self, chunks: List[Document]) -> int:
        """Embed and upsert chunks into ChromaDB.

        Each chunk must have metadata keys: chunk_id, source.
        Uses chunk_id as the Chroma document id so duplicates are
        automatically overwritten.

        Args:
            chunks: List of LangChain Document objects.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        ids = [c.metadata["chunk_id"] for c in chunks]
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        try:
            vectors = self.embeddings.embed_documents(texts)
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(ids)} chunk(s)")
            return len(ids)
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  Delete all chunks belonging to a specific source file             #
    # ------------------------------------------------------------------ #
    def delete_by_source(self, source_path: str) -> int:
        """Delete every chunk whose metadata.source matches source_path.

        Args:
            source_path: The source value stored in chunk metadata.

        Returns:
            Number of chunks deleted.
        """
        try:
            results = self.collection.get(
                where={"source": source_path},
                include=[],
            )
            ids_to_delete = results["ids"]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunk(s) for source: {source_path}")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Delete by source failed ({source_path}): {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  Similarity search                                                 #
    # ------------------------------------------------------------------ #
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Return top-k most similar documents for a query string.

        Args:
            query: The user query text.
            k: Number of results to return.

        Returns:
            List of LangChain Document objects.
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas"],
            )

            docs: List[Document] = []
            for text, meta in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                docs.append(Document(page_content=text, metadata=meta))
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Total count                                                       #
    # ------------------------------------------------------------------ #
    def get_total_count(self) -> int:
        """Return the total number of vectors in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
