"""
VietMoney RAG — ChatEngine
Multilingual Q&A engine: cache → vector retrieval → context compression → Groq LLM.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  System prompt — multilingual, financial focus                     #
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = """You are VietMoney Assistant, a multilingual financial advisor chatbot.

LANGUAGE RULE (highest priority — always follow this):
- Detect the language of the user's question automatically
- ALWAYS respond in THE SAME LANGUAGE as the user's question
- Question in English   → Answer in English
- Question in Vietnamese → Answer in Vietnamese
- Question in any other language → Answer in that language
- NEVER mix languages in a single response

KNOWLEDGE RULE:
- Use ONLY the provided context to answer
- If the answer is not found in the context, respond honestly
  in the same language as the question:
  (EN) "I don't have enough information to answer this question."
  (VI) "Tôi không có đủ thông tin để trả lời câu hỏi này."
- Never fabricate financial information"""

# Max characters per chunk (~400 tokens ≈ 1600 chars)
MAX_CHUNK_CHARS = 1600


class ChatEngine:
    """Multilingual chat engine with caching, vector retrieval, and Groq LLM."""

    def __init__(self, vector_store, llm, embeddings, cache_path: Path):
        """
        Args:
            vector_store: VectorStoreManager instance.
            llm: LangChain-compatible LLM (e.g. ChatGroq).
            embeddings: Embedding model (for reference, vector_store uses it internally).
            cache_path: Path to cache.json file.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.cache_path = Path(cache_path)
        self.cache: dict = self._load_cache()
        self.chat_history = []
        self.max_history = 5

    # ------------------------------------------------------------------ #
    #  Query Rewriting (For Contextual Memory)                           #
    # ------------------------------------------------------------------ #
    def _rewrite_query(self, query: str) -> str:
        """Use LLM to rewrite contextual follow-up questions to standalone queries."""
        if not self.chat_history:
            return query
            
        from langchain.schema import HumanMessage, SystemMessage
        
        history_str = "\n".join([f"User: {u}\nBot: {b}" for u, b in self.chat_history])
        sys_prompt = (
            "You are a standalone query generator. Given the chat history and the user's latest message, "
            "synthesize a single standalone search query out of the latest message that includes all necessary context from the history. "
            "Your output MUST ONLY be the rewritten query text, nothing else. "
            "If the latest message is fully independent and needs no context, just output it as is."
        )
        user_msg = f"History:\n{history_str}\n\nLatest user message: {query}\nRewritten standalone query:"
        
        try:
            resp = self.llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
            rewritten = resp.content.strip()
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query

    # ------------------------------------------------------------------ #
    #  Main ask method                                                   #
    # ------------------------------------------------------------------ #
    def ask(self, query: str) -> str:
        """Process a user query through the full RAG pipeline with Chat History buffer.

        Pipeline: rewrite query → cache check → retrieval → compression → LLM → save history & cache.

        Args:
            query: User's question string.

        Returns:
            Answer string from cache or LLM.
        """
        start_time = time.time()

        # --- Step 1: Rewrite Query (if history exists) ---
        standalone_query = self._rewrite_query(query)

        # --- Step 2: Cache check (History-Aware) ---
        # Include history state in cache key to avoid cross-context bleeding
        history_state = "|".join([f"{u}{b}" for u,b in self.chat_history])
        cache_str = f"{history_state}|{query.strip().lower()}"
        cache_key = hashlib.md5(cache_str.encode("utf-8")).hexdigest()
        
        cached = self._check_cache(cache_key)
        if cached is not None:
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[CACHE HIT] Answered from cache ({elapsed:.0f}ms)")
            # Add to history even if cached
            self.chat_history.append((query, cached))
            if len(self.chat_history) > self.max_history:
                self.chat_history.pop(0)
            return cached

        # --- Step 3: Vector retrieval (using standalone query) ---
        try:
            relevant_docs = self.vector_store.similarity_search(standalone_query, k=3)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            relevant_docs = []

        # --- Step 4: Context compression ---
        context = self._compress_context(relevant_docs)

        if not context.strip():
            context = "(No relevant context found in the knowledge base.)"

        # --- Step 5: Call Groq LLM with Chat History ---
        try:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage

            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            
            # Inject history into the prompt
            for h_q, h_a in self.chat_history:
                messages.append(HumanMessage(content=h_q))
                messages.append(AIMessage(content=h_a))
                
            # Latest query with RAG context
            messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))
            
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Return error but do NOT cache — so next attempt gets a fresh try
            return (
                "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. "
                "/ Sorry, an error occurred while processing your question."
            )

        # --- Step 6: Save cache & update history ---
        self._save_cache(cache_key, answer)
        self.chat_history.append((query, answer))
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Answered in {elapsed:.0f}ms")
        return answer

    # ------------------------------------------------------------------ #
    #  Cache methods                                                     #
    # ------------------------------------------------------------------ #
    def _check_cache(self, key: str) -> Optional[str]:
        """Return cached answer if key exists, else None."""
        return self.cache.get(key)

    def _save_cache(self, key: str, answer: str) -> None:
        """Save answer to in-memory cache and persist to disk."""
        self.cache[key] = answer
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist cache: {e}")

    def _load_cache(self) -> dict:
        """Load cache from disk, or return empty dict."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} cached answer(s)")
                return data
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Cache file corrupted, starting fresh: {e}")
        return {}

    # ------------------------------------------------------------------ #
    #  Context compression                                               #
    # ------------------------------------------------------------------ #
    def _compress_context(self, chunks: List[Document]) -> str:
        """Clean and compress retrieved chunks into a single context string.

        - Removes excessive whitespace
        - Removes special characters that don't add meaning
        - Truncates each chunk to ~400 tokens (MAX_CHUNK_CHARS)

        Args:
            chunks: List of retrieved Document objects.

        Returns:
            Concatenated, cleaned context string.
        """
        if not chunks:
            return ""

        compressed_parts: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.page_content

            # Remove excessive whitespace (multiple spaces/newlines → single)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            text = text.strip()

            # Truncate to max chars
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS] + "..."

            if text:
                compressed_parts.append(f"[Source {i}]\n{text}")

        return "\n\n".join(compressed_parts)
