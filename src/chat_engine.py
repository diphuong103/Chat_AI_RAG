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
SYSTEM_PROMPT = """You are VietMoney Assistant — a friendly and enthusiastic Vietnam Travel and Financial Assistant.

LANGUAGE RULE (highest priority — always follow this):
- Detect the language of the user's question automatically
- ALWAYS respond in THE SAME LANGUAGE as the user's question
- Question in English   → Answer in English
- Question in Vietnamese → Answer in Vietnamese
- Question in any other language → Answer in that language
- NEVER mix languages in a single response

TONE RULE (very important):
- Be warm, friendly, and enthusiastic — like a local friend giving travel advice
- Use appropriate emoji (🏖️ 🌿 🎉 📍 ✨ 🍜) to make responses lively
- Start responses with a warm greeting or excited opener, for example:
  (VI) "Ôi, bạn hỏi đúng chỗ rồi! 🎉" or "Tuyệt vời! Để mình chia sẻ nhé! ✨"
  (EN) "Great question! 🎉" or "Oh, you're going to love this! ✨"
- End with an encouraging note, e.g. "Chúc bạn có chuyến đi thật vui! 🥰"
- NEVER sound robotic or formal — be natural and conversational

QUALITY RULE (critical — prevents garbled output):
- Write each sentence COMPLETELY before starting the next one
- NEVER repeat the same sentence or phrase twice
- NEVER interleave words from two different sentences
- Each bullet point must be a single, coherent, complete thought
- If listing places, list each place ONCE with a brief description

KNOWLEDGE RULE:
- Use ONLY the provided context to answer
- If the answer is not found in the context, respond honestly
  in the same language as the question:
  (EN) "I don't have enough information to answer this question."
  (VI) "Mình chưa có đủ thông tin để trả lời câu hỏi này bạn ơi 😊"
- Never fabricate financial information

FORMATTING RULE:
- Provide helpful, concise, and accurate information
- Format itineraries in clear days (Day 1, Day 2...)
- Include budget estimates and ATM locations if relevant
- Use bullet points and short paragraphs for readability"""

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
            "IMPORTANT: Keep the rewritten query in the SAME LANGUAGE as the user's latest message. "
            "Do NOT translate the query to another language. "
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
            relevant_docs = self.vector_store.similarity_search(standalone_query, k=5)
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
            answer = self._clean_response(response.content.strip())
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
    #  Streaming ask method (for FastAPI SSE)                             #
    # ------------------------------------------------------------------ #
    async def stream_ask(self, query: str, history: list[dict]):
        """Async generator: stream the RAG answer token-by-token.

        Yields (event_type, content) tuples:
            ("token", "partial text...")  — each LLM token
            ("sources", [{...}, ...])    — document sources at end

        Args:
            query: User question.
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        from langchain.schema import HumanMessage, SystemMessage, AIMessage

        # Rebuild chat_history from provided history
        paired_history = []
        for i in range(0, len(history) - 1, 2):
            if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":
                paired_history.append((history[i]["content"], history[i + 1]["content"]))
        self.chat_history = paired_history[-self.max_history:]

        # Step 1: Rewrite query
        standalone_query = self._rewrite_query(query)

        # Step 2: Vector retrieval
        try:
            relevant_docs = self.vector_store.similarity_search(standalone_query, k=5)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            relevant_docs = []

        # Step 3: Context compression
        context = self._compress_context(relevant_docs)
        if not context.strip():
            context = "(No relevant context found in the knowledge base.)"

        # Step 4: Build messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for h_q, h_a in self.chat_history:
            messages.append(HumanMessage(content=h_q))
            messages.append(AIMessage(content=h_a))
        messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

        # Step 5: Stream LLM response & collect full text for cleaning
        full_response = ""
        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full_response += token
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            full_response = "Xin lỗi, đã xảy ra lỗi. / Sorry, an error occurred."

        # Clean the full response before yielding
        cleaned = self._clean_response(full_response)
        yield ("token", cleaned)

        # Step 6: Yield sources
        sources = self._extract_sources(relevant_docs)
        if sources:
            yield ("sources", sources)

    # ------------------------------------------------------------------ #
    #  Extract document sources for reference cards                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_sources(docs: List[Document]) -> list[dict]:
        """Extract unique source metadata from retrieved docs.

        Returns:
            List of dicts with keys: source, title, url.
        """
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata
            source_path = meta.get("source", "")
            title = meta.get("title", "")
            url = meta.get("url", "")

            # Use URL or source path as dedup key
            key = url or source_path
            if key and key not in seen:
                seen.add(key)
                sources.append({
                    "source": Path(source_path).name if source_path else "",
                    "title": title or Path(source_path).stem if source_path else "Unknown",
                    "url": url,
                })
        return sources

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
    #  Response cleaning (removes duplicate/garbled lines)                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove duplicate or near-duplicate lines from LLM output.

        Handles a known issue where the LLM sometimes interleaves or
        repeats sentences in Vietnamese text generation.

        Args:
            text: Raw LLM response string.

        Returns:
            Cleaned text with duplicate lines removed.
        """
        if not text:
            return text

        lines = text.split("\n")
        seen: set[str] = set()
        cleaned_lines: list[str] = []

        for line in lines:
            # Normalize for comparison: strip whitespace and collapse spaces
            normalized = re.sub(r"\s+", " ", line.strip()).lower()

            # Skip empty lines (but keep one blank line for formatting)
            if not normalized:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            # Skip exact duplicate lines
            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned_lines.append(line)

        # Remove trailing empty lines
        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

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
