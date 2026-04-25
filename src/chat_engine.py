"""
VietMoney RAG — ChatEngine
Multilingual Q&A engine: cache → vector retrieval → context compression → Groq LLM.
Integrated with live exchange rate data for financial consulting.
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  System prompt — VietMoney Financial Consulting Expert             #
# ------------------------------------------------------------------ #
SYSTEM_PROMPT_TEMPLATE = """### ROLE
Bạn là Chuyên gia Tư vấn Tài chính cấp cao của VietMoney. Nhiệm vụ của bạn là giải đáp thắc mắc của khách hàng về dịch vụ cầm đồ, đổi tiền, và tỷ giá ngoại tệ một cách chính xác, minh bạch.

### KNOWLEDGE CONTEXT
1. **Dữ liệu RAG (Tĩnh):** Quy trình, hạn mức, thủ tục, vị trí chi nhánh.
2. **Dữ liệu Live (Động):** Tỷ giá ngoại tệ thời gian thực (được cung cấp trong từng lượt hỏi).

### NGUYÊN TẮC VÀNG (STRICT RULES)
- **Sự thật là trên hết:** Nếu thông tin không có trong "Ngữ cảnh" hoặc "Tỷ giá", tuyệt đối không tự đoán. Hãy trả lời: "VietMoney hiện chưa có thông tin chính xác về mục này, xin vui lòng để lại số điện thoại để nhân viên tư vấn trực tiếp."
- **Ưu tiên Tỷ giá Live:** Khi khách hỏi về giá tiền/đổi tiền, PHẢI ưu tiên dữ liệu từ phần [LIVE DATA].
- **Định dạng số:** Luôn dùng dấu chấm phân cách hàng nghìn (Ví dụ: 25.400 VND).
- **Tính toán:** Nếu khách yêu cầu tính toán (ví dụ: đổi 100 USD), hãy thực hiện phép tính dựa trên tỷ giá được cung cấp và làm tròn 2 chữ số thập phân.

### PHONG CÁCH PHẢN HỒI
- Ngôn ngữ chuyên nghiệp, lịch sự nhưng gần gũi.
- Câu trả lời nên trình bày theo dạng: Tóm tắt ý chính -> Chi tiết (bullet points) -> Lời chào/Gợi ý hành động.

QUALITY RULE (critical — prevents garbled output):
- Write each sentence COMPLETELY before starting the next one
- NEVER repeat the same sentence or phrase twice
- NEVER interleave words from two different sentences
- Each bullet point must be a single, coherent, complete thought

### THÔNG TIN HỆ THỐNG
- Thời gian hiện tại: {current_time}
- [LIVE DATA]: {live_rate_info}

### NGỮ CẢNH TRUY XUẤT (RAG):
{clean_ctx}

---
### CÂU HỎI CỦA KHÁCH HÀNG:
"{query}"

### TRẢ LỜI:
"""

# Max characters per chunk (~400 tokens ≈ 1600 chars)
MAX_CHUNK_CHARS = 1600


class ChatEngine:
    """Multilingual chat engine with caching, vector retrieval, and Groq LLM."""

    def __init__(self, vector_store, llm, embeddings, cache_path: Path,
                 exchange_rate_service=None):
        """
        Args:
            vector_store: VectorStoreManager instance.
            llm: LangChain-compatible LLM (e.g. ChatGroq).
            embeddings: Embedding model (for reference, vector_store uses it internally).
            cache_path: Path to cache.json file.
            exchange_rate_service: Optional ExchangeRateService for live rates.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.cache_path = Path(cache_path)
        self.exchange_rate_service = exchange_rate_service
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
            context = "(Không tìm thấy ngữ cảnh liên quan trong cơ sở kiến thức.)"

        # --- Step 5: Build prompt with live data & call Groq LLM ---
        try:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage

            # Get live exchange rate info
            live_rate_info = ""
            if self.exchange_rate_service:
                live_rate_info = self.exchange_rate_service.format_rates_for_prompt()
            if not live_rate_info:
                live_rate_info = "(Không có dữ liệu tỷ giá tại thời điểm này)"

            # Build the system prompt with all dynamic data
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            system_content = SYSTEM_PROMPT_TEMPLATE.format(
                current_time=current_time,
                live_rate_info=live_rate_info,
                clean_ctx=context,
                query=query,
            )

            messages = [SystemMessage(content=system_content)]

            # Inject history into the prompt
            for h_q, h_a in self.chat_history:
                messages.append(HumanMessage(content=h_q))
                messages.append(AIMessage(content=h_a))

            # Latest query
            messages.append(HumanMessage(content=query))

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
            context = "(Không tìm thấy ngữ cảnh liên quan trong cơ sở kiến thức.)"

        # Step 4: Build messages with live rate data
        live_rate_info = ""
        if self.exchange_rate_service:
            live_rate_info = self.exchange_rate_service.format_rates_for_prompt()
        if not live_rate_info:
            live_rate_info = "(Không có dữ liệu tỷ giá tại thời điểm này)"

        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            current_time=current_time,
            live_rate_info=live_rate_info,
            clean_ctx=context,
            query=query,
        )

        messages = [SystemMessage(content=system_content)]
        for h_q, h_a in self.chat_history:
            messages.append(HumanMessage(content=h_q))
            messages.append(AIMessage(content=h_a))
        messages.append(HumanMessage(content=query))

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
