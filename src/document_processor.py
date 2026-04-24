"""
VietMoney RAG — DocumentProcessor
Handles loading PDF/TXT files and splitting them into chunks with MD5-based chunk IDs.
"""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads documents from file and splits them into chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    # ------------------------------------------------------------------ #
    #  Load a single file into LangChain Documents                       #
    # ------------------------------------------------------------------ #
    def load(self, file_path: Path) -> List[Document]:
        """Load a PDF or TXT file and return a list of Documents.

        Args:
            file_path: Path to the file (.pdf or .txt).

        Returns:
            List of LangChain Document objects.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            elif suffix == ".json":
                import json
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        content = data.get("content", "")
                        metadata = {
                            "source_type": data.get("source_type", "unknown"),
                            "title": data.get("title", ""),
                            "url": data.get("url", "")
                        }
                        docs = [Document(page_content=content, metadata=metadata)]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON {file_path.name}: {e}")
                    return []
            else:
                logger.warning(f"Unsupported file type: {suffix} — skipping {file_path.name}")
                return []

            # Normalize source metadata to use forward-slash relative path
            for doc in docs:
                doc.metadata["source"] = str(file_path)

            logger.info(f"Loaded {len(docs)} page(s)/item(s) from {file_path.name}")
            return docs

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Split documents into chunks and assign chunk_id                   #
    # ------------------------------------------------------------------ #
    def split(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks and add chunk_id (MD5 of content) to metadata.

        Args:
            docs: List of LangChain Document objects.

        Returns:
            List of chunked Documents, each with metadata keys:
                - source: original file path
                - chunk_id: MD5 hex digest of chunk content
        """
        if not docs:
            return []

        chunks = self.splitter.split_documents(docs)

        for chunk in chunks:
            # Normalize Unicode (NFC) for consistent Vietnamese diacritics
            chunk.page_content = self.clean_text(chunk.page_content)
            chunk_id = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["language"] = self.detect_language(chunk.page_content)

        logger.info(f"Split into {len(chunks)} chunk(s)")
        return chunks

    # ------------------------------------------------------------------ #
    #  Compute MD5 hash of an entire file (for change detection)         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file's binary content.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest string.
        """
        file_path = Path(file_path)
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for buf in iter(lambda: f.read(8192), b""):
                    hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path.name}: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  Text cleaning — Unicode normalization & junk removal              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize Unicode and clean up whitespace.

        Applies NFC normalization (critical for Vietnamese diacritics),
        collapses excessive whitespace, and strips junk characters.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        if not text:
            return text
        # NFC normalization — ensures consistent Vietnamese characters
        text = unicodedata.normalize("NFC", text)
        # Collapse excessive newlines (max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Language detection — lightweight heuristic                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language of text using character-range heuristics.

        Checks for Vietnamese diacritical marks first, then CJK ranges,
        then defaults to English.

        Args:
            text: Input text (first 500 chars are analyzed).

        Returns:
            Language code: 'vi', 'en', 'zh', 'ja', 'ko', or 'other'.
        """
        sample = text[:500]

        # Vietnamese-specific diacritics (ắ, ằ, ẳ, ẵ, ặ, ơ, ư, đ, etc.)
        vi_pattern = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)
        vi_count = len(vi_pattern.findall(sample))
        if vi_count >= 5:
            return "vi"

        # CJK Unified Ideographs (Chinese)
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", sample))
        if zh_count >= 5:
            return "zh"

        # Japanese Hiragana/Katakana
        ja_count = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", sample))
        if ja_count >= 3:
            return "ja"

        # Korean Hangul
        ko_count = len(re.findall(r"[\uac00-\ud7af\u1100-\u11ff]", sample))
        if ko_count >= 3:
            return "ko"

        # Default to English for Latin-script text
        latin_count = len(re.findall(r"[a-zA-Z]", sample))
        if latin_count > len(sample) * 0.3:
            return "en"

        return "other"
