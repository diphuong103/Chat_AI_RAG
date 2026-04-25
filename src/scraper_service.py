"""
VietMoney RAG — ScraperService
Scrapes content from Wikipedia and arbitrary URLs, saves as .txt files.
"""

import logging
import re
import unicodedata
from pathlib import Path

import cloudscraper
import requests
import wikipedia
import trafilatura
from bs4 import BeautifulSoup
from slugify import slugify

logger = logging.getLogger(__name__)


class ScraperService:
    """Scrapes Wikipedia articles and web pages, saves cleaned text to disk."""

    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: Project root directory (e.g. D:/Project_x/Train_AI_VIETMONEY).
        """
        self.base_dir = Path(base_dir)
        self.wiki_dir = self.base_dir / "data" / "raw" / "wiki"
        self.web_dir = self.base_dir / "data" / "raw" / "web"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Wikipedia scraping                                                #
    # ------------------------------------------------------------------ #
    def scrape_wikipedia(self, keyword: str, lang: str = "vi") -> str:
        """Fetch a Wikipedia article by keyword and save as .txt.

        Tries the specified language first, falls back to English if not found.

        Args:
            keyword: Search term (e.g. "lãi suất ngân hàng").
            lang: Wikipedia language code, default "vi".

        Returns:
            Absolute path to the saved file, or empty string on failure.
        """
        for attempt_lang in [lang, "en"] if lang != "en" else ["en"]:
            try:
                wikipedia.set_lang(attempt_lang)
                page = wikipedia.page(keyword, auto_suggest=True)
                content = page.content

                if not content or len(content.strip()) < 50:
                    logger.warning(
                        f"Wikipedia ({attempt_lang}): content too short for '{keyword}', trying next language"
                    )
                    continue

                slug = self._slugify(keyword)
                file_path = self.wiki_dir / f"{slug}.json"
                
                content = self._clean_text(content)
                data = {
                    "source_type": "wikipedia",
                    "title": page.title,
                    "url": page.url,
                    "language": attempt_lang,
                    "content": content
                }

                import json
                file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"Đã lưu: wiki/{file_path.name}")
                return str(file_path)

            except wikipedia.exceptions.DisambiguationError as e:
                logger.warning(f"Wikipedia disambiguation for '{keyword}': {e.options[:5]}")
                # Try the first suggestion
                try:
                    wikipedia.set_lang(attempt_lang)
                    page = wikipedia.page(e.options[0])
                    content = page.content
                    slug = self._slugify(keyword)
                    file_path = self.wiki_dir / f"{slug}.json"
                    
                    content = self._clean_text(content)
                    data = {
                        "source_type": "wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "language": attempt_lang,
                        "content": content
                    }
                    import json
                    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    logger.info(f"Đã lưu: wiki/{file_path.name}")
                    return str(file_path)
                except Exception:
                    continue

            except wikipedia.exceptions.PageError:
                logger.warning(f"Wikipedia ({attempt_lang}): page not found for '{keyword}'")
                continue

            except Exception as e:
                logger.error(f"Wikipedia ({attempt_lang}) error for '{keyword}': {e}")
                continue

        logger.error(f"Could not scrape Wikipedia for '{keyword}' in any language")
        return ""

    def scrape_wikipedia_deep(self, keyword: str, lang: str = "vi", max_related: int = 10) -> list[str]:
        """Deep scrape: main keyword + related internal links/searches.

        Args:
            keyword: The main search keyword (e.g., 'Di tích lịch sử Việt Nam').
            lang: Language code.
            max_related: Maximum number of related pages to scrape.

        Returns:
            List of absolute paths to all saved files.
        """
        saved_paths = []
        
        # 1. Scrape the main keyword first
        logger.info(f"[DeepScrape] Bắt đầu cào chính: '{keyword}'")
        main_path = self.scrape_wikipedia(keyword, lang)
        if main_path:
            saved_paths.append(main_path)
            
        # 2. Extract related keywords (Expansion + Internal Links + Search)
        related_pages = self._get_wiki_related_pages(keyword, lang)
        expanded_keywords = self._expand_keywords(keyword)
        
        # Combine and deduplicate
        all_targets = list(dict.fromkeys(expanded_keywords + related_pages))
        
        # If the main keyword was already processed, remove it
        if keyword in all_targets:
            all_targets.remove(keyword)
            
        logger.info(f"[DeepScrape] Tìm thấy {len(all_targets)} bài viết liên quan. Đang cào tối đa {max_related} bài...")
        
        # 3. Scrape related pages
        scraped_count = 0
        for target in all_targets:
            if scraped_count >= max_related:
                break
            # Skip empty or overly broad targets
            if not target or len(target) < 3:
                continue
                
            logger.info(f"[DeepScrape] Đang cào bài liên quan ({scraped_count+1}/{max_related}): '{target}'")
            try:
                path = self.scrape_wikipedia(target, lang)
                if path:
                    saved_paths.append(path)
                    scraped_count += 1
            except Exception as e:
                logger.error(f"[DeepScrape] Lỗi khi cào bài '{target}': {e}")
                
        return saved_paths

    def _expand_keywords(self, keyword: str) -> list[str]:
        """Expand a geography/topic keyword with common tourism sub-topics."""
        # Generic topics that apply broadly
        sub_topics = [
            "Lễ hội", "Ẩm thực", "Du lịch", "Đặc sản", "Lịch sử", 
            "Điểm tham quan", "Văn hóa", "Di tích"
        ]
        
        expanded = []
        for topic in sub_topics:
            expanded.append(f"{topic} {keyword}")
        return expanded
        
    def _get_wiki_related_pages(self, keyword: str, lang: str = "vi") -> list[str]:
        """Use Wikipedia Search API and MediaWiki link extraction to find related pages."""
        related = []
        try:
            wikipedia.set_lang(lang)
            
            # 1. Use standard search API
            search_results = wikipedia.search(keyword, results=10)
            related.extend(search_results)
            
            # 2. Try to get links from the main page
            try:
                page = wikipedia.page(keyword, auto_suggest=True)
                links = page.links
                valid_links = [l for l in links if not l.startswith("Thể loại:") and not l.startswith("Bản mẫu:")]
                related.extend(valid_links[:15]) # Take top 15 links
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to get related pages for '{keyword}': {e}")
            
        return list(dict.fromkeys(related)) # deduplicate


    # ------------------------------------------------------------------ #
    #  URL scraping (trafilatura primary, BeautifulSoup fallback)        #
    # ------------------------------------------------------------------ #
    def scrape_url(self, url: str) -> str:
        """Fetch and extract main content from an arbitrary URL.

        Primary: uses cloudscraper to bypass anti-bot, then trafilatura to extract 
                 clean text (removes nav, ads, footers).
        Fallback: BeautifulSoup <p> extraction.

        Args:
            url: Full URL to scrape.

        Returns:
            Absolute path to the saved file, or empty string on failure.
        """
        content = None
        html = None

        # Fetch HTML using cloudscraper to bypass 403 Forbidden / Cloudflare
        try:
            logger.info(f"Tải HTML từ: {url}")
            scraper_client = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
            resp = scraper_client.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.error(f"Failed to fetch url {url}: {e}")
            return ""

        # --- Primary extraction: trafilatura (cleanest) ---
        title = ""
        if html:
            extracted = trafilatura.extract(html, include_links=False, include_images=False, include_comments=False)
            if extracted and len(extracted.strip()) > 50:
                content = self._clean_text(extracted)
                logger.info(f"Trafilatura lọc thành công bài viết (bỏ qua rác).")
                # Try to get title from BeautifulSoup since trafilatura strips the head
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    title = soup.title.string.strip() if soup.title else ""
                except Exception:
                    pass

        # --- Fallback: BeautifulSoup ---
        if not content and html:
            try:
                logger.info(f"Trafilatura không lấy được, fallback sang BeautifulSoup.")
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string.strip() if soup.title else ""
                paragraphs = soup.find_all("p")
                raw = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                content = self._clean_text(raw)
            except Exception as e:
                logger.error(f"BeautifulSoup fallback failed: {e}")
                return ""

        if not content or len(content.strip()) < 50:
            logger.error(f"Không tìm thấy nội dung hữu ích (hoặc quá ngắn) tại {url}")
            return ""

        slug = self._slugify(url)
        file_path = self.web_dir / f"{slug}.json"
        
        if not title:
            title = url

        # Detect language from content
        from src.document_processor import DocumentProcessor
        detected_lang = DocumentProcessor.detect_language(content)

        data = {
            "source_type": "web",
            "title": title,
            "url": url,
            "language": detected_lang,
            "content": content
        }

        try:
            import json
            file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Đã lưu: web/{file_path.name}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save web content: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  Text cleaning helper                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_text(text: str) -> str:
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
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Slugify helper                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filesystem-safe slug.

        Uses python-slugify, limited to 80 characters.

        Args:
            text: Input string (URL, keyword, etc.)

        Returns:
            Slugified string.
        """
        return slugify(text, max_length=80)
