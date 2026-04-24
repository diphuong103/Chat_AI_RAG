"""
Webhook server using FastAPI.
This server listens for POST requests from external automation tools like Make.com.
It extracts data (e.g., RSS items with URLs), saves them or scrapes them, and triggers ingestion.
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.scraper_service import ScraperService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VietMoney Webhook Server")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_WEB_DIR = BASE_DIR / "data" / "raw" / "web"
RAW_WEB_DIR.mkdir(parents=True, exist_ok=True)

class RssPayload(BaseModel):
    title: Optional[str] = None
    url: str
    description: Optional[str] = None
    content: Optional[str] = None

def trigger_ingestion_pipeline():
    """Run the ingestion pipeline in the background."""
    from ingest_data import run_pipeline
    try:
        logger.info("Triggering background ingestion pipeline...")
        run_pipeline()
        logger.info("Background ingestion pipeline completed.")
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")

@app.post("/webhook/make")
async def receive_make_payload(payload: RssPayload, background_tasks: BackgroundTasks):
    """
    Endpoint for Make.com to push RSS items.
    If 'content' is provided, we save it as a JSON file right away.
    If only 'url' is provided, we use the scraper service to fetch the content.
    After saving, we background-trigger the ingest pipeline.
    """
    logger.info(f"Received webhook payload from Make.com: url={payload.url}")

    saved_path = None

    if payload.content and payload.title:
        # We have direct content, save as json
        doc_id = str(uuid.uuid4())[:8]
        file_name = f"rss_make_{doc_id}.json"
        saved_path = RAW_WEB_DIR / file_name
        
        data = {
            "title": payload.title,
            "url": payload.url,
            "content": payload.content,
            "source": "Make.com RSS Webhook",
            "scraped_at": datetime.now().isoformat()
        }
        
        try:
            with open(saved_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved payload content to {saved_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
            raise HTTPException(status_code=500, detail="Could not save content")
            
    elif payload.url:
        # Scrape the URL
        scraper = ScraperService(base_dir=BASE_DIR)
        saved_path_str = scraper.scrape_url(payload.url)
        if saved_path_str:
            saved_path = Path(saved_path_str)
            logger.info(f"Successfully scraped URL: {payload.url} -> {saved_path}")
        else:
            logger.warning(f"Failed to scrape URL or blocked: {payload.url}")
            return {"status": "error", "message": "Failed to scrape URL"}
            
    else:
        raise HTTPException(status_code=400, detail="Must provide at least 'url', or 'url' + 'content'")

    # Background task to ingest data incrementally
    if saved_path:
        background_tasks.add_task(trigger_ingestion_pipeline)
    
    return {
        "status": "success",
        "message": "Data received and scheduled for ingestion.",
        "file_saved": str(saved_path) if saved_path else None
    }
