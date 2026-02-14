import time
import sqlite3
import json
import logging
import requests
from typing import List, Dict, Optional
from .config import DATA_DIR, CACHE_DB_PATH, SEMANTIC_SCHOLAR_RATE_LIMIT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fields we always want from Semantic Scholar
PAPER_FIELDS = "paperId,title,abstract,url,year,citationCount,authors,references,citations"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


class ContentFetcher:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite cache."""
        self.conn = sqlite3.connect(CACHE_DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                data TEXT,
                timestamp REAL
            )
        ''')
        self.conn.commit()

    def _get_from_cache(self, paper_id: str) -> Optional[Dict]:
        """Retrieve paper data from cache."""
        self.cursor.execute('SELECT data FROM papers WHERE id = ?', (paper_id,))
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def _save_to_cache(self, paper_id: str, data: Dict):
        """Save paper data to cache."""
        self.cursor.execute('INSERT OR REPLACE INTO papers (id, data, timestamp) VALUES (?, ?, ?)',
                            (paper_id, json.dumps(data), time.time()))
        self.conn.commit()

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for papers by keyword using the Semantic Scholar API directly.
        Explicitly requests abstracts via the 'fields' parameter.
        Returns a list of paper dicts, or raises on unrecoverable failure.
        """
        logger.info(f"Searching for: '{query}' (limit={limit})")

        url = f"{S2_API_BASE}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": PAPER_FIELDS,
        }

        last_error = None
        for attempt in range(3):
            try:
                time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)  # Pre-delay to avoid 429
                resp = requests.get(url, params=params, timeout=20)

                if resp.status_code == 429:
                    wait = (attempt + 1) * 10
                    logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt+1}/3...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                body = resp.json()
                raw_items = body.get("data", [])

                if not raw_items:
                    logger.warning("Semantic Scholar returned 0 results for this query.")
                    return []

                papers = []
                skipped_no_abstract = 0
                for item in raw_items:
                    abstract = item.get("abstract") or ""
                    if not abstract:
                        skipped_no_abstract += 1

                    paper_data = {
                        "id": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "abstract": abstract,
                        "url": item.get("url", ""),
                        "year": item.get("year"),
                        "citationCount": item.get("citationCount", 0),
                        "authors": [a.get("name", "") for a in (item.get("authors") or [])],
                    }
                    if paper_data["id"]:
                        self._save_to_cache(paper_data["id"], paper_data)
                        papers.append(paper_data)

                logger.info(
                    f"Search returned {len(papers)} papers "
                    f"({skipped_no_abstract} without abstracts, kept anyway for metadata)."
                )
                return papers

            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.error(f"HTTP error on attempt {attempt+1}: {e}")
            except Exception as e:
                last_error = e
                logger.error(f"Search failed on attempt {attempt+1}: {e}")
                # Short backoff before retry for non-429 errors
                time.sleep(3)

        # If we exhausted retries, raise so the caller knows
        raise RuntimeError(
            f"Search failed after 3 retries. Last error: {last_error}"
        )

    def get_details(self, paper_id: str) -> Optional[Dict]:
        """Get details for a specific paper ID."""
        cached = self._get_from_cache(paper_id)
        if cached and cached.get("abstract"):
            return cached

        try:
            time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
            url = f"{S2_API_BASE}/paper/{paper_id}"
            params = {"fields": PAPER_FIELDS}
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 429:
                logger.warning(f"Rate limited fetching {paper_id}, skipping.")
                return cached  # Return stale cache if available
            resp.raise_for_status()

            paper = resp.json()
            data = {
                "id": paper.get("paperId", paper_id),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract") or "",
                "url": paper.get("url", ""),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount", 0),
                "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
                "references": [
                    ref.get("paperId") for ref in (paper.get("references") or [])
                    if ref.get("paperId")
                ],
                "citations": [
                    cit.get("paperId") for cit in (paper.get("citations") or [])
                    if cit.get("paperId")
                ],
            }
            self._save_to_cache(data["id"], data)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return cached  # Return stale cache if we have it

    def get_batch_details(self, paper_ids: List[str]) -> List[Dict]:
        """Get details for multiple papers. Uses POST batch endpoint, falls back to individual fetches."""
        results = []
        to_fetch = []

        # Check cache first
        for pid in paper_ids:
            cached = self._get_from_cache(pid)
            if cached and cached.get("abstract"):
                results.append(cached)
            else:
                to_fetch.append(pid)

        if not to_fetch:
            return results

        logger.info(f"Fetching batch of {len(to_fetch)} papers...")

        try:
            time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
            url = f"{S2_API_BASE}/paper/batch"
            params = {"fields": PAPER_FIELDS}
            body = {"ids": to_fetch}
            resp = requests.post(url, params=params, json=body, timeout=30)
            resp.raise_for_status()

            for paper in resp.json():
                if not paper or not paper.get("paperId"):
                    continue
                data = {
                    "id": paper.get("paperId"),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract") or "",
                    "url": paper.get("url", ""),
                    "year": paper.get("year"),
                    "citationCount": paper.get("citationCount", 0),
                    "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
                    "references": [
                        ref.get("paperId") for ref in (paper.get("references") or [])
                        if ref.get("paperId")
                    ],
                    "citations": [
                        cit.get("paperId") for cit in (paper.get("citations") or [])
                        if cit.get("paperId")
                    ],
                }
                self._save_to_cache(data["id"], data)
                results.append(data)
        except Exception as e:
            logger.warning(f"Batch fetch failed ({e}), falling back to individual fetches...")
            for pid in to_fetch:
                res = self.get_details(pid)
                if res:
                    results.append(res)

        return results
