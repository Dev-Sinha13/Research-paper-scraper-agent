import time
import sqlite3
import json
import logging
from typing import List, Dict, Optional
from semanticscholar import SemanticScholar
from .config import DATA_DIR, CACHE_DB_PATH, SEMANTIC_SCHOLAR_RATE_LIMIT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentFetcher:
    def __init__(self):
        self.sch = SemanticScholar(timeout=20)
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
        """Search for papers by keyword."""
        logger.info(f"Searching for: {query}")
        
        for attempt in range(3):
            try:
                time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)  # Pre-delay to avoid 429
                results = self.sch.search_paper(query, limit=limit)
                papers = []
                for item in results:
                    paper_data = {
                        'id': item.paperId,
                        'title': item.title,
                        'abstract': item.abstract or "",
                        'url': item.url,
                        'year': item.year,
                        'citationCount': item.citationCount,
                        'authors': [a.name for a in item.authors] if item.authors else []
                    }
                    self._save_to_cache(item.paperId, paper_data)
                    papers.append(paper_data)
                    time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
                return papers
            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 10  # 10s, 20s, 30s
                    logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt+1}/3...")
                    time.sleep(wait)
                else:
                    logger.error(f"Search failed: {e}")
                    return []
        
        logger.error("Search failed after 3 retries (rate limited).")
        return []

    def get_details(self, paper_id: str) -> Optional[Dict]:
        """Get details for a specific paper ID."""
        cached = self._get_from_cache(paper_id)
        if cached:
            return cached
        
        try:
            time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
            paper = self.sch.get_paper(paper_id)
            if not paper:
                return None
            
            data = {
                'id': paper.paperId,
                'title': paper.title,
                'abstract': paper.abstract or "",
                'url': paper.url,
                'year': paper.year,
                'citationCount': paper.citationCount,
                'authors': [a.name for a in paper.authors] if paper.authors else [],
                'references': [ref.paperId for ref in paper.references if ref.paperId] if paper.references else [],
                'citations': [cit.paperId for cit in paper.citations if cit.paperId] if paper.citations else []
            }
            self._save_to_cache(paper_id, data)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return None

    def get_batch_details(self, paper_ids: List[str]) -> List[Dict]:
        """Get details for multiple papers efficiently."""
        results = []
        to_fetch = []

        # Check cache first
        for pid in paper_ids:
            cached = self._get_from_cache(pid)
            if cached:
                results.append(cached)
            else:
                to_fetch.append(pid)
        
        if not to_fetch:
            return results

        logger.info(f"Fetching batch of {len(to_fetch)} papers...")
        try:
            # Semantic Scholar API might support batch, if not we loop
            # The library has get_papers() method usually
            papers = self.sch.get_papers(to_fetch)
            for paper in papers:
                if not paper: continue
                data = {
                    'id': paper.paperId,
                    'title': paper.title,
                    'abstract': paper.abstract or "",
                    'url': paper.url,
                    'year': paper.year,
                    'citationCount': paper.citationCount or 0,
                    'authors': [a.name for a in paper.authors] if paper.authors else [],
                    'references': [ref.paperId for ref in paper.references if ref.paperId] if paper.references else [],
                    'citations': [cit.paperId for cit in paper.citations if cit.paperId] if paper.citations else []
                }
                self._save_to_cache(paper.paperId, data)
                results.append(data)
                time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT / 10) # Smaller sleep for batch
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fallback to singular fetch if batch fails
            for pid in to_fetch:
                res = self.get_details(pid)
                if res: results.append(res)
        
        return results
