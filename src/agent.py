from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from .models import ResearchState, Paper
from .fetcher import ContentFetcher
from .embeddings import Embedder
from .rag import RAGClient
from .config import SIMILARITY_THRESHOLD, MAX_DEPTH, MAX_SEARCH_RESULTS
import logging
import time

logger = logging.getLogger(__name__)

class ResearchGraph:
    def __init__(self):
        self.fetcher = ContentFetcher()
        self.embedder = Embedder()
        self.rag = RAGClient()
        self.workflow = StateGraph(ResearchState)
        self._build_graph()

    def _build_graph(self):
        # Add Nodes
        self.workflow.add_node("search_seeds", self.search_seeds)
        self.workflow.add_node("expand_node", self.expand_node)
        self.workflow.add_node("filter_node", self.filter_node)
        self.workflow.add_node("synthesize_node", self.synthesize_node)
        
        # Set Entry Point
        self.workflow.set_entry_point("search_seeds")
        
        # Add Edges
        self.workflow.add_edge("search_seeds", "filter_node")
        self.workflow.add_conditional_edges(
            "filter_node",
            self.should_continue,
            {
                "continue": "expand_node",
                "stop": "synthesize_node"
            }
        )
        self.workflow.add_edge("expand_node", "filter_node")
        self.workflow.add_edge("synthesize_node", END)

    def compile(self):
        return self.workflow.compile()

    # --- Node Implementations ---
    
    def synthesize_node(self, state: ResearchState) -> Dict:
        """Use LLM to summarize findings."""
        logger.info("Node: Synthesize")
        papers = state.get('papers', {})
        query = state.get('query', "")
        
        # Convert dict values to list
        paper_list = list(papers.values())
        
        # Sort by relevance
        paper_list.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        summary = self.rag.summarize_papers(paper_list, query)
        
        return {"summary": summary}


    def search_seeds(self, state: ResearchState) -> Dict:
        """Initial search for seed papers."""
        logger.info("Node: Search Seeds")
        query = state.get('query', '')
        
        # Initialize start time if not present
        start_time = state.get('start_time')
        if not start_time:
            start_time = time.time()
        
        # Generate query vector if not present
        if not state.get('query_vector'):
            state['query_vector'] = self.embedder.embed(query)
        
        query_vector = state['query_vector']
        if not query_vector:
            raise RuntimeError(
                "Embedding model failed to produce a query vector. "
                "Check that the embedding model loaded correctly."
            )

        # Fetch seeds — this now raises on total failure instead of silently returning []
        raw_papers = self.fetcher.search(query, limit=MAX_SEARCH_RESULTS)
        
        if not raw_papers:
            logger.warning("Search returned 0 papers. The query may be too specific or the API returned no results.")
        
        # Convert to Paper objects
        new_papers = {}
        skipped = 0
        for p in raw_papers:
            abstract = p.get('abstract', '')
            
            # If no abstract, still keep the paper but give it a low relevance score
            if not abstract:
                skipped += 1
                paper_obj: Paper = {
                    'id': p['id'],
                    'title': p['title'],
                    'abstract': "",
                    'authors': p.get('authors', []),
                    'year': p.get('year'),
                    'citation_count': p.get('citationCount', 0),
                    'url': p.get('url', ''),
                    'vector': [],
                    'relevance_score': 0.1,  # Low default score
                    'summary': ""
                }
                new_papers[p['id']] = paper_obj
                continue
            
            # Embed paper abstract and score relevance
            vec = self.embedder.embed(abstract)
            sim = self.embedder.similarity(query_vector, vec)
            
            paper_obj: Paper = {
                'id': p['id'],
                'title': p['title'],
                'abstract': abstract,
                'authors': p.get('authors', []),
                'year': p.get('year'),
                'citation_count': p.get('citationCount', 0),
                'url': p.get('url', ''),
                'vector': vec,
                'relevance_score': sim,
                'summary': ""
            }
            new_papers[p['id']] = paper_obj
        
        logger.info(
            f"Seed search produced {len(new_papers)} papers "
            f"({skipped} without abstracts)."
        )
            
        return {
            "papers": new_papers,
            "queue": list(new_papers.keys()),
            "visited_ids": set(),
            "current_depth": 0,
            "start_time": start_time
        }

    def expand_node(self, state: ResearchState) -> Dict:
        """Expand the most relevant unvisited paper."""
        logger.info("Node: Expand")
        queue = state.get('queue', [])
        visited = state.get('visited_ids', set())
        papers = state.get('papers', {})
        query_vector = state.get('query_vector', [])
        
        # Get next paper from queue that hasn't been visited
        current_id = None
        for pid in queue:
            if pid not in visited:
                current_id = pid
                break
        
        if not current_id:
            return {"queue": []}  # Nothing left to expand
            
        # Mark as visited
        visited.add(current_id)
        logger.info(f"Expanding paper: {papers.get(current_id, {}).get('title', current_id)}")
        
        # Fetch details (references)
        details = self.fetcher.get_details(current_id)
        if not details or not details.get('references'):
            return {"visited_ids": visited}
            
        # Fetch data for references (Limit to top 5 to save time/quota)
        ref_ids = details['references'][:5] 
        new_raw_papers = self.fetcher.get_batch_details(ref_ids)
        
        new_papers = {}
        for p in new_raw_papers:
            if p['id'] in papers: continue  # Already have it
            abstract = p.get('abstract', '')
            
            if not abstract:
                # Keep paper with low score
                paper_obj: Paper = {
                    'id': p['id'],
                    'title': p.get('title', ''),
                    'abstract': "",
                    'authors': p.get('authors', []),
                    'year': p.get('year'),
                    'citation_count': p.get('citationCount', 0),
                    'url': p.get('url', ''),
                    'vector': [],
                    'relevance_score': 0.1,
                    'summary': ""
                }
                new_papers[p['id']] = paper_obj
                continue
            
            # Embed
            vec = self.embedder.embed(abstract)
            sim = self.embedder.similarity(query_vector, vec)
            
            paper_obj: Paper = {
                'id': p['id'],
                'title': p.get('title', ''),
                'abstract': abstract,
                'authors': p.get('authors', []),
                'year': p.get('year'),
                'citation_count': p.get('citationCount', 0),
                'url': p.get('url', ''),
                'vector': vec,
                'relevance_score': sim,
                'summary': ""
            }
            new_papers[p['id']] = paper_obj

        # Merge
        papers.update(new_papers)
        
        # Update queue (add new papers)
        new_queue = list(queue) + list(new_papers.keys())
        
        logger.info(f"Expand added {len(new_papers)} new papers.")
        
        return {
            "papers": papers,
            "queue": new_queue,
            "visited_ids": visited,
            "current_depth": state.get('current_depth', 0) + 1
        }

    def filter_node(self, state: ResearchState) -> Dict:
        """Re-rank the queue based on relevance."""
        logger.info("Node: Filter & Rank")
        papers = state['papers']
        queue = state['queue']
        
        # Remove papers from queue that aren't in our papers dict (safety check)
        queue = [pid for pid in queue if pid in papers]
        
        # Sort queue by relevance score of the papers
        # High relevance first
        queue.sort(key=lambda pid: papers[pid].get('relevance_score', 0), reverse=True)
        
        return {"queue": queue}

    def should_continue(self, state: ResearchState):
        """Decide whether to stop or continue."""
        # Check timeout
        start_time = state.get('start_time')
        max_duration = state.get('max_duration', 120)  # Increased default to 120s
        if start_time and (time.time() - start_time > max_duration):
            logger.info("Max duration reached. Stopping.")
            return "stop"

        if state['current_depth'] >= MAX_DEPTH:
            return "stop"
        if not state['queue']:
            return "stop"
        
        return "continue"
