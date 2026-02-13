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
        
        # We can store summary in state, but ResearchState needs a field for it?
        # Let's just return it as a "global summary" maybe?
        # For now, we update the papers with individual summaries if needed, 
        # or just return a key in the state.
        
        return {"summary": summary}


    # --- Node Implementations ---

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
            
        # Fetch seeds
        raw_papers = self.fetcher.search(query, limit=MAX_SEARCH_RESULTS)
        
        # Convert to Paper objects
        new_papers = {}
        for p in raw_papers:
            if not p['abstract']: continue
            
            # Embed paper abstract
            vec = self.embedder.embed(p['abstract'])
            sim = self.embedder.similarity(state['query_vector'], vec)
            
            paper_obj: Paper = {
                'id': p['id'],
                'title': p['title'],
                'abstract': p['abstract'],
                'authors': p['authors'],
                'year': p['year'],
                'citation_count': p['citationCount'],
                'url': p['url'],
                'vector': vec,
                'relevance_score': sim,
                'summary': ""
            }
            new_papers[p['id']] = paper_obj
            
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
        
        # Fetch details (references)
        details = self.fetcher.get_details(current_id)
        if not details or not details.get('references'):
            return {"visited_ids": visited}
            
        # Fetch data for references (Limit to top 5 to save time/quota)
        ref_ids = details['references'][:5] 
        new_raw_papers = self.fetcher.get_batch_details(ref_ids)
        
        new_papers = {}
        for p in new_raw_papers:
            if p['id'] in papers: continue # Already have it
            if not p['abstract']: continue
            
            # Embed
            vec = self.embedder.embed(p['abstract'])
            sim = self.embedder.similarity(state['query_vector'], vec)
            
            paper_obj: Paper = {
                'id': p['id'],
                'title': p['title'],
                'abstract': p['abstract'],
                'authors': p['authors'],
                'year': p['year'],
                'citation_count': p['citationCount'],
                'url': p['url'],
                'vector': vec,
                'relevance_score': sim,
                'summary': ""
            }
            new_papers[p['id']] = paper_obj

        # Merge
        papers.update(new_papers)
        
        # Update queue (add new papers)
        new_queue = list(queue) + list(new_papers.keys())
        
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
        
        # Sort queue by relevance score of the papers
        # High relevance first
        queue.sort(key=lambda pid: papers[pid]['relevance_score'], reverse=True)
        
        # Prune: Remove low relevance from queue (but keep in graph?)
        # For now, just sort so we expand the best ones first.
        
        return {"queue": queue}

    def should_continue(self, state: ResearchState):
        """Decide whether to stop or continue."""
        # Check timeout
        start_time = state.get('start_time')
        max_duration = state.get('max_duration', 60) # Default 60s
        if start_time and (time.time() - start_time > max_duration):
            logger.info("Max duration reached. Stopping.")
            return "stop"

        if state['current_depth'] >= MAX_DEPTH:
            return "stop"
        if not state['queue']:
            return "stop"
        
        # Check if we have enough "High Relevance" papers?
        # For now, just depth + queue check
        return "continue"
