from typing import List, Dict, Set, Optional, TypedDict, Annotated
import operator

class Paper(TypedDict):
    """Represents a single research paper."""
    id: str  # Semantic Scholar ID
    title: str
    abstract: str
    authors: List[str]
    year: int
    citation_count: int
    url: str
    vector: Optional[List[float]] = None  # Embedding vector
    relevance_score: float = 0.0
    summary: str = ""  # RAG-generated summary

class ResearchState(TypedDict):
    """
    The state of the research agent.
    Managed by LangGraph to track progress.
    """
    query: str  # The original user query/abstract
    query_vector: List[float]  # Embedding of the query
    
    papers: Dict[str, Paper]  # All found papers, keyed by ID
    
    # Queue of Paper IDs to process in the next step
    # We use a list for the queue to maintain order
    queue: List[str] 
    
    # Set of IDs we have already processed/expanded
    visited_ids: Set[str]
    
    current_depth: int
    max_depth: int
    start_time: float # Timestamp when search started
    max_duration: float # Max seconds to run
    summary: str  # Global summary of findings
