import google.generativeai as genai
import os
import logging
from typing import List, Dict
from .config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

class RAGClient:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found. RAG features will be disabled.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(LLM_MODEL_NAME)

    def summarize_papers(self, papers: List[Dict], query: str) -> str:
        """
        Generate a synthesis of the provided papers related to the query.
        """
        if not self.model:
            return "RAG Disabled: No API Key provided."
            
        if not papers:
            return "No papers found to summarize."

        # Construct Prompt
        context = "\n\n".join([
            f"Title: {p['title']}\nYear: {p['year']}\nAbstract: {p['abstract']}" 
            for p in papers[:10]  # Limit context window
        ])
        
        prompt = f"""
        You are a research assistant. The user is investigating: "{query}".
        
        Here are the most relevant papers found:
        
        {context}
        
        Synthesize these findings into a concise 1-paragraph summary. 
        Highlight the key themes and how they relate to the user's query.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"RAG Generation failed: {e}")
            return "Failed to generate summary."
