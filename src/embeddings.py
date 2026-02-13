from fastembed import TextEmbedding
import numpy as np
import logging
from .config import EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            try:
                # FastEmbed uses a different model name format, but handles mapping internally
                model_name = "BAAI/bge-small-en-v1.5" # Very fast and good semantic search
                logger.info(f"Loading embedding model: {model_name}...")
                cls._instance.model = TextEmbedding(model_name=model_name)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                cls._instance.model = None
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text."""
        if not text or not self.model:
            return []
        
        # fastembed returns a generator, convert to list
        # It handles batch processing efficiently
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        
        # Use numpy for fast cosine similarity
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
