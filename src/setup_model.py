from fastembed import TextEmbedding
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    model_name = "BAAI/bge-small-en-v1.5"
    logger.info(f"Starting download for: {model_name}")
    start = time.time()
    
    try:
        # This triggers the download
        model = TextEmbedding(model_name=model_name)
        # Test it
        vec = list(model.embed(["Hello world"]))
        
        duration = time.time() - start
        logger.info(f"Model downloaded and verified in {duration:.2f} seconds.")
        logger.info("You can now restart the Streamlit app.")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()
