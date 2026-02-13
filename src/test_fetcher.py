from src.fetcher import ContentFetcher
import logging

logging.basicConfig(level=logging.INFO)

def test_fetcher():
    fetcher = ContentFetcher()
    
    print("Testing get_details...")
    # Test with a known paper ID (Attention is All You Need)
    paper_id = "649def34f8be52c8b66281af98ae884c09aef38b" 
    details = fetcher.get_details(paper_id)
    if details:
        print(f"Success: Found {details['title']}")
    else:
        print("Failed to get details")

    print("\nTesting get_batch_details...")
    ids = ["649def34f8be52c8b66281af98ae884c09aef38b"]
    batch = fetcher.get_batch_details(ids)
    print(f"Batch result count: {len(batch)}")

if __name__ == "__main__":
    test_fetcher()
