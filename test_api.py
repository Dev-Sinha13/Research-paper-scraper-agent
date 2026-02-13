"""Test Semantic Scholar API directly to check if rate limit has cooled down."""
import time
import sys
sys.path.append(".")

from semanticscholar import SemanticScholar

print("Testing Semantic Scholar API...")
print("=" * 50)

sch = SemanticScholar(timeout=20)

try:
    print("Searching for: 'attention mechanism'")
    results = sch.search_paper("attention mechanism", limit=3)
    
    count = 0
    for paper in results:
        count += 1
        print(f"\n  Paper {count}:")
        print(f"    Title: {paper.title}")
        print(f"    Year: {paper.year}")
        print(f"    ID: {paper.paperId}")
    
    if count == 0:
        print("\n  No results returned. API may still be rate-limited.")
    else:
        print(f"\n  SUCCESS: Found {count} papers. API is working!")
        
except Exception as e:
    print(f"\n  ERROR: {e}")
    if "429" in str(e):
        print("  The API is still rate-limited. Wait a few more minutes.")
    else:
        print("  This is a different error. Check your network connection.")
