import sys
try:
    import langgraph
    import sentence_transformers
    import semanticscholar
    import networkx
    import streamlit
    print("SUCCESS: All dependencies imported.")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
