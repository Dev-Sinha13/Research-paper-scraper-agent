import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import sys
import os

# Add project root to path so we can import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import ResearchGraph
from src.config import MAX_DEPTH

st.set_page_config(page_title="Deep Research Agent", layout="wide")

# Cache the agent so it only initializes once
@st.cache_resource
def get_agent():
    return ResearchGraph()

st.title("Papers -> Knowledge Graph")
st.markdown("Enter an abstract to discover relevant research pathways.")

# Sidebar Config
st.sidebar.header("Configuration")
max_depth = st.sidebar.slider("Depth", 1, 3, 2)
similarity_threshold = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, 0.5)
max_duration = st.sidebar.slider("Max Search Time (s)", 30, 300, 60)

# Input
query = st.text_area("Research Abstract / Topic", height=150)

if st.button("Generate Graph"):
    if not query:
        st.warning("Please enter an abstract.")
    else:
        with st.status("Running Research Agent...", expanded=True) as status:
            try:
                st.write("⏳ Loading AI models...")
                research_graph = get_agent()
                st.write("✅ Models loaded!")
                
                # Initial State
                initial_state = {
                    "query": query,
                    "current_depth": 0,
                    "visited_ids": set(),
                    "queue": [],
                    "papers": {},
                    "max_duration": max_duration
                }
                
                st.write("🔍 Searching for seed papers...")
                final_state = research_graph.compile().invoke(initial_state)
                st.write("✅ Research complete!")
                status.update(label="Research Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error!", state="error")
                st.error(f"Agent failed: {e}")
                st.info("This is often caused by Semantic Scholar API rate limits (429). Wait 5 minutes and try again.")
                st.stop()
            
            papers = final_state.get('papers', {})
            st.success(f"Found {len(papers)} relevant papers!")
            
            # Visualize
            if papers:
                # Build NetworkX Graph
                G = nx.DiGraph()
                
                for pid, paper in papers.items():
                    # Add Node
                    label = f"{paper['year']} - {paper['title'][:30]}..."
                    color = "#97c2fc" if paper.get('relevance_score', 0) > 0.7 else "#ffff00"
                    
                    G.add_node(pid, label=label, title=paper['title'], color=color)
                    
                    # Add Edges (Citations/References)
                    # Note: We only have edges if we fetched the neighbors.
                    # Currently ResearchState doesn't explicit store edges, 
                    # but we can infer them if both nodes exist in 'papers'.
                    # For now, just show nodes. To show edges, we need to track them.
                    pass 

                # Visualization with PyVis
                net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
                net.from_nx(G)
                net.repulsion(node_distance=100, spring_length=200)
                
                # Save and read
                path = "graph.html"
                net.save_graph(path)
                
                with open(path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=600)
                
                # Show Details
                st.subheader("Paper Details")
                for pid, paper in papers.items():
                    with st.expander(f"{paper['title']} ({paper['year']}) - Score: {paper.get('relevance_score', 0):.2f}"):
                        st.write(paper['abstract'])
                        st.write(f"Url: {paper['url']}")
