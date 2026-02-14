import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import sys
import os
import logging

# Add project root to path so we can import packages from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import ResearchGraph
from src.config import MAX_DEPTH

# Enable logging to see what's happening in the terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

st.set_page_config(page_title="Deep Research Agent", layout="wide")

# Cache the agent so it only initializes once
@st.cache_resource
def get_agent():
    return ResearchGraph()

st.title("Papers -> Knowledge Graph")
st.markdown("Enter an abstract or topic to discover relevant research pathways.")

# Sidebar Config
st.sidebar.header("Configuration")
max_depth = st.sidebar.slider("Depth", 1, 3, 2)
similarity_threshold = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, 0.5)
max_duration = st.sidebar.slider("Max Search Time (s)", 30, 300, 120)

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
                
                st.write("🔍 Searching for seed papers via Semantic Scholar...")
                final_state = research_graph.compile().invoke(initial_state)
                st.write("✅ Research complete!")
                status.update(label="Research Complete!", state="complete", expanded=False)
            except RuntimeError as e:
                status.update(label="Error!", state="error")
                st.error(f"Agent error: {e}")
                st.info(
                    "**Troubleshooting tips:**\n"
                    "- If you see a rate-limit (429) error, wait 5 minutes and try again.\n"
                    "- If the search returned 0 results, try a broader topic.\n"
                    "- Check the terminal/console for detailed logs."
                )
                st.stop()
            except Exception as e:
                status.update(label="Error!", state="error")
                st.error(f"Unexpected error: {e}")
                st.stop()
            
            papers = final_state.get('papers', {})

            # Filter by threshold if desired
            relevant_papers = {
                pid: p for pid, p in papers.items()
                if p.get('relevance_score', 0) >= similarity_threshold
            }

            st.success(
                f"Found **{len(papers)}** total papers, "
                f"**{len(relevant_papers)}** above relevance threshold ({similarity_threshold:.2f})."
            )

            # Show summary if available
            summary = final_state.get('summary', '')
            if summary and summary != "No papers found to summarize.":
                st.subheader("🧠 AI Summary")
                st.write(summary)
            
            # Visualize
            display_papers = relevant_papers if relevant_papers else papers
            if display_papers:
                # Build NetworkX Graph
                G = nx.DiGraph()
                
                for pid, paper in display_papers.items():
                    label = f"{paper.get('year', '?')} - {paper['title'][:40]}..."
                    score = paper.get('relevance_score', 0)

                    # Color based on relevance score
                    if score > 0.7:
                        color = "#4CAF50"  # Green — high relevance
                    elif score > 0.4:
                        color = "#FFC107"  # Amber — moderate
                    else:
                        color = "#9E9E9E"  # Grey — low

                    G.add_node(
                        pid,
                        label=label,
                        title=f"{paper['title']}\nScore: {score:.2f}",
                        color=color,
                        size=15 + int(score * 25),
                    )

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
                st.subheader("📄 Paper Details")
                for pid, paper in sorted(
                    display_papers.items(),
                    key=lambda x: x[1].get('relevance_score', 0),
                    reverse=True
                ):
                    score = paper.get('relevance_score', 0)
                    abstract = paper.get('abstract', '') or '*No abstract available*'
                    with st.expander(
                        f"{'🟢' if score > 0.7 else '🟡' if score > 0.4 else '⚪'} "
                        f"{paper['title']} ({paper.get('year', '?')}) — Score: {score:.2f}"
                    ):
                        st.write(abstract)
                        if paper.get('url'):
                            st.markdown(f"[Open on Semantic Scholar]({paper['url']})")
            else:
                st.warning(
                    "No papers met the relevance threshold. "
                    "Try lowering the threshold in the sidebar or using a broader query."
                )
