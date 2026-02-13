# Research Paper Scraper Agent 🧠📚

An AI-powered agentic system that explores research papers, builds knowledge graphs, and helps you discover relevant academic literature.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 🚀 Features

- **Automated Research**: Scrapes Semantic Scholar and Arxiv to find papers relevant to your query.
- **Knowledge Graph Construction**: Visualizes connections between papers based on citations and relevance.
- **Relevance Scoring**: Uses embeddings (FastEmbed) to score papers based on similarity to your research topic.
- **Interactive UI**: Explore the graph and paper details through a modern Streamlit interface.
- **Rate Limit Handling**: Built-in mechanisms to handle API rate limits gracefully.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Dev-Sinha13/Research-paper-scraper-agent.git
    cd Research-paper-scraper-agent
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Setup**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    # Optional: Semantic Scholar API Key if you have one
    # S2_API_KEY=your_key_here 
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run src/app.py
```

1.  Open your browser to the local URL provided (usually `http://localhost:8501`).
2.  Enter a research abstract or topic in the input box.
3.  Adjust parameters (Depth, Relevance Threshold) in the sidebar.
4.  Click **Generate Graph** to start the agent.

## 📂 Project Structure

-   `src/app.py`: Main Streamlit application entry point.
-   `src/agent.py`: Core LangGraph agent logic.
-   `src/fetcher.py`: Modules for fetching papers from APIs.
-   `src/models.py`: Pydantic models for data structures.
-   `src/config.py`: Configuration settings.
-   `data/`: Directory for caching results and embeddings.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

[MIT](LICENSE)