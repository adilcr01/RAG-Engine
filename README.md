# RAG-Engine

**RAG-Engine** is a powerful Retrieval-Augmented Generation system designed to transform static documents into interactive knowledge bases. By combining advanced vector search with state-of-the-art Large Language Models (LLMs), it allows users to have natural conversations with their data while obtaining precise, cited answers.

## 🚀 Key Features

*   **Intelligent Knowledge Injection**: Automatically splits and embeds PDF documents to provide the LLM with relevant context for ogni query.
*   **Semantic Retrieval**: Leverages **FAISS** (Facebook AI Similarity Search) for high-performance vector matching, ensuring the most relevant document sections are found instantly.
*   **Source Citation**: Every response includes specific page references from the source document, ensuring transparency and trust.
*   **Hybrid Architecture**: Combines local high-speed vector storage with cloud-based LLM inference via the **Hugging Face Inference API**.
*   **Plug-and-Play**: Optimized for Python 3.12 with minimal dependencies and robust error handling.

## 🛠️ Technology Stack

*   **LLM Integration**: LangChain & Hugging Face Endpoint
*   **Vector Database**: FAISS (Local)
*   **Embeddings**: Hugging Face (sentence-transformers)
*   **Document Processing**: PyPDF
*   **Orchestration**: LangChain Expression Language (LCEL)

## 📖 How It Works

1.  **Ingest**: `ingest.py` processes the PDF, creates semantic chunks, and builds a local FAISS index.
2.  **Retrieve**: When a user asks a question, the engine retrieves the top relevant context from the FAISS database.
3.  **Generate**: The engine pushes the retrieved knowledge along with the question to the LLM to generate a precise response.
