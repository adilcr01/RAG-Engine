# RAG-Engine

**RAG-Engine** is a powerful Retrieval-Augmented Generation system designed to transform static documents into interactive knowledge bases. By combining advanced vector search with state-of-the-art Large Language Models (LLMs), it allows users to have natural conversations with their data while obtaining precise, cited answers.

![RAG-Engine Frontend](/C:/Users/adila/.gemini/antigravity/brain/60efeb97-bc5a-4cc2-a665-831e311e1215/rag_engine_frontend_1766987192880.png)

## 🚀 Key Features

* **Intelligent Knowledge Injection**: Automatically splits and embeds PDF documents to provide the LLM with relevant context for ogni query.
* **Semantic Retrieval**: Leverages **FAISS** (Facebook AI Similarity Search) for high-performance vector matching, ensuring the most relevant document sections are found instantly.
* **Source Citation**: Every response includes specific page references from the source document, ensuring transparency and trust.
* **Hybrid Architecture**: Combines local high-speed vector storage with cloud-based LLM inference via the **Hugging Face Inference API**.

## ⚙️ Configuration

To run this project, you need a Hugging Face API Token.

1. Create a `.env` file in the **parent directory** of `ai_chat_project`.
2. Add your token:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

### Default Models
* **LLM**: `google/gemma-2-9b-it` (Optimized for text-generation)
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`

## 🛠️ Technology Stack

* **Backend**: FastAPI (Python)
* **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (ES6+)
* **LLM Integration**: LangChain & Hugging Face Endpoint
* **Vector Database**: FAISS (Local)
* **Embeddings**: Hugging Face (sentence-transformers)
* **Document Processing**: PyPDF

## 🏃 Local Setup

1. **Activate Environment**:
   ```powershell
   ..\ai_env\Scripts\activate
   ```
2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```powershell
   python main.py
   ```
4. **Access**: Open [http://localhost:8000](http://localhost:8000)

## 📖 How It Works

1. **Ingest**: `ingest.py` processes the PDF, creates semantic chunks, and builds a local FAISS index.
2. **Retrieve**: When a user asks a question, the engine retrieves the top relevant context from the FAISS database.
3. **Generate**: The engine pushes the retrieved knowledge along with the question to the LLM to generate a precise response.
