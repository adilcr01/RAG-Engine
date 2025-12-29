import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (API Key)
load_dotenv(r"..\.env")

# Check API Key
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

def ingest_pdf(pdf_path: str, index_name: str = "faiss_index"):
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(
        documents=texts, 
        embedding=embeddings
    )
    vectorstore.save_local(index_name)
    print(f"Ingestion complete. FAISS index saved to '{index_name}'.")
    return len(texts)

if __name__ == "__main__":
    default_pdf = r"..\Copy of Spring Hill - Nelly's Italian Cafe - Lease (Fully Executed) 01.17.14.pdf"
    ingest_pdf(default_pdf)
