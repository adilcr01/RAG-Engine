import os
import shutil
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

# Load environment variables
load_dotenv(r"..\.env")

# Check API Key
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

class PDFWithOCRLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ocr = RapidOCR()

    def load(self) -> List[Document]:
        documents = []
        try:
            doc = fitz.open(self.file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                
                # If text is minimal, try OCR
                if len(text) < 50:
                    print(f"Page {page_num + 1} seems to be an image. Attempting OCR...")
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # RapidOCR expects file path or bytes
                    ocr_result, _ = self.ocr(img_data)
                    
                    if ocr_result:
                        # ocr_result is a list of [box, text, score]
                        text = "\n".join([res[1] for res in ocr_result])
                        print(f"OCR extracted {len(text)} characters.")
                
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": self.file_path, "page": page_num}
                    ))
            
            doc.close()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            
        return documents

def ingest_pdf(pdf_path: str, index_name: str = "faiss_index", clear_existing: bool = True):
    print(f"Loading PDF from {pdf_path}...")
    
    loader = PDFWithOCRLoader(pdf_path)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No text could be extracted from the PDF. The file might be corrupted or empty.")

    print(f"Loaded {len(documents)} pages.")
    
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        raise ValueError("Document was loaded but resulted in 0 text chunks.")

    print(f"Split into {len(texts)} chunks.")
    
    print("Creating embeddings and vector store...")
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Clear existing index if requested
        if clear_existing and os.path.exists(index_name):
            print(f"Clearing existing index at '{index_name}'...")
            shutil.rmtree(index_name)
            
        # Check if index already exists (after potential clearing)
        if os.path.exists(index_name):
            print(f"Loading existing index from '{index_name}'...")
            vectorstore = FAISS.load_local(
                index_name,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Adding {len(texts)} new chunks to existing index...")
            vectorstore.add_documents(texts)
        else:
            print(f"Creating new index '{index_name}'...")
            vectorstore = FAISS.from_documents(
                documents=texts, 
                embedding=embeddings
            )
        
        vectorstore.save_local(index_name)
        print(f"Ingestion complete. FAISS index saved to '{index_name}'.")
        return len(texts)
    except Exception as e:
        print(f"Error during embedding/vector store creation: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to alert the caller (UI)
        raise e

if __name__ == "__main__":
    # Test with a dummy file if needed
    pass
