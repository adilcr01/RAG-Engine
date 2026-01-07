import os
import shutil
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ingest import ingest_pdf
from chat import RAGChat

app = FastAPI(title="RAG-Engine API")

# Setup directories
UPLOAD_DIR = "uploads"
INDEX_NAME = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Chat System
chat_system = RAGChat(index_name=INDEX_NAME)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger ingestion
        num_chunks = ingest_pdf(file_path, index_name=INDEX_NAME)
        
        # Re-initialize chat system to load new index
        chat_system._initialize()
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Successfully ingested {file.filename}. Split into {num_chunks} chunks.",
            "filename": file.filename
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/delete")
async def delete_files():
    try:
        # Clear uploads directory
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        
        # Clear FAISS index
        if os.path.exists(INDEX_NAME):
            shutil.rmtree(INDEX_NAME)
        
        # Re-initialize chat system to clear state
        chat_system._initialize()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Successfully removed document and cleared cache."
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/status")
async def get_status():
    try:
        files = os.listdir(UPLOAD_DIR)
        if files:
            return JSONResponse(content={"status": "synced", "filename": files[0]})
        return JSONResponse(content={"status": "empty"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat_endpoint(query: str = Form(...)):
    try:
        response = chat_system.ask(query)
        # Ensure response has the correct format
        if "error" in response:
            return JSONResponse(content={"answer": response.get("error", "An error occurred"), "sources": []}, status_code=500)
        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"answer": f"Server error: {str(e)}", "sources": []}, status_code=500)

# New endpoint for fullstack integration - accepts document text directly
from pydantic import BaseModel
# import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    document_text: str
    question: str

@app.post("/api/chat")
async def api_chat_endpoint(request: ChatRequest):
    """Chat endpoint for fullstack integration - uses provided document context"""
    try:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        from langchain_core.prompts import PromptTemplate
        
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
             return JSONResponse(content={"error": "HUGGINGFACEHUB_API_TOKEN not configured", "answer": "Server Error: API Key missing"}, status_code=500)

        endpoint = HuggingFaceEndpoint(
            repo_id="google/gemma-2-9b-it",
            huggingfacehub_api_token=api_key,
            temperature=0.3,
            max_new_tokens=512,
            task="text-generation"
        )
        llm = ChatHuggingFace(llm=endpoint)
        
        prompt_template = """You are a helpful legal document assistant. Answer the user's question based ONLY on the provided document context. If the answer cannot be found in the document, say so clearly.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

Please provide a helpful, accurate answer based on the document above. Include specific references to relevant parts of the document when applicable."""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm
        
        response = chain.invoke({"context": request.document_text, "question": request.question})
        
        # ChatHuggingFace returns an AIMessage object, we need the content
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return JSONResponse(content={
            "answer": answer,
            "status": "success"
        })
    except Exception as e:
        print(f"Error in api_chat_endpoint: {e}")
        return JSONResponse(content={"error": str(e), "answer": f"AI Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
