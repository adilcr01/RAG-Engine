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

@app.post("/chat")
async def chat_endpoint(query: str = Form(...)):
    try:
        response = chat_system.ask(query)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
