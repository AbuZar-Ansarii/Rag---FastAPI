from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import traceback
from app.rag import rag_system
from app.models import QueryRequest, QueryResponse, UploadResponse
from app.config import settings

app = FastAPI(title="Ayurveda RAG System API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Ensure required directories exist
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.PERSIST_DIRECTORY, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Ayurveda RAG System API is running beautifuly."}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(file: UploadFile = File(...)):
    # Only allow PDF files
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        # Process document
        chunks_created = rag_system.upload_documents(file_path)
        return UploadResponse(
            message="Documents uploaded and processed successfully.",
            document_count=1,
            chunks_created=chunks_created
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if not rag_system.vector_db:
            raise HTTPException(
                status_code=400,
                detail="No documents loaded. Please upload documents first."
            )
        result = rag_system.query(request.query)
        return QueryResponse(
            query =result["query"],
            answer =result["answer"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "system": "RAG with Gemini"}