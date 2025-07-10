from pydantic import BaseModel, Field, root_validator
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query : str

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query : str
    answer: str

class UploadResponse(BaseModel):
    message: str
    document_count: int