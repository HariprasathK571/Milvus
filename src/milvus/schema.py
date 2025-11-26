from pydantic import BaseModel
from typing import List


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    distance: float
    source: str
    page: int
    total_page: int
    content_preview: str


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


class IngestionResponse(BaseModel):
    file_name: str
    chunks_inserted: int