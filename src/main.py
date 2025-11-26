from fastapi import FastAPI
from src.milvus.routes import milvus_router
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# FastAPI bootstrap
# ------------------------------------------------------------------------------

app = FastAPI(title="PDF Milvus RAG API", version="1.0.0")

version_prefix =f"/api"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(milvus_router, prefix=f"{version_prefix}/milvus", tags=["milvus"])