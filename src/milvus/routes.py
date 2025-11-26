import os
import uuid
import hashlib
from fastapi import UploadFile, File, HTTPException,APIRouter,Depends
from .schema import IngestionResponse,RetrievalRequest,RetrievalResponse
from .service import ingest_pdf_to_milvus,retrieve_from_milvus,init_milvus,init_sentence_model
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.core import get_session,init_db
from sqlalchemy.exc import IntegrityError  
from sqlmodel import desc, select
from src.db.models import UploadedFile


milvus_router = APIRouter()

@milvus_router.on_event("startup")
async def on_startup():  # ← Make it async
    """
    Initialize global resources on app startup.
    """
    print("🚀 Initializing Milvus resources...")
    init_sentence_model()
    init_milvus()
    
    print("📦 Creating database tables...")
    await init_db()  # ← Add this line
    print("✅ Database tables created")

HASH_CHUNK_SIZE = 8192
MEDIA_DIR = "./milvus/media"


async def compute_file_hash_stream(file: UploadFile, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    sha256 = hashlib.sha256()
    await file.seek(0)
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        sha256.update(chunk)
    await file.seek(0)
    return sha256.hexdigest()


#--------------------------------------------------------------------------
#    file upload by storing and checking the hash on the PostgresDB
#---------------------------------------------------------------------------

@milvus_router.post("/ingest", response_model=IngestionResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Compute hash
    file_hash = await compute_file_hash_stream(file)

    # Check for existing hash in DB
    stmt = select(UploadedFile).where(UploadedFile.file_hash == file_hash)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="This file is already uploaded.")

    # Ensure media directory exists
    os.makedirs(MEDIA_DIR, exist_ok=True)

    base_name = file.filename.rsplit(".", 1)[0]
    suffix = f"_{uuid.uuid4().hex}.pdf"
    stored_filename = base_name + suffix
    saved_path = os.path.join(MEDIA_DIR, stored_filename)

    # Persist file to disk
    try:
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)
        await file.seek(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Ingest into Milvus
    try:
        chunk_count = ingest_pdf_to_milvus(saved_path, file.filename)
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if chunk_count == 0:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=400, detail="No text chunks were extracted from PDF.")

    # Store hash record in DB (with transaction)
    new_record = UploadedFile(
        file_hash=file_hash,
        original_filename=file.filename,
        stored_filename=stored_filename,
    )
    db.add(new_record)
    try:
        await db.commit()
    except IntegrityError:
        # Hash already inserted concurrently by another request
        await db.rollback()
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=400, detail="This file is already uploaded.")
    except Exception as e:
        await db.rollback()
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail=f"Failed to record file metadata: {e}")

    return IngestionResponse(file_name=file.filename, chunks_inserted=chunk_count)





#--------------------------------------------------------------------------
#    file upload by storing and checking the hash on the Json file
#---------------------------------------------------------------------------
# Assume you uploaded a PDF named "document.pdf" (20 MB in size). Internally, this function:
# Reads the first 8 KB and updates hash.
# Reads the second 8 KB and updates hash.
# Repeats until entire 20 MB is processed.
# Computes the final SHA-256 hash for the entire file content.
# This hash uniquely represents the file content and can be used to detect duplicates or verify integrity.

# HASH_RECORD_FILE = "./milvus/media"

# def load_hash_records():
#     if os.path.exists(HASH_RECORD_FILE):
#         try:
#             with open(HASH_RECORD_FILE, "r") as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, OSError):
#             # File is empty or invalid, treat as empty dict
#             return {}
#     return {}

# def save_hash_records(records):
#     with open(HASH_RECORD_FILE, "w") as f:
#         json.dump(records, f)

# async def compute_file_hash_stream(file: UploadFile, chunk_size: int = 8192) -> str:
#     sha256 = hashlib.sha256()
#     await file.seek(0)  # Reset pointer to start
#     while True:
#         chunk = await file.read(chunk_size)
#         if not chunk:
#             break
#         sha256.update(chunk)
#     await file.seek(0)  # Reset pointer again after reading
#     return sha256.hexdigest()

# @milvus_router.post("/ingest", response_model=IngestionResponse)
# async def ingest_endpoint(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     # Compute SHA-256 hash of file content via streaming
#     file_hash = await compute_file_hash_stream(file)

#     hash_records = load_hash_records()
#     if file_hash in hash_records:
#         raise HTTPException(status_code=400, detail="This file is already uploaded.")

#     # Ensure media directory exists
#     media_dir = "./media"
#     os.makedirs(media_dir, exist_ok=True)

#     # Save file to media dir with unique filename
#     suffix = f"_{uuid.uuid4().hex}.pdf"
#     base_name = file.filename.rsplit('.', 1)[0]
#     saved_filename = base_name + suffix
#     saved_path = os.path.join(media_dir, saved_filename)

#     try:
#         content = await file.read()
#         with open(saved_path, "wb") as f:
#             f.write(content)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

#     # Ingest PDF to Milvus - replace this with your actual ingestion logic
#     try:
#         chunk_count = ingest_pdf_to_milvus(saved_path, file.filename)
#     except Exception as e:
#         os.remove(saved_path)  # cleanup on failure
#         raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

#     if chunk_count == 0:
#         os.remove(saved_path)
#         raise HTTPException(status_code=400, detail="No text chunks were extracted from PDF.")

#     # Record file hash to prevent duplicates
#     hash_records[file_hash] = saved_filename
#     save_hash_records(hash_records)

#     return IngestionResponse(file_name=file.filename, chunks_inserted=chunk_count)

#-----------------------------------------------------------------------to upload the pdf Data into the milvus without saving it on the local directory-----------------------------------
# @milvus_router.post("/ingest", response_model=IngestionResponse)
# async def ingest_endpoint(file: UploadFile = File(...)):
#     """
#     Ingest a single PDF file into Milvus.
#     """
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     # Save to a temporary file
#     try:
#         suffix = f"_{uuid.uuid4().hex}.pdf"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             content = await file.read()
#             tmp.write(content)
#             temp_path = tmp.name
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

#     try:
#         chunk_count = ingest_pdf_to_milvus(temp_path, file.filename)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
#     finally:
#         try:
#             os.remove(temp_path)
#         except Exception:
#             pass

#     if chunk_count == 0:
#         raise HTTPException(status_code=400, detail="No text chunks were extracted from PDF.")

#     return IngestionResponse(file_name=file.filename, chunks_inserted=chunk_count)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@milvus_router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(request: RetrievalRequest):
    """
    Retrieve top-k relevant chunks for a text query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        response = retrieve_from_milvus(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    return response

