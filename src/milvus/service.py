from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pymilvus import (
    connections,
    db,
    utility,
    FieldSchema,
    DataType,
    Collection,
    CollectionSchema,
)
from .schema import RetrievedChunk,RetrievalResponse
from .config import COLLECTION_NAME,DB_NAME,EMBEDDING_DIM,MILVUS_HOST,MILVUS_PORT



# ------------------------------------------------------------------------------
# Global resources (model + Milvus connection + collection)
# ------------------------------------------------------------------------------

sentence_model: SentenceTransformer | None = None
collection: Collection | None = None


def init_sentence_model() -> SentenceTransformer:
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
        )
    return sentence_model


def init_milvus() -> Collection:
    """
    Connect to Milvus, ensure DB and collection exist, create index & load.
    """
    global collection

    if collection is not None:
        return collection

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Ensure database
    if DB_NAME not in db.list_database():
        db.create_database(DB_NAME)
    db.using_database(DB_NAME)

    # Create collection if needed
    if COLLECTION_NAME not in utility.list_collections():
        id_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        source_field = FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=255,
        )
        page_field = FieldSchema(
            name="Page",
            dtype=DataType.INT64,
        )
        total_page_field = FieldSchema(
            name="Total_Page",
            dtype=DataType.INT64,
        )
        embedding_field = FieldSchema(
            name="embeddings",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        )
        content_field = FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=2048,
        )

        schema = CollectionSchema(
            fields=[
                id_field,
                source_field,
                page_field,
                total_page_field,
                embedding_field,
                content_field,
            ],
            description="PDF chunks with embeddings",
        )
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Create HNSW index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            },
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
    else:
        collection = Collection(name=COLLECTION_NAME)

    # Load collection into memory
    collection.load()
    return collection
 

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def ingest_pdf_to_milvus(temp_pdf_path: str, original_filename: str) -> int:
    """
    Load PDF, split into chunks, embed, and insert into Milvus.
    Returns the number of chunks inserted.
    """
    coll = init_milvus()
    model = init_sentence_model()

    # Load PDF using PyPDFLoader
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    # Normalize metadata types if needed
    for doc in docs:
        if "page" in doc.metadata:
            try:
                doc.metadata["page"] = int(doc.metadata["page"])
            except Exception:
                doc.metadata["page"] = 0
        if "page_label" in doc.metadata:
            try:
                doc.metadata["page_label"] = int(doc.metadata["page_label"])
            except Exception:
                doc.metadata["page_label"] = 0

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        return 0

    # Prepare fields
    sources = [original_filename for _ in chunks]
    pages = [int(chunk.metadata.get("page_label", 0)) for chunk in chunks]
    total_pages = [int(chunk.metadata.get("total_pages", 0)) for chunk in chunks]
    contents = [chunk.page_content[:2048] for chunk in chunks]

    # Compute embeddings
    embeddings = model.encode(contents, convert_to_numpy=True).tolist()

    if len(embeddings[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension {len(embeddings[0])} does not match configured dim {EMBEDDING_DIM}"
        )

    # Milvus expects data column-wise, in field order (excluding auto_id primary)
    insert_data = [
        sources,
        pages,
        total_pages,
        embeddings,
        contents,
    ]

    # Map to field names except auto_id primary ("id")
    coll.insert(insert_data)
    coll.flush()

    return len(chunks)


def retrieve_from_milvus(query: str, top_k: int = 5) -> RetrievalResponse:
    coll = init_milvus()
    model = init_sentence_model()

    query_embedding = model.encode([query])

    search_params = {
        "metric_type": "L2",
        "params": {"ef": 64},
    }

    results = coll.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        output_fields=["source", "Page", "Total_Page", "content"],
    )

    hits = results[0]
    resp_results: List[RetrievedChunk] = []
    for hit in hits:
        entity = hit.entity
        content = entity.get("content", "")
        preview = content[:400]  # truncate for response
        resp_results.append(
            RetrievedChunk(
                distance=float(hit.distance),
                source=str(entity.get("source", "")),
                page=int(entity.get("Page", 0)),
                total_page=int(entity.get("Total_Page", 0)),
                content_preview=preview,
            )
        )

    return RetrievalResponse(query=query, results=resp_results)

# ------------------------------------------------------------------------------
# To run:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# ------------------------------------------------------------------------------

