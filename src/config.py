
import os

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

DB_NAME = os.getenv("MILVUS_DB_NAME", "DemoDB")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "Report")
EMBEDDING_DIM = 1024
HASH_CHUNK_SIZE = 8192
MEDIA_DIR = "./milvus/media"