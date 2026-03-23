"""Configuration for RAG chatbot."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")

# RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 4

# Embedding (sentence-transformers for local; or use OpenAIEmbeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM (set OPENAI_API_KEY for OpenAI)
USE_OPENAI = os.getenv("OPENAI_API_KEY", "").strip() != ""
