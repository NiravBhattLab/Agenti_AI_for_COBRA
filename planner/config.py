import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(_PROJECT_ROOT, "gsm_procedural_knowledge_base"),
)
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "gsm_workflows_v2")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
