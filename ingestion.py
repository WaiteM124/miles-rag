"""Document ingestion: chunk, embed, upsert to Qdrant. Synthetic docs only."""

from __future__ import annotations

import glob
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += size - overlap
    return [c for c in chunks if c.strip()]


def stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Generate a deterministic ID for a chunk based on source and content hash."""
    h = hashlib.sha256(f"{source}:{chunk_index}:{content}".encode()).hexdigest()[:32]
    return h


def load_documents(data_dir: str = "data/docs") -> list[dict]:
    docs = []
    for path in glob.glob(f"{data_dir}/**/*.md", recursive=True):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": path, "text": f.read()})
    return docs


def ingest(
    data_dir: str = "data/docs",
    client: QdrantClient | None = None,
    embedder: SentenceTransformer | None = None,
) -> int:
    """Ingest documents incrementally. Creates collection if needed, upserts chunks."""
    client = client or QdrantClient(url=QDRANT_URL)
    embedder = embedder or SentenceTransformer(EMBEDDING_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points = []
    for doc in load_documents(data_dir):
        for idx, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = stable_chunk_id(doc["source"], idx, chunk)
            vec = embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vec,
                    payload={"content": chunk, "source": doc["source"]},
                )
            )

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


if __name__ == "__main__":
    n = ingest()
    print(f"ingested {n} chunks")
