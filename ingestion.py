"""Document ingestion: chunk, embed, upsert to Qdrant. Synthetic docs only."""
import glob
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return [c for c in chunks if c.strip()]


def load_documents(data_dir: str = "data/docs") -> list[dict]:
    docs = []
    for path in glob.glob(f"{data_dir}/**/*.md", recursive=True):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": path, "text": f.read()})
    return docs


def ingest(data_dir: str = "data/docs") -> int:
    client = QdrantClient(url=QDRANT_URL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for doc in load_documents(data_dir):
        for chunk in chunk_text(doc["text"]):
            vec = embedder.encode(chunk).tolist()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"content": chunk, "source": doc["source"]},
            ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


if __name__ == "__main__":
    n = ingest()
    print(f"ingested {n} chunks")
