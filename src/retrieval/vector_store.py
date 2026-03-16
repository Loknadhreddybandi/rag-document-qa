import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.ingest.chunker import Chunk
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")


class VectorStore:
    def __init__(self):
        print(f"Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dimension = 384
        self.index = None
        self.metadata: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        print(f"Embedding {len(chunks)} chunks...")

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self.metadata = [
            {
                "text": c.text,
                "source": c.source,
                "page_num": c.page_num,
                "chunk_id": c.chunk_id
            }
            for c in chunks
        ]

        print(f"FAISS index built: {self.index.ntotal} vectors stored")

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to existing index without resetting it."""
        print(f"Adding {len(chunks)} chunks to existing index...")

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)

        self.metadata.extend([
            {
                "text": c.text,
                "source": c.source,
                "page_num": c.page_num,
                "chunk_id": c.chunk_id
            }
            for c in chunks
        ])

        print(f"Index now has {self.index.ntotal} vectors total")

    def save(self, path: str = FAISS_INDEX_PATH) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(self.metadata, f)
        print(f"Index saved to {path}/")

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        index_path = f"{path}/index.faiss"
        meta_path = f"{path}/metadata.json"

        if not Path(index_path).exists():
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)

        print(f"Loaded existing index: {self.index.ntotal} vectors")
        return True

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Index not built. Upload a PDF first.")

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0
