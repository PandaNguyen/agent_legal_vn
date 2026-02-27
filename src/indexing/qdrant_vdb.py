from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVectorParams, SparseIndexParams, PointStruct,
    VectorParams, Distance, SparseVector,
    Prefetch, FusionQuery, Fusion
)
from fastembed import SparseTextEmbedding
from config.settings import settings
from src.embeddings.embed_data import Embeddata, batch_iterate
from src.indexing.corpus_loader import ChunkDocument


class QdrantVDB:
    """
    Vector database wrapper for Qdrant.

    Responsibilities
    ----------------
    - Manage the Qdrant client lifecycle
    - Create / drop the hybrid collection (dense + sparse)
    - Ingest List[ChunkDocument] → embed → upsert with rich payload
    - Execute hybrid search (dense + sparse BM25 via RRF fusion)
    """

    def __init__(
        self,
        collection_name: str = None,
        vector_dim: int = None,
        batch_size: int = None,
    ):
        self.collection_name = collection_name or settings.collection_name
        self.vector_dim = vector_dim or settings.vector_dim
        self.batch_size = batch_size or settings.batch_size
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_embedding_model)
        self.client: QdrantClient = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def initialize_client(self):
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60,          # seconds – tránh WriteTimeout với payload lớn
            )
            logger.info(f"Initialized Qdrant client with URL: {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed Qdrant client connection")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self):
        """Create a fresh hybrid collection, dropping any existing one."""
        self._check_client()
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        logger.info(f"Created collection: {self.collection_name}")

    def collection_exists(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def get_collection_info(self) -> dict:
        self._check_client()
        if not self.collection_exists():
            return {"exists": False}
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "collection_name": self.collection_name,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # Ingestion — accepts List[ChunkDocument] directly
    # ------------------------------------------------------------------

    def ingest_data(self, chunks: List[ChunkDocument], embed_data: Embeddata):
        """
        Embed and upsert a list of ChunkDocuments into Qdrant.

        Parameters
        ----------
        chunks      : list of ChunkDocument (produced by CorpusLoader)
        embed_data  : Embeddata instance (holds the dense embedding model)
        """
        self._check_client()
        logger.info(f"Ingesting {len(chunks)} chunks into '{self.collection_name}'")

        UPSERT_BATCH = 4          # nhỏ để tránh WriteTimeout trên Qdrant Cloud
        texts = [c.text for c in chunks]
        total_inserted = 0

        # Embed toàn bộ theo embed batch_size (local, nhanh)
        logger.info("Generating dense embeddings...")
        all_dense = embed_data.generate_embeddings(texts)

        logger.info("Generating sparse (BM25) embeddings...")
        all_sparse = list(self.sparse_model.embed(texts))

        # Upsert theo batch nhỏ hơn lên cloud
        for batch_start in range(0, len(chunks), UPSERT_BATCH):
            batch_end = min(batch_start + UPSERT_BATCH, len(chunks))
            points = []
            for i in range(batch_start, batch_end):
                chunk = chunks[i]
                dense_vec = all_dense[i]
                sparse_vec = all_sparse[i]
                points.append(
                    PointStruct(
                        id=i,
                        vector={
                            "dense": (
                                dense_vec.tolist()
                                if hasattr(dense_vec, "tolist")
                                else list(dense_vec)
                            ),
                            "sparse": SparseVector(
                                indices=sparse_vec.indices.tolist(),
                                values=sparse_vec.values.tolist(),
                            ),
                        },
                        payload=chunk.to_payload(),
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            total_inserted += len(points)
            logger.info(f"Upserted [{batch_end}/{len(chunks)}]")

        logger.info(f"Ingestion complete. Total points: {total_inserted}")

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def search(self, query: str, embed_data: Embeddata, top_k: int = None) -> List[dict]:
        """
        Hybrid search: dense (semantic) + sparse (BM25) fused via RRF.

        Returns
        -------
        List of dicts with the full payload plus a `score` field.
        """
        self._check_client()
        top_k = top_k or settings.top_k

        dense_query = embed_data.get_query_embedding(query)
        sparse_query = list(self.sparse_model.embed([query]))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_query, using="dense", limit=top_k * 2),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {"score": hit.score, **hit.payload}
            for hit in results.points
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_client(self):
        if not self.client:
            raise RuntimeError("Qdrant client is not initialized. Call initialize_client() first.")