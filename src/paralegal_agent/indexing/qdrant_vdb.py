from loguru import logger
from typing import List, Optional
from langchain_core.documents import Document
from paralegal_agent.config import config
import torch
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVectorParams, SparseIndexParams, PointStruct,
    VectorParams, Distance, SparseVector,
    Prefetch, FusionQuery, Fusion
)
from tqdm import tqdm
from paralegal_agent.config.config import settings
from paralegal_agent.data.document_loader import load_corpus
from paralegal_agent.data.unit_merger import merge_units
from paralegal_agent.data.legal_splitter import split_law
from paralegal_agent.embeddings.embed_data import Embeddata

class JsonLoader:
    def __init__(self):
        pass
    def build_documents(self,corpus_path: str, limit: int | None = None):
        """
        Full pipeline: corpus_final.json → List[Document] ready for embedding.

        Args:
            corpus_path: Path to corpus_final.json
            limit:       If set, only process first N laws (useful for testing)

        Returns:
            List of LangChain Documents
        """
        logger.info(f"Loading corpus from: {corpus_path}")
        laws = load_corpus(corpus_path)
        if limit:
            laws = laws[:limit]
        logger.info(f"{len(laws)} laws loaded")

        all_documents = []
        for law in laws:
            merged = merge_units(law["content"])
            docs = split_law(law, merged)
            all_documents.extend(docs)

        logger.info(f"{len(all_documents)} chunks total")
        return all_documents    


class QdrantVDB:

    def __init__(self,
                 collection_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 sparse_model_name: Optional[str] = None,
                 file_data_path: Optional[str] = None,
                 batch_size: Optional[int] = None):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.batch_size = batch_size or settings.batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = embedding_model_name or settings.embeddings_model
        
        
        sparse_name = sparse_model_name or settings.sparse_embedding_model
        self.sparse_embedding = SparseTextEmbedding(model_name=sparse_name)
        
        self.client = None
        
        self.size_embedding = settings.vector_dim
        self.file_data_path = file_data_path or "data/legal_corpus.json"
        self.loader = JsonLoader()
    def load_data(self) -> List[Document]:
        documents = self.loader.build_documents(self.file_data_path)
        return documents
    def initialize_client(self) -> Optional[QdrantClient]:
        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            logger.info(f"Initialized Qdrant client with URL: {settings.qdrant_url}")
            return self.client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e
    def create_collection(self):
        if self.client is None:
            logger.info("Qdrant client is not initialized.")
            raise RuntimeError("Qdrant client initialization failed. Call initialize_client() first.")
        if self.client.collection_exists(self.collection_name):
            raise RuntimeError(f"Collection '{self.collection_name}' already exists.")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
               "dense": VectorParams(
                    size=self.size_embedding,
                    distance=Distance.COSINE
                )
            }, 
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")
    def delete_collection_if_exists(self):
        """Xóa collection cũ nếu tồn tại"""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")
        
        try:
            if self.client.collection_exists(self.collection_name):
                logger.info(f"Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
                logger.info("Collection deleted")

            else:
                logger.info(f"Collection '{self.collection_name}' does not exist")
        except Exception as e:
            logger.warning(f"Warning: Could not check/delete collection: {e}")
    def embedd_and_store(self, documents, embed_data: Embeddata):
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized")
        
        points = []
        total_uploaded = 0
        
        with tqdm(total=len(documents), desc="Processing documents") as pbar:
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i+self.batch_size]
                texts = [d.page_content for d in batch]
                
                # SỬA PHẦN NÀY - Generate embeddings
                dense = embed_data.generate_embeddings(texts)  # Dùng hàm embed động
                sparse = list(self.sparse_embedding.embed(texts))
                
                # Create points
                for j, doc in enumerate(batch):
                    # Đảm bảo dense embedding là list
                    if hasattr(dense[j], 'tolist'):
                        dense_vector = dense[j].tolist()
                    else:
                        dense_vector = dense[j]
                    
                    points.append(
                        PointStruct(
                            id=i+j,
                            vector={
                                "dense": dense_vector,
                                "sparse": {
                                    "indices": sparse[j].indices.tolist(),
                                    "values": sparse[j].values.tolist()
                                }
                            },
                            payload = {
                                "text": doc.page_content,
                                "metadata": doc.metadata
                            }

                        )
                    )
                
                # Upload when batch is full
                if len(points) >= 10:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=False
                    )
                    total_uploaded += len(points)
                    pbar.set_postfix({"uploaded": total_uploaded})
                    points = []
                
                pbar.update(len(batch))
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            total_uploaded += len(points)
        
        logger.info(f"Upload completed! Total vectors: {total_uploaded}")

    def load_data_and_store(self):
        documents = self.load_data()
        self.initialize_client()
        self.create_collection()
        self.embedd_and_store(documents)
    def _check_client(self):
        if not self.client:
            raise RuntimeError("Qdrant client is not initialized. Call initialize_client() first.")
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
        sparse_query = list(self.sparse_embedding.embed([query]))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_query, using="dense", limit=top_k * 2 + 1),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using="sparse",
                    limit=top_k * 2 + 1,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        logger.info(f"Search completed. Found {len(results.points)} results.")
        return [
            {"score": hit.score, **hit.payload}
            for hit in results.points
        ]
