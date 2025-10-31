import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever


class MilvusRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for Milvus Standalone.
    Retrieves top-k relevant chunks as LangChain Document objects.
    """

    collection_name: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    top_k: int = 3

    # Internal (non-pydantic) fields
    _collection: Optional[Collection] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Connect to Milvus
        print(f"🔌 Connecting to Milvus at {self.milvus_host}:{self.milvus_port} ...")
        connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)

        # Load collection
        self._collection = Collection(self.collection_name)
        self._collection.load()

        # Load embedding model
        self._model = SentenceTransformer(self.model_name)
        print(f"✅ Connected to Milvus collection: {self.collection_name}")

    # ----------------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """Convert query text into embedding vector."""
        return self._model.encode([query])

    # ----------------------------------------------------------------------
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain-compatible retrieval method."""
        query_embedding = self.embed_query(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self._collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=self.top_k,
            output_fields=["chunk_text"]
        )

        documents = []
        for hit in results[0]:
            chunk_text = hit.entity.get("chunk_text")
            score = 1 - hit.distance
            documents.append(Document(page_content=chunk_text, metadata={"score": score}))
        # print("these are the retrieved dopcuments ---",documents)

        return documents

    # ----------------------------------------------------------------------
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieval support."""
        return self._get_relevant_documents(query)
