# import numpy as np
# from typing import List, Tuple
# from sentence_transformers import SentenceTransformer
# from pymilvus import connections, Collection
# from langchain.schema.retriever import BaseRetriever


# class MilvusRetriever(BaseRetriever):
#     """
#     Retrieve relevant chunks from Milvus based on query embedding similarity.
#     Works with Milvus Standalone instance.
#     """
#     def __init__(self,
#                  collection_name: str = "documents_chunks",
#                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
#                  milvus_host: str = "localhost",
#                  milvus_port: str = "19530"):
        
#         super().__init__()
#         # Connect to Milvus
#         print(f"🔌 Connecting to Milvus at {milvus_host}:{milvus_port} ...")
#         connections.connect(alias="default", host=milvus_host, port=milvus_port)
        

#         # Load the collection
#         self.collection = Collection(collection_name)
#         self.model = SentenceTransformer(model_name)

#         # Load collection metadata
#         self.collection.load()
#         print(f"Connected to Milvus collection: {collection_name}")

#     # ----------------------------------------------------------------------
#     def embed_query(self, query: str) -> np.ndarray:
#         """Convert query into an embedding using SentenceTransformer."""
#         embedding = self.model.encode([query])
#         return embedding

#     # ----------------------------------------------------------------------
#     def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
#         """
#         Search for the top_k most relevant chunks in Milvus.
#         Returns list of (chunk_text, similarity_score).
#         """
#         # Step 1: Embed the query
#         query_embedding = self.embed_query(query)

#         # Step 2: Perform vector search
#         search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
#         results = self.collection.search(
#             data=query_embedding,
#             anns_field="embedding",
#             param=search_params,
#             limit=top_k,
#             output_fields=["chunk_text"]
#         )

#         # Step 3: Format results
#         retrieved = []
#         for hit in results[0]:
#             chunk = hit.entity.get("chunk_text")
#             score = 1 - hit.distance  # Convert distance → similarity (for cosine)
#             retrieved.append((chunk, round(score, 4)))

#         return retrieved


# # if __name__ == "__main__":
# #     retriever = MilvusRetriever(
# #         collection_name="documents_chunks",
# #         model_name="sentence-transformers/all-MiniLM-L6-v2",
# #         milvus_host="localhost",
# #         milvus_port="19530"
# #     )

# #     query = "Compare performance between Apple M3 Max and Intel Xeon"
# #     results = retriever.search(query, top_k=5)

# #     print("\n🔹 Top Retrieved Chunks:")
# #     for i, (chunk, score) in enumerate(results, 1):
# #         print(f"\nResult {i}: (score={score})\n{chunk}")

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
    top_k: int = 5

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
