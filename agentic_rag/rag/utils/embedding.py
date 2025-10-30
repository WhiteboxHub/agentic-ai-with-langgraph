import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from concurrent.futures import ThreadPoolExecutor, as_completed

class EmbeddingGenerator:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        input_dir="../data/chunks",
        milvus_host="localhost",
        milvus_port="19530",
        collection_name="documents_chunks"
    ):
        # Model setup
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.input_dir = input_dir

        # Milvus setup
        self.collection_name = collection_name
        connections.connect("default", host=milvus_host, port=milvus_port)
        print(f"Connected to Milvus at {milvus_host}:{milvus_port}")

        # Create collection if not exists
        self._create_collection_if_not_exists()

    # ----------------------------------------------------------------
    def _create_collection_if_not_exists(self):
        """Create Milvus collection schema if not exists"""
        if utility.has_collection(self.collection_name):
            print(f"Using existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, description="Document chunks with embeddings")

        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Created Milvus collection: {self.collection_name}")

    # ----------------------------------------------------------------
    def read_chunks(self) -> Dict[str, List[str]]:
        """Read all chunked text files from input directory"""
        chunks_dict = {}
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    # Split based on [Chunk x] markers or double newlines
                    chunks = [c.strip() for c in text.split("[Chunk") if c.strip()]
                    chunks_dict[file] = chunks
        return chunks_dict

    # ----------------------------------------------------------------
    def generate_embeddings(self, text_list: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformers model"""
        embeddings = self.model.encode(text_list, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    

    # ----------------------------------------------------------------
    def store_in_milvus(self, texts: List[str], embeddings: np.ndarray):
        """Insert text and embedding pairs into Milvus"""
        data = [
            texts,                # for chunk_text
            embeddings.tolist()   # for embedding
        ]
        self.collection.insert(
            data,
            fields=["chunk_text", "embedding"]  # Explicit field mapping
        )
        print(f"Inserted {len(texts)} records into Milvus collection '{self.collection_name}'")


    def process_file(self,filename, chunks):
            print(f"\nProcessing file: {filename} ({len(chunks)} chunks)")
            embeddings = self.generate_embeddings(chunks)
            self.store_in_milvus(chunks, embeddings)
            return filename
            
    # ----------------------------------------------------------------
    def process_all_files(self):
        """Read chunks, embed them, and store in Milvus"""
        chunks_dict = self.read_chunks()

        
        #     return filename


        # for filename, chunks in chunks_dict.items():
        #     print(f"\nProcessing file: {filename} ({len(chunks)} chunks)")
        #     embeddings = self.generate_embeddings(chunks)
        #     self.store_in_milvus(chunks, embeddings)

        with ThreadPoolExecutor(max_workers=4) as executor:  
            futures = [
                executor.submit(self.process_file, filename, chunks)
                for filename, chunks in chunks_dict.items()
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Completed: {result}")
                except Exception as e:
                    print(f" Error processing file: {e}")


        print("\nAll embeddings stored in Milvus successfully!")

        # Build and load index after insertion
        print("\nCreating index after data insertion...")
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print("Index created successfully!")

        self.collection.load()
        print("Collection loaded and ready for search.")

# --------------------------------------------------------------------
# Run directly
# --------------------------------------------------------------------
# if __name__ == "__main__":
#     embedder = EmbeddingGenerator(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         input_dir="../data/chunks",
#         milvus_host="localhost",
#         milvus_port="19530",
#         collection_name="doc_chunks"
#     )

#     embedder.process_all_files()
