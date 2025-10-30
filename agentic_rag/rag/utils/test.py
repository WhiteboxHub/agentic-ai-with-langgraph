from pymilvus import Collection
from sentence_transformers import SentenceTransformer

from pymilvus import connections

connections.connect(
    alias="default",
    host="localhost",  
    port="19530"
)
collection = Collection("doc_chunks")
collection.load()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "what is docing issue? "
embedding = model.encode([query])

results = collection.search(
    data=embedding,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)

for r in results[0]:
    print(r.entity.get("text"), "→", r.distance)
