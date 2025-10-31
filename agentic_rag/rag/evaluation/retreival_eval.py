import json
# from utils.retriever import MilvusRetriever
import os
import sys

# Get the base directory path (one level up from this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one directory to reach 'rag'
BASE_DIR = os.path.dirname(BASE_DIR)

# Add the base directory to the system path
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Now import your module
from utils.retriever import MilvusRetriever

print("import succes")

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def precision_recall_at_k_semantic(retrieved, relevant, k=5, threshold=0.6):
    retrieved_top_k = retrieved[:k]
    
    # Handle both LangChain Documents and (text, score) tuples
    if hasattr(retrieved_top_k[0], "page_content"):
        retrieved_texts = [doc.page_content for doc in retrieved_top_k]
    elif isinstance(retrieved_top_k[0], tuple):
        retrieved_texts = [chunk for chunk, _ in retrieved_top_k]
    else:
        retrieved_texts = retrieved_top_k

    relevant_embs = model.encode(relevant, convert_to_tensor=True)
    retrieved_embs = model.encode(retrieved_texts, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(retrieved_embs, relevant_embs)
    hits = (cos_scores > threshold).any(dim=1).sum().item()
    
    precision = hits / k
    recall = hits / len(relevant) if len(relevant) > 0 else 0
    return precision, recall


def evaluate_retrieval(ground_truth_file="./ground_truth.json", k=2):
    print("entered")
    retriever = MilvusRetriever(
        collection_name="documents_chunks",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        milvus_host="localhost",
        milvus_port="19530"
    )

    if not os.path.exists(ground_truth_file):
        print("Ground truth file not found.")
        return

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_precision, total_recall = 0, 0

    for item in dataset:
        query = item["query"]
        relevant_chunks = [r.strip() for r in item["relevant_chunks"]]

        retrieved = retriever._get_relevant_documents(query)

        precision, recall = precision_recall_at_k_semantic(retrieved, relevant_chunks, k)
        print(f"\n🔹 Query: {query}")
        print(f"Precision@{k}: {precision:.2f}, Recall@{k}: {recall:.2f}")

        total_precision += precision
        total_recall += recall

    n = len(dataset)
    print("\nAverage Precision@{}: {:.2f}".format(k, total_precision / n))
    print("Average Recall@{}: {:.2f}".format(k, total_recall / n))


if __name__ == "__main__":
    evaluate_retrieval(k=3)
