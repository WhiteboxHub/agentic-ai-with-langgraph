import json
from utils.retriever import MilvusRetriever

def precision_recall_at_k(retrieved, relevant, k=5):
    retrieved_top_k = retrieved[:k]
    relevant_count = sum(1 for chunk, _ in retrieved_top_k if chunk in relevant)
    
    precision = relevant_count / k
    recall = relevant_count / len(relevant) if len(relevant) > 0 else 0
    return precision, recall

def evaluate_retrieval(ground_truth_file="evaluation/ground_truth.json", k=5):
    print("entered")
    retriever = MilvusRetriever(
        collection_name="documents_chunks",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        milvus_host="localhost",
        milvus_port="19530"
    )
    if not ground_truth_file:
        print("cant fine the json file ---------")
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_precision, total_recall = 0, 0

    for item in dataset:
        query = item["query"]
        relevant_chunks = item["relevant_chunks"]

        retrieved = retriever.search(query, top_k=k)

        precision, recall = precision_recall_at_k(retrieved, relevant_chunks, k)
        print(f"\n🔹 Query: {query}")
        print(f"Precision@{k}: {precision:.2f}, Recall@{k}: {recall:.2f}")

        total_precision += precision
        total_recall += recall

    n = len(dataset)
    print("\nAverage Precision@{}: {:.2f}".format(k, total_precision / n))
    print("Average Recall@{}: {:.2f}".format(k, total_recall / n))

if __name__ == "__main__":
    
    evaluate_retrieval(k=5)
    
