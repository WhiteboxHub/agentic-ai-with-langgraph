# temporary_retriever.py
class TemporaryRetriever:
    def __init__(self):
        self.docs = {
            "policy": "Medicaid covers hospitalization, preventive care, and maternity benefits.",
            "claims": "Claims are processed within 10 business days after submission."
        }

    def retrieve(self, query):
        if "policy" in query.lower():
            return self.docs["policy"]
        elif "claim" in query.lower():
            return self.docs["claims"]
        else:
            return "No relevant document found."
