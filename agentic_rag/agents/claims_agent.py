# claims_agent.py
class ClaimsAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state):
        query = state["query"]
        docs = self.retriever.retrieve(query)
        state["answer"] = f"[Claims Agent] {docs}"
        return state
