# policy_agent.py
class PolicyAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state):
        query = state["query"]
        docs = self.retriever.retrieve(query)
        state["answer"] = f"[Policy Agent] {docs}"
        return state
