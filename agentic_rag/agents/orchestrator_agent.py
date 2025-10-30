# orchestrator_agent.py
class OrchestratorAgent:
    def classify_intent(self, query):
        q = query.lower()
        if "policy" in q or "eligibility" in q:
            return "policy"
        elif "claim" in q or "reimbursement" in q:
            return "claims"
        elif "why" in q or "how" in q:
            return "reasoning"
        else:
            return "fallback"

    # This node just passes the query along with intent for graph routing
    def run(self, state):
        query = state["query"]
        intent = self.classify_intent(query)
        state["intent"] = intent
        return state
