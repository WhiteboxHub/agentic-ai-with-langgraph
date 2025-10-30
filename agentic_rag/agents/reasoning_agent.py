# reasoning_agent.py
from policy_agent import PolicyAgent
from claims_agent import ClaimsAgent

class ReasoningAgent:
    def __init__(self, retriever):
        self.policy_agent = PolicyAgent(retriever)
        self.claims_agent = ClaimsAgent(retriever)

    def run(self, state):
        query = state["query"]
        policy_resp = self.policy_agent.run({"query": query})["answer"]
        claim_resp = self.claims_agent.run({"query": query})["answer"]
        state["answer"] = (f"[Reasoning Agent] Combined reasoning:\n"
                           f"- Policy Context: {policy_resp}\n"
                           f"- Claims Context: {claim_resp}\n"
                           f"→ Final Interpretation: The claim denial might be due to eligibility rules.")
        return state
