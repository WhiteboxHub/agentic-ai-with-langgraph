# fallback_agent.py
class FallbackAgent:
    def run(self, state):
        state["answer"] = "[Fallback Agent] I could not determine the intent. Please clarify."
        return state
