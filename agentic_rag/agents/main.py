# main.py
from langgraph.graph import StateGraph, START,END
from orchestrator_agent import OrchestratorAgent
from policy_agent import PolicyAgent
from claims_agent import ClaimsAgent
from reasoning_agent import ReasoningAgent
from fallback_agent import FallbackAgent
from temporary_retriever import TemporaryRetriever
from memory_manager import MemoryManager

# Initialize components
retriever = TemporaryRetriever()
memory = MemoryManager()

orchestrator_node = OrchestratorAgent()
policy_node = PolicyAgent(retriever)
claims_node = ClaimsAgent(retriever)
reasoning_node = ReasoningAgent(retriever)
fallback_node = FallbackAgent()

from typing import TypedDict, Dict, Any

class AgentState(TypedDict):
    query: str
    intent: str
    answer: str
    context: Dict[str, Any]

# Build LangGraph DAG
graph = StateGraph(state_schema=AgentState,entry_point="orchestrator")

graph.add_node("orchestrator", orchestrator_node.run)
graph.add_node("policy_agent", policy_node.run)
graph.add_node("claims_agent", claims_node.run)
graph.add_node("reasoning_agent", reasoning_node.run)
graph.add_node("fallback_agent", fallback_node.run)

# Conditional routing based on intent
graph.add_conditional_edges(
    "orchestrator",
    lambda state: state["intent"],
    {
        "policy": "policy_agent",
        "claims": "claims_agent",
        "reasoning": "reasoning_agent",
        "fallback": "fallback_agent"
    }
)

# All nodes go to END
graph.add_edge("policy_agent", END)
graph.add_edge("claims_agent", END)
graph.add_edge("reasoning_agent", END)
graph.add_edge("fallback_agent", END)
graph.add_edge(START, "orchestrator")

workflow = graph.compile()

# ------------------------
# Run interactive session
# ------------------------
if __name__ == "__main__":
    print("Agentic RAG System (LangGraph DAG) running. Type 'exit' to quit.")
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            break
        final_state = workflow.invoke({"query": query})
        print(f"Agent: {final_state['answer']}")
