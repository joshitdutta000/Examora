from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    analyze_node,
    retrieve_node,
    reason_node,
    recommend_node,
    report_node,
)

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze",    analyze_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("reason",     reason_node)
    graph.add_node("recommend",  recommend_node)
    graph.add_node("report",     report_node)

    # Define flow
    graph.set_entry_point("analyze")
    graph.add_edge("analyze",   "retrieve")
    graph.add_edge("retrieve",  "reason")
    graph.add_edge("reason",    "recommend")
    graph.add_edge("recommend", "report")
    graph.add_edge("report",    END)

    return graph.compile()


# Convenience function for the UI
def run_agent(inputs: dict) -> dict:
    app = build_graph()
    result = app.invoke(inputs)
    return result