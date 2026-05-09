from langgraph.graph import StateGraph, START, END

from src.state import AEMonitorState
from src.agents.fetcher import fetcher_node, should_analyze
from src.agents.analyzer import analyzer_node
from src.agents.reporter import reporter_node, error_node


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph for the AE Monitor.

    Graph structure:
        START
          ↓
        fetcher_node
          ↓ (conditional)
          ├── "analyze"      → analyzer_node → reporter_node → END
          └── "handle_error" → error_node → END
    """
    graph = StateGraph(AEMonitorState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("fetch",        fetcher_node)
    graph.add_node("analyze",      analyzer_node)
    graph.add_node("report",       reporter_node)
    graph.add_node("handle_error", error_node)

    # ── Define edges ──────────────────────────────────────────────────────────
    # Entry point
    graph.add_edge(START, "fetch")

    # Conditional routing after fetch
    graph.add_conditional_edges(
        "fetch",
        should_analyze,               # function returns "analyze" or "handle_error"
        {
            "analyze":      "analyze",
            "handle_error": "handle_error",
        }
    )

    # Happy path
    graph.add_edge("analyze", "report")
    graph.add_edge("report",  END)

    # Error path
    graph.add_edge("handle_error", END)

    return graph.compile()


# Module-level compiled graph — import this in app.py and main.py
ae_graph = build_graph()
