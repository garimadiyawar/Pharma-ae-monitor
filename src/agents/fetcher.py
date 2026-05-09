from src.state import AEMonitorState
from src.tools.openfda import fetch_adverse_events


def fetcher_node(state: AEMonitorState) -> dict:
    """
    Agent 1: Fetches adverse event reports from OpenFDA.

    Calls the OpenFDA FAERS API, cleans the response, and writes
    raw_reports, total_count, and fetch_error into the shared state.
    """
    drug_name = state["drug_name"]
    print(f"\n[Fetcher] Querying OpenFDA for: {drug_name}")

    result = fetch_adverse_events.invoke({"drug_name": drug_name, "limit": 100})

    if result["error"] or result["total"] == 0:
        print(f"[Fetcher] No data found: {result['error']}")
        return {
            "raw_reports": [],
            "total_count": 0,
            "fetch_error": result["error"] or f"No records found for '{drug_name}'.",
        }

    print(f"[Fetcher] Found {result['total']:,} total records. Fetched {len(result['results'])}.")
    return {
        "raw_reports": result["results"],
        "total_count": result["total"],
        "fetch_error": None,
    }


def should_analyze(state: AEMonitorState) -> str:
    """
    Conditional edge: decide whether to analyze or handle the error.
    LangGraph calls this after fetcher_node to pick the next node.
    """
    if state.get("fetch_error"):
        return "handle_error"
    return "analyze"
