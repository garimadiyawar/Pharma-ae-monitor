from typing import TypedDict, Optional


class AEMonitorState(TypedDict):
    """
    Shared state passed between every node in the LangGraph.
    Each agent reads what it needs and writes its outputs here.
    """

    # Input
    drug_name: str

    # Fetcher outputs
    raw_reports: list[dict]       # raw OpenFDA event records
    total_count: int              # how many total records OpenFDA has
    fetch_error: Optional[str]    # set if API call fails or no data found

    # Analyzer outputs
    clusters: list[dict]          # reactions grouped by organ system
    top_reactions: list[str]      # top 10 reactions by frequency
    serious_count: int            # how many reports are flagged serious
    death_count: int              # how many reports mention death

    # Reporter outputs
    report: str                   # final narrative report
    signals: list[dict]           # list of {signal, severity: RED|YELLOW|GREEN, evidence}
    recommendation: str           # overall recommendation for human review
