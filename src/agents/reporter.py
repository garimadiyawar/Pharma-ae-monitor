import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AEMonitorState


def reporter_node(state: AEMonitorState) -> dict:
    """
    Agent 3: Writes the final safety signal report.

    Takes the structured clusters and signals from the Analyzer and produces:
    - A narrative executive summary
    - A structured list of signals with severity ratings
    - A clear recommendation for human reviewers
    """
    print(f"\n[Reporter] Generating safety signal report for {state['drug_name']}...")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    red_signals = [s for s in state["signals"] if s["severity"] == "RED"]
    yellow_signals = [s for s in state["signals"] if s["severity"] == "YELLOW"]

    if red_signals:
        recommendation = "ESCALATE — One or more RED signals identified. Human pharmacovigilance review required."
    elif yellow_signals:
        recommendation = "MONITOR — Yellow signals present. Schedule routine signal review within 30 days."
    else:
        recommendation = "ROUTINE — No significant signals. Continue standard monitoring schedule."

    messages = [
        SystemMessage(content=(
            "You are a senior pharmacovigilance analyst writing a regulatory-grade safety signal report. "
            "Be precise, clinical, and concise. Avoid hedging language. "
            "This report will be reviewed by a human expert."
        )),
        HumanMessage(content=(
            f"Drug: {state['drug_name']}\n"
            f"Total FAERS records: {state['total_count']:,}\n"
            f"Sample size: {len(state['raw_reports'])}\n"
            f"Serious reports: {state['serious_count']} "
            f"({100*state['serious_count']//max(len(state['raw_reports']),1)}%)\n"
            f"Death reports: {state['death_count']}\n"
            f"Top reactions: {', '.join(state['top_reactions'][:5])}\n\n"
            f"Organ system clusters:\n{json.dumps(state['clusters'][:5], indent=2)}\n\n"
            f"Identified signals:\n{json.dumps(state['signals'], indent=2)}\n\n"
            f"Overall recommendation: {recommendation}\n\n"
            "Write a 3-paragraph safety signal report:\n"
            "Paragraph 1: Overview of the data (what was searched, how much data, overall seriousness profile)\n"
            "Paragraph 2: Key findings — the most important signals, their evidence, clinical significance\n"
            "Paragraph 3: Recommendation and suggested next steps for the human reviewer\n\n"
            "Keep it under 250 words. Clinical tone. No bullet points."
        )),
    ]

    response = llm.invoke(messages)

    print("[Reporter] Report generated.")

    return {
        "report": response.content.strip(),
        "recommendation": recommendation,
    }


def error_node(state: AEMonitorState) -> dict:
    """
    Error handler: Called when the Fetcher finds no data.
    Returns a clean error report rather than crashing.
    """
    print(f"\n[Error Handler] No data for '{state['drug_name']}'. Generating error report.")
    return {
        "report": (
            f"No adverse event records were found in OpenFDA FAERS for '{state['drug_name']}'. "
            f"\n\nThis may be because:\n"
            f"• The drug name is misspelled (try the generic name)\n"
            f"• The drug is very new or not yet in FAERS\n"
            f"• The drug is not marketed in the US\n\n"
            f"Error detail: {state.get('fetch_error', 'Unknown error')}"
        ),
        "signals": [],
        "recommendation": "RETRY — Check drug name and try again.",
    }
