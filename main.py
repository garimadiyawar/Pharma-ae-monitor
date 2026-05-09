"""
CLI runner for the Pharma AE Monitor.
Usage: python main.py --drug ibuprofen
"""
import argparse
import json
import os

from dotenv import load_dotenv
from src.graph import ae_graph

load_dotenv()


def run(drug_name: str) -> dict:
    """Run the full LangGraph pipeline for a given drug."""
    initial_state = {
        "drug_name": drug_name,
        "raw_reports": [],
        "total_count": 0,
        "fetch_error": None,
        "clusters": [],
        "top_reactions": [],
        "serious_count": 0,
        "death_count": 0,
        "report": "",
        "signals": [],
        "recommendation": "",
    }

    print(f"\n{'='*60}")
    print(f"  Pharma Adverse Event Monitor")
    print(f"  Drug: {drug_name}")
    print(f"{'='*60}")

    final_state = ae_graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("  SAFETY SIGNAL REPORT")
    print(f"{'='*60}\n")
    print(final_state["report"])
    print(f"\n  Recommendation: {final_state['recommendation']}")

    if final_state.get("signals"):
        print(f"\n  Signals identified:")
        for s in final_state["signals"]:
            badge = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(s["severity"], "⚪")
            print(f"  {badge}  {s['signal']}")
            print(f"       {s['evidence']}")

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharma Adverse Event Monitor")
    parser.add_argument("--drug", required=True, help="Drug name to analyze")
    args = parser.parse_args()

    run(args.drug)
