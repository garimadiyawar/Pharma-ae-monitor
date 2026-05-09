"""
Streamlit frontend for the Pharma Adverse Event Monitor.
Run with: streamlit run app.py
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Must be first Streamlit call
st.set_page_config(
    page_title="Pharma AE Monitor",
    page_icon="💊",
    layout="wide",
)

from src.graph import ae_graph  # noqa: E402 (import after set_page_config)


# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .signal-red    { background:#FEE2E2; border-left:4px solid #DC2626; padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; }
    .signal-yellow { background:#FEF9C3; border-left:4px solid #CA8A04; padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; }
    .signal-green  { background:#DCFCE7; border-left:4px solid #16A34A; padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; }
    .rec-escalate  { background:#FEE2E2; padding:14px 18px; border-radius:8px; font-weight:600; color:#991B1B; }
    .rec-monitor   { background:#FEF9C3; padding:14px 18px; border-radius:8px; font-weight:600; color:#713F12; }
    .rec-routine   { background:#DCFCE7; padding:14px 18px; border-radius:8px; font-weight:600; color:#14532D; }
    .rec-retry     { background:#F1F5F9; padding:14px 18px; border-radius:8px; font-weight:600; color:#475569; }
    .metric-card   { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.title("💊 Pharma Adverse Event Monitor")
st.caption("Powered by OpenFDA FAERS database · LangGraph multi-agent pipeline · Claude AI")
st.divider()


# ── Input ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    drug_name = st.text_input(
        "Drug name",
        placeholder="e.g. ibuprofen, metformin, atorvastatin, sildenafil",
        label_visibility="collapsed",
    )
with col2:
    run_button = st.button("Analyze ▶", use_container_width=True, type="primary")

st.caption("Uses real FDA adverse event data. Try common drugs for best results.")


# ── Run ────────────────────────────────────────────────────────────────────
if run_button and drug_name:
    initial_state = {
        "drug_name": drug_name.strip(),
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

    with st.status(f"Running 3-agent pipeline for **{drug_name}**...", expanded=True) as status:
        st.write("🔍 Agent 1: Fetching OpenFDA FAERS records...")
        # Stream node-by-node using LangGraph's stream()
        final_state = None
        for event in ae_graph.stream(initial_state):
            node_name = list(event.keys())[0]
            if node_name == "fetch":
                if event["fetch"].get("fetch_error"):
                    st.write("⚠️ Agent 1: No data found — routing to error handler")
                else:
                    count = event["fetch"].get("total_count", 0)
                    fetched = len(event["fetch"].get("raw_reports", []))
                    st.write(f"✅ Agent 1: Found {count:,} records · fetched {fetched}")
                    st.write("🧬 Agent 2: Clustering reactions by organ system...")
            elif node_name == "analyze":
                clusters = event["analyze"].get("clusters", [])
                signals = event["analyze"].get("signals", [])
                st.write(f"✅ Agent 2: {len(clusters)} organ systems · {len(signals)} signals identified")
                st.write("📋 Agent 3: Writing safety signal report...")
            elif node_name == "report":
                st.write("✅ Agent 3: Report complete")
            elif node_name == "handle_error":
                st.write("⚠️ Error handler: Generating error report")
            final_state = list(event.values())[0]

        status.update(label="Pipeline complete", state="complete")

    if not final_state:
        st.error("Something went wrong. Please try again.")
        st.stop()

    # ── Merge all node outputs into a complete state ───────────────────────
    # When streaming, each event only has that node's outputs.
    # Re-invoke to get the complete final state.
    complete_state = ae_graph.invoke(initial_state)

    # ── Results ────────────────────────────────────────────────────────────
    st.divider()

    # Recommendation banner
    rec = complete_state.get("recommendation", "")
    if "ESCALATE" in rec:
        st.markdown(f'<div class="rec-escalate">🔴 {rec}</div>', unsafe_allow_html=True)
    elif "MONITOR" in rec:
        st.markdown(f'<div class="rec-monitor">🟡 {rec}</div>', unsafe_allow_html=True)
    elif "ROUTINE" in rec:
        st.markdown(f'<div class="rec-routine">🟢 {rec}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="rec-retry">⚪ {rec}</div>', unsafe_allow_html=True)

    st.write("")

    # Metrics row
    if complete_state.get("total_count"):
        m1, m2, m3, m4 = st.columns(4)
        total = complete_state["total_count"]
        sampled = len(complete_state.get("raw_reports", []))
        serious = complete_state.get("serious_count", 0)
        deaths = complete_state.get("death_count", 0)
        serious_pct = f"{100*serious//max(sampled,1)}%" if sampled else "—"

        m1.metric("Total FAERS records", f"{total:,}")
        m2.metric("Sample analyzed", sampled)
        m3.metric("Serious reports", f"{serious} ({serious_pct})")
        m4.metric("Death reports", deaths)

    st.divider()

    # Two-column layout: report + signals
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Safety signal report")
        st.markdown(complete_state.get("report", "No report generated."))

        # Top reactions
        if complete_state.get("top_reactions"):
            st.subheader("Top reactions")
            for i, r in enumerate(complete_state["top_reactions"][:8], 1):
                st.markdown(f"`{i}.` {r}")

    with right:
        st.subheader("Identified signals")
        signals = complete_state.get("signals", [])
        if signals:
            for s in signals:
                sev = s.get("severity", "GREEN").upper()
                css = {"RED": "signal-red", "YELLOW": "signal-yellow", "GREEN": "signal-green"}.get(sev, "signal-green")
                badge = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(sev, "🟢")
                st.markdown(
                    f'<div class="{css}"><strong>{badge} {s.get("signal","")}</strong>'
                    f'<br><small>{s.get("evidence","")}</small></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No signals returned.")

        # Organ system breakdown
        if complete_state.get("clusters"):
            st.subheader("Organ system breakdown")
            for cluster in complete_state["clusters"][:7]:
                pct = int(100 * cluster["serious"] / max(cluster["total"], 1))
                st.markdown(
                    f"**{cluster['organ_system']}**  "
                    f"`{cluster['total']} reports` · `{pct}% serious`"
                )
                st.progress(min(pct / 100, 1.0))

elif run_button and not drug_name:
    st.warning("Please enter a drug name.")
else:
    # Landing state
    st.info(
        "Enter a drug name above to run the 3-agent LangGraph pipeline. "
        "The system will fetch real FDA adverse event data, cluster reactions by organ system, "
        "identify safety signals, and generate a pharmacovigilance-grade report."
    )
    with st.expander("How it works"):
        st.markdown("""
**Agent 1 — Fetcher**
Calls the OpenFDA FAERS API and extracts structured fields from adverse event reports.
If no data is found, the graph routes directly to an error handler (conditional edge in LangGraph).

**Agent 2 — Analyzer**  
Uses a hybrid approach: rule-based keyword matching to cluster reactions into organ systems (fast, deterministic),
then an LLM pass to identify the 3–5 most significant safety signals with RED/YELLOW/GREEN severity ratings.

**Agent 3 — Reporter**  
Takes the structured signals and clusters and writes a regulatory-grade narrative report
with an overall recommendation for human pharmacovigilance reviewers.

All steps are traced in LangSmith for full observability.
        """)
