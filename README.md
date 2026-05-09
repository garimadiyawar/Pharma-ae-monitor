# Pharma Adverse Event Monitor

A 3-agent LangGraph pipeline that detects drug safety signals from real FDA adverse event data.

Given a drug name, the system fetches reports from the OpenFDA FAERS database, clusters reactions by organ system, identifies safety signals (RED / YELLOW / GREEN), and generates a pharmacovigilance-grade report — the kind of work IQVIA's pharmacovigilance teams do at scale.

---

## Architecture

```
START
  ↓
Agent 1 — Fetcher       Calls OpenFDA API. If no data → error handler.
  ↓ (conditional routing)
Agent 2 — Analyzer      Rule-based organ system clustering + LLM signal identification.
  ↓
Agent 3 — Reporter      Writes narrative report + overall recommendation.
  ↓
END
```

All steps are traced in LangSmith for full observability.

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/garimadiyawar/pharma-ae-monitor
cd pharma-ae-monitor
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and LANGCHAIN_API_KEY

# 3. Run the Streamlit UI
streamlit run app.py

# Or run from CLI
python main.py --drug ibuprofen
python main.py --drug metformin
python main.py --drug sildenafil
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Observability | LangSmith |
| LLM | Claude 3.5 Haiku (via langchain-anthropic) |
| Data source | OpenFDA FAERS API (free, no auth) |
| Frontend | Streamlit |
| Language | Python 3.11+ |

---

## Design decisions worth discussing

**Why hybrid in Agent 2?**  
The Analyzer uses rule-based keyword matching first, then an LLM pass. This is deliberate — passing 100 raw event records directly to an LLM is expensive, slow, and produces inconsistent output. The rule-based step reduces the problem to a structured cluster summary, which the LLM can reason over reliably.

**Why conditional routing?**  
The graph routes to an error node rather than raising an exception when no data is found. This is what enterprise-grade agentic systems require — graceful degradation, not crashes.

**Why structured LLM output?**  
Agent 2 prompts the LLM for JSON only, then strips markdown fences and parses safely. This is the production-grade pattern for LLM-to-code handoff.

**Why LangSmith from day one?**  
Every node execution is traced automatically. In interviews, you can pull up the LangSmith dashboard and show exactly how the agents hand off — inputs, outputs, latency, token usage.

---

## Example drugs to test

- `ibuprofen` — large dataset, interesting GI + cardiac signals
- `metformin` — well-studied, mostly green signals
- `sildenafil` — cardiac signals, good for showing RED flags
- `warfarin` — bleeding signals, classic pharmacovigilance case
- `fakedrugxyz` — triggers the error handler / conditional routing

---

## Interview talking points

1. **"Why LangGraph over plain sequential code?"**  
   LangGraph gives you a real state machine. The conditional edge after the Fetcher means the graph can re-route at runtime based on what the data actually is — that's not possible with a simple pipeline.

2. **"What's the hybrid design in Agent 2?"**  
   Rule-based first, LLM second. The LLM never sees raw records — it sees a clean cluster summary. This is how real agentic systems are built: LLMs for reasoning, not data wrangling.

3. **"How would you scale this?"**  
   Add a retry loop in the Fetcher for rate-limit handling. Add a human-in-the-loop node after the Reporter for RED signals. Parallelize the Analyzer across organ systems with LangGraph's Send() API.

4. **"What does FAERS stand for?"**  
   FDA Adverse Event Reporting System — the primary post-market drug safety database in the US. IQVIA works with this data extensively in its pharmacovigilance services.
