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
git clone https://github.com/garimadiyawar/Pharma-ae-monitor
cd Pharma-ae-monitor
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in GROQ_API_KEY and LANGCHAIN_API_KEY

# 3. Run the Streamlit UI
streamlit run app.py
OR
python -m streamlit run app.py

# Or run from CLI
python main.py --drug ibuprofen
python main.py --drug metformin
python main.py --drug sildenafil
```

---

## Tech stack

| Layer               | Technology                                 |
|---------------------|--------------------------------------------|
| Agent orchestration | LangGraph                                  |
| Observability       | LangSmith                                  |
| LLM                 | Claude 3.5 Haiku (via langchain-anthropic) |
| Data source         | OpenFDA FAERS API (free, no auth)          |
| Frontend            | Streamlit                                  |
| Language            | Python 3.11+                               |

---

## Example drugs to test

- `ibuprofen` — large dataset, interesting GI + cardiac signals
- `metformin` — well-studied, mostly green signals
- `sildenafil` — cardiac signals, good for showing RED flags
- `warfarin` — bleeding signals, classic pharmacovigilance case
- `fakedrugxyz` — triggers the error handler / conditional routing

