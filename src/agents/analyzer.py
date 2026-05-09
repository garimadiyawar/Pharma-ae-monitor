import json
from collections import Counter

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AEMonitorState


# Coarse MedDRA-inspired organ system mapping
# Maps reaction keywords → organ system categories
ORGAN_SYSTEM_MAP = {
    "Cardiac": ["cardiac", "heart", "myocardial", "arrhythmia", "palpitation",
                "tachycardia", "bradycardia", "atrial", "ventricular", "coronary"],
    "Gastrointestinal": ["nausea", "vomiting", "diarrhoea", "diarrhea", "abdominal",
                         "gastric", "intestinal", "colitis", "pancreatitis", "hepatic",
                         "liver", "constipation", "dyspepsia"],
    "Neurological": ["headache", "dizziness", "seizure", "convulsion", "tremor",
                     "neuropathy", "stroke", "syncope", "confusion", "insomnia",
                     "paraesthesia", "cerebral", "migraine"],
    "Respiratory": ["dyspnoea", "dyspnea", "cough", "pneumonia", "bronchospasm",
                    "pulmonary", "respiratory", "asthma", "rhinitis", "throat"],
    "Dermatological": ["rash", "urticaria", "pruritus", "erythema", "dermatitis",
                       "alopecia", "skin", "sweating", "oedema", "swelling"],
    "Musculoskeletal": ["arthralgia", "myalgia", "muscle", "joint", "back pain",
                        "fracture", "osteoporosis", "tendon"],
    "Immunological / Allergic": ["anaphylaxis", "hypersensitivity", "allerg",
                                  "angioedema", "immune", "antibody"],
    "Haematological": ["bleeding", "haemorrhage", "hemorrhage", "thrombocytopenia",
                       "anaemia", "anemia", "leucopenia", "coagulation"],
    "Renal / Urinary": ["renal", "kidney", "nephropathy", "creatinine", "urinary",
                        "proteinuria", "dysuria"],
    "General": ["fatigue", "pain", "fever", "pyrexia", "malaise", "death",
                "asthenia", "weight", "appetite", "fall"],
}


def classify_reaction(reaction: str) -> str:
    """Map a reaction string to the best-matching organ system."""
    reaction_lower = reaction.lower()
    for system, keywords in ORGAN_SYSTEM_MAP.items():
        if any(kw in reaction_lower for kw in keywords):
            return system
    return "Other"


def analyzer_node(state: AEMonitorState) -> dict:
    """
    Agent 2: Clusters raw adverse event reports by organ system.

    Step 1 (rule-based): Fast deterministic grouping using keyword matching.
    Step 2 (LLM): Identifies the most significant patterns and emerging signals.
    This hybrid approach is more reliable than asking the LLM to process 100 raw records.
    """
    reports = state["raw_reports"]
    drug_name = state["drug_name"]
    print(f"\n[Analyzer] Processing {len(reports)} reports for {drug_name}...")

    # ── Step 1: Rule-based clustering ─────────────────────────────────────────
    all_reactions: list[str] = []
    serious_count = 0
    death_count = 0
    cluster_map: dict[str, dict] = {
        system: {"reactions": [], "serious": 0, "total": 0}
        for system in list(ORGAN_SYSTEM_MAP.keys()) + ["Other"]
    }

    for report in reports:
        if report.get("serious"):
            serious_count += 1
        if report.get("serious_death"):
            death_count += 1

        for reaction in report.get("reactions", []):
            all_reactions.append(reaction)
            system = classify_reaction(reaction)
            cluster_map[system]["reactions"].append(reaction)
            cluster_map[system]["total"] += 1
            if report.get("serious"):
                cluster_map[system]["serious"] += 1

    # Count reaction frequencies
    reaction_counter = Counter(all_reactions)
    top_reactions = [r for r, _ in reaction_counter.most_common(10)]

    # Build clean cluster list (drop empty systems)
    clusters = [
        {
            "organ_system": system,
            "total": data["total"],
            "serious": data["serious"],
            "top_reactions": [
                r for r, _ in Counter(data["reactions"]).most_common(5)
            ],
        }
        for system, data in cluster_map.items()
        if data["total"] > 0
    ]
    clusters.sort(key=lambda x: x["total"], reverse=True)

    print(f"[Analyzer] Clustered into {len(clusters)} organ systems. "
          f"Serious: {serious_count}, Deaths: {death_count}")

    # ── Step 2: LLM signal identification ────────────────────────────────────
    # Give the LLM only the cluster summary, not 100 raw records
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    cluster_summary = json.dumps(clusters, indent=2)

    messages = [
        SystemMessage(content=(
            "You are a pharmacovigilance analyst. You review adverse event cluster data "
            "and identify safety signals that may need regulatory attention. "
            "Be concise and evidence-based. Do not hallucinate — only report what the data shows."
        )),
        HumanMessage(content=(
            f"Drug: {drug_name}\n"
            f"Total reports sampled: {len(reports)} (of {state['total_count']:,} total in FAERS)\n"
            f"Serious reports: {serious_count} ({100*serious_count//max(len(reports),1)}%)\n"
            f"Reports mentioning death: {death_count}\n\n"
            f"Cluster summary by organ system:\n{cluster_summary}\n\n"
            "Identify the 3-5 most significant safety signals. For each signal, rate it:\n"
            "- RED: high frequency + high serious rate, or any deaths\n"
            "- YELLOW: notable pattern worth monitoring\n"
            "- GREEN: low frequency or expected known effect\n\n"
            "Respond ONLY with a JSON array, no other text:\n"
            '[{"signal": "...", "severity": "RED|YELLOW|GREEN", "evidence": "..."}, ...]'
        )),
    ]

    response = llm.invoke(messages)
    raw_json = response.content.strip()

    # Strip markdown fences if the model wraps it
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    try:
        signals = json.loads(raw_json)
    except json.JSONDecodeError:
        # Fallback: extract signals from clusters directly
        signals = [
            {
                "signal": clusters[0]["organ_system"] if clusters else "Unknown",
                "severity": "YELLOW",
                "evidence": "LLM parsing failed — manual review recommended."
            }
        ]

    print(f"[Analyzer] Identified {len(signals)} signals.")

    return {
        "clusters": clusters,
        "top_reactions": top_reactions,
        "serious_count": serious_count,
        "death_count": death_count,
        "signals": signals,
    }
