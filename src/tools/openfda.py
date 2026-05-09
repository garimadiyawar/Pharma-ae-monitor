import requests
from langchain_core.tools import tool


OPENFDA_BASE = "https://api.fda.gov/drug/event.json"


@tool
def fetch_adverse_events(drug_name: str, limit: int = 100) -> dict:
    """
    Fetch adverse event reports for a drug from the OpenFDA FAERS database.

    Args:
        drug_name: The drug's brand or generic name (e.g. 'aspirin', 'ibuprofen')
        limit: Max number of reports to fetch (default 100, max 1000)

    Returns:
        Dictionary with 'total', 'results' (list of event records), and 'error' if any.
    """
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": min(limit, 1000),
    }

    try:
        response = requests.get(OPENFDA_BASE, params=params, timeout=15)

        if response.status_code == 404:
            # OpenFDA returns 404 when no records match
            return {
                "total": 0,
                "results": [],
                "error": f"No adverse event records found for '{drug_name}'. "
                         "Check spelling or try the generic name."
            }

        response.raise_for_status()
        data = response.json()

        total = data.get("meta", {}).get("results", {}).get("total", 0)
        results = data.get("results", [])

        # Extract only the fields we care about — keep records lean
        cleaned = []
        for record in results:
            patient = record.get("patient", {})
            reactions = [
                r.get("reactionmeddrapt", "unknown")
                for r in patient.get("reaction", [])
            ]
            drugs = [
                d.get("medicinalproduct", "unknown")
                for d in patient.get("drug", [])
            ]
            cleaned.append({
                "reactions": reactions,
                "drugs": drugs,
                "serious": record.get("serious", 0) == 1,
                "serious_death": record.get("seriousnessdeath", 0) == 1,
                "serious_hospitalization": record.get("seriousnesshospitalization", 0) == 1,
                "serious_lifethreat": record.get("seriousnesslifethreatening", 0) == 1,
                "patient_age": patient.get("patientonsetage"),
                "patient_sex": patient.get("patientsex"),
                "outcomes": [
                    r.get("reactionoutcome") for r in patient.get("reaction", [])
                ],
            })

        return {"total": total, "results": cleaned, "error": None}

    except requests.exceptions.Timeout:
        return {"total": 0, "results": [], "error": "OpenFDA API timed out. Try again."}
    except requests.exceptions.RequestException as e:
        return {"total": 0, "results": [], "error": f"API error: {str(e)}"}
