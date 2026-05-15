from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import httpx
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="MedLabelIQ",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_API_BASE_URL = os.getenv(
    "MEDLABELIQ_API_BASE_URL",
    "http://127.0.0.1:8011",
)

NO_DRUG_FILTER = "No drug filter"
NO_FAMILY_FILTER = "No section-family filter"

FALLBACK_DRUG_OPTIONS = [
    NO_DRUG_FILTER,
    "acetaminophen",
    "albuterol",
    "amoxicillin",
    "apixaban",
    "atorvastatin",
    "ibuprofen",
    "isotretinoin",
    "lisinopril",
    "metformin",
    "methotrexate",
    "omeprazole",
    "sertraline",
]

FALLBACK_FAMILY_OPTIONS = [
    NO_FAMILY_FILTER,
    "warnings_and_precautions",
    "warnings",
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "clinical_pharmacology",
    "clinical_studies",
    "use_in_specific_populations",
    "patient_counseling_information",
    "patient_package_insert",
    "medication_guide",
    "description",
    "overdosage",
]

EXAMPLE_QUESTIONS = [
    {
        "label": "Metformin warning",
        "query": "Can metformin cause dangerous acid buildup in the blood?",
        "drug": "metformin",
        "family": "warnings_and_precautions",
    },
    {
        "label": "RxNorm identity",
        "query": "Is Eliquis the same as apixaban?",
        "drug": NO_DRUG_FILTER,
        "family": NO_FAMILY_FILTER,
    },
    {
        "label": "Mixed-source answer",
        "query": "Is Eliquis the same as apixaban and can it prevent stroke?",
        "drug": NO_DRUG_FILTER,
        "family": NO_FAMILY_FILTER,
    },
    {
        "label": "Generic-name lookup",
        "query": "What is the generic name of Glucophage?",
        "drug": NO_DRUG_FILTER,
        "family": NO_FAMILY_FILTER,
    },
    {
        "label": "Apixaban abstention",
        "query": "Does apixaban treat bacterial infections?",
        "drug": "apixaban",
        "family": NO_FAMILY_FILTER,
    },
    {
        "label": "Interaction routing",
        "query": "Can Eliquis be taken with aspirin?",
        "drug": NO_DRUG_FILTER,
        "family": NO_FAMILY_FILTER,
    },
]


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            opacity: 0.78;
            margin-bottom: 1.25rem;
        }

        .summary-card {
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.035);
            min-height: 98px;
        }

        .summary-label {
            font-size: 0.80rem;
            opacity: 0.72;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .summary-value {
            font-size: 1.02rem;
            font-weight: 650;
        }

        .status-pill {
            display: inline-block;
            padding: 0.36rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .pill-success {
            background: rgba(34, 197, 94, 0.16);
            color: rgb(74, 222, 128);
            border: 1px solid rgba(74, 222, 128, 0.38);
        }

        .pill-warning {
            background: rgba(234, 179, 8, 0.16);
            color: rgb(250, 204, 21);
            border: 1px solid rgba(250, 204, 21, 0.38);
        }

        .pill-info {
            background: rgba(59, 130, 246, 0.16);
            color: rgb(96, 165, 250);
            border: 1px solid rgba(96, 165, 250, 0.38);
        }

        .pill-neutral {
            background: rgba(148, 163, 184, 0.16);
            color: rgb(203, 213, 225);
            border: 1px solid rgba(203, 213, 225, 0.25);
        }

        .answer-card {
            padding: 1.15rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.035);
            margin-top: 0.65rem;
            margin-bottom: 1rem;
        }

        .answer-text {
            font-size: 1.08rem;
            line-height: 1.65;
            margin-top: 0.6rem;
            margin-bottom: 0.65rem;
        }

        .filter-row {
            opacity: 0.82;
            font-size: 0.87rem;
            margin-top: 0.45rem;
            margin-bottom: 0.75rem;
        }

        .citation-chip {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.16);
            color: rgb(165, 180, 252);
            border: 1px solid rgba(165, 180, 252, 0.34);
            font-size: 0.80rem;
            font-weight: 650;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .evidence-card {
            padding: 0.95rem 1rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(255,255,255,0.11);
            background: rgba(255,255,255,0.03);
            margin-bottom: 0.85rem;
        }

        .evidence-heading {
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .muted-small {
            font-size: 0.84rem;
            opacity: 0.74;
        }

        .sidebar-card {
            padding: 0.75rem 0.8rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            margin-bottom: 0.65rem;
        }

        .corpus-card {
            padding: 0.82rem 0.85rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            margin-bottom: 0.7rem;
            font-size: 0.88rem;
            line-height: 1.6;
        }

                .route-card {
            padding: 0.95rem 1rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(255,255,255,0.11);
            background: rgba(255,255,255,0.03);
            margin-top: 0.65rem;
            margin-bottom: 0.85rem;
            line-height: 1.6;
        }

        .identity-card {
            padding: 0.95rem 1rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(147, 197, 253, 0.28);
            background: rgba(59, 130, 246, 0.07);
            margin-bottom: 0.85rem;
        }

        .identity-heading {
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .citation-legend {
            font-size: 0.84rem;
            opacity: 0.78;
            margin-top: 0.3rem;
            margin-bottom: 0.6rem;
        }

        .corpus-card strong {
            font-weight: 700;
        }

        .recent-query {
            padding: 0.62rem 0.7rem;
            border-radius: 0.72rem;
            border: 1px solid rgba(255,255,255,0.09);
            background: rgba(255,255,255,0.025);
            margin-bottom: 0.45rem;
            font-size: 0.84rem;
        }

        .section-gap {
            margin-top: 1.25rem;
        }

        .diag-box {
            padding: 0.9rem 1rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.028);
            margin-top: 0.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Session state
# =============================================================================

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "debug_query_input" not in st.session_state:
    st.session_state.debug_query_input = ""

if "selected_drug" not in st.session_state:
    st.session_state.selected_drug = NO_DRUG_FILTER

if "selected_family" not in st.session_state:
    st.session_state.selected_family = NO_FAMILY_FILTER

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "include_evidence" not in st.session_state:
    st.session_state.include_evidence = True

if "include_diagnostics" not in st.session_state:
    st.session_state.include_diagnostics = True

if "last_answer_response" not in st.session_state:
    st.session_state.last_answer_response = None

if "last_debug_response" not in st.session_state:
    st.session_state.last_debug_response = None

if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []


# =============================================================================
# Utility functions
# =============================================================================

def normalize_api_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def optional_drug(value: str) -> str | None:
    return None if value == NO_DRUG_FILTER else value


def optional_family(value: str) -> str | None:
    return None if value == NO_FAMILY_FILTER else value


def get_json_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
        return str(payload)
    except Exception:
        return response.text


def safe_set_selected_option(
    *,
    key: str,
    allowed_options: list[str],
    fallback: str,
) -> None:
    current_value = st.session_state.get(key, fallback)

    if current_value not in allowed_options:
        st.session_state[key] = fallback


def humanize_family_name(value: str) -> str:
    return value.replace("_", " ").title()


def humanize_source_name(value: str | None) -> str:
    if value == "dailymed_label":
        return "DailyMed Label"
    if value == "rxnorm_identity":
        return "RxNorm Identity"
    if value == "multi_source_composed":
        return "Multi-Source Composed"
    return value or "Not routed"


def comma_join_or_dash(values: list[str] | None) -> str:
    if not values:
        return "—"
    return ", ".join(values)


def format_optional_timestamp(value: str | None) -> str:
    if not value:
        return "Not recorded"

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


# =============================================================================
# Metadata / health API calls
# =============================================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_base_url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{api_base_url}/health")
            response.raise_for_status()

        return {
            "ok": True,
            "payload": response.json(),
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@st.cache_data(ttl=30, show_spinner=False)
def fetch_drugs(api_base_url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{api_base_url}/drugs")
            response.raise_for_status()

        payload = response.json()

        dynamic_options = [
            item["concept_name"]
            for item in payload.get("drugs", [])
            if item.get("concept_name")
        ]

        return {
            "ok": True,
            "payload": payload,
            "options": [NO_DRUG_FILTER, *dynamic_options],
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "payload": None,
            "options": FALLBACK_DRUG_OPTIONS,
        }


@st.cache_data(ttl=30, show_spinner=False)
def fetch_families(api_base_url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{api_base_url}/families")
            response.raise_for_status()

        payload = response.json()

        dynamic_options = [
            item["retrieval_family"]
            for item in payload.get("families", [])
            if item.get("retrieval_family")
        ]

        return {
            "ok": True,
            "payload": payload,
            "options": [NO_FAMILY_FILTER, *dynamic_options],
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "payload": None,
            "options": FALLBACK_FAMILY_OPTIONS,
        }


@st.cache_data(ttl=30, show_spinner=False)
def fetch_corpus_stats(api_base_url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{api_base_url}/corpus/stats")
            response.raise_for_status()

        return {
            "ok": True,
            "payload": response.json(),
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }


# =============================================================================
# QA / retrieval API calls
# =============================================================================

def call_answer_api(
    *,
    api_base_url: str,
    query: str,
    drug: str | None,
    family: str | None,
    top_k: int,
    include_evidence: bool,
    include_diagnostics: bool,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "drug": drug,
        "family": family,
        "top_k": top_k,
        "include_evidence": include_evidence,
        "include_diagnostics": include_diagnostics,
    }

    started = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{api_base_url}/qa/answer",
            json=payload,
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if response.status_code >= 400:
        raise RuntimeError(
            f"API error {response.status_code}: {get_json_error_detail(response)}"
        )

    result = response.json()
    result["_client_latency_ms"] = latency_ms
    return result


def call_retrieval_debug_api(
    *,
    api_base_url: str,
    query: str,
    drug: str | None,
    family: str | None,
    top_k: int,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "drug": drug,
        "family": family,
        "top_k": top_k,
    }

    started = time.perf_counter()

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{api_base_url}/retrieval/debug",
            json=payload,
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if response.status_code >= 400:
        raise RuntimeError(
            f"API error {response.status_code}: {get_json_error_detail(response)}"
        )

    result = response.json()
    result["_client_latency_ms"] = latency_ms
    return result


# =============================================================================
# Presentation helpers
# =============================================================================

def add_recent_query(
    *,
    query: str,
    answer_status: str,
    drug: str | None,
) -> None:
    item = {
        "query": query,
        "status": answer_status,
        "drug": drug or NO_DRUG_FILTER,
    }

    st.session_state.recent_queries.insert(0, item)
    st.session_state.recent_queries = st.session_state.recent_queries[:6]

def load_example_question(
    *,
    query: str,
    drug: str,
    family: str,
    drug_options: list[str],
    family_options: list[str],
) -> None:
    """
    Load an example prompt and synchronize sidebar filters safely.

    This runs as a Streamlit widget callback, so session-state updates happen
    before the selectbox widgets are re-instantiated on the next rerun.
    """
    st.session_state.query_input = query

    st.session_state.selected_drug = (
        drug
        if drug in drug_options
        else NO_DRUG_FILTER
    )

    st.session_state.selected_family = (
        family
        if family in family_options
        else NO_FAMILY_FILTER
    )


def pill(label: str, class_name: str) -> str:
    return f'<span class="status-pill {class_name}">{label}</span>'


def render_sidebar_health(api_base_url: str) -> None:
    st.sidebar.markdown("### Backend health")

    health = fetch_health(api_base_url)

    if not health["ok"]:
        st.sidebar.error("API unavailable")
        st.sidebar.caption(health["error"])
        return

    payload = health["payload"]

    overall = payload.get("status", "unknown")
    if overall == "ok":
        st.sidebar.success("Backend healthy")
    else:
        st.sidebar.warning(f"Backend status: {overall}")

    postgres = payload.get("postgres", {})
    qdrant = payload.get("qdrant", {})
    llm = payload.get("llm", {})

    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <strong>Postgres:</strong> {postgres.get("status", "unknown")}<br/>
            <strong>Qdrant:</strong> {qdrant.get("status", "unknown")}<br/>
            <strong>LLM:</strong> {llm.get("status", "unknown")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Detailed health payload"):
        st.json(payload)


def render_sidebar_corpus_stats(api_base_url: str) -> None:
    st.sidebar.markdown("### Corpus snapshot")

    stats_result = fetch_corpus_stats(api_base_url)

    if not stats_result["ok"]:
        st.sidebar.warning("Corpus stats unavailable")
        st.sidebar.caption(stats_result["error"])
        return

    payload = stats_result["payload"]
    latest_build = payload.get("latest_build") or {}

    built_at_display = format_optional_timestamp(
        latest_build.get("built_at")
    )

    st.sidebar.markdown(
        f"""
        <div class="corpus-card">
            <strong>Drugs:</strong> {payload.get("drug_count", "—")}<br/>
            <strong>Sections:</strong> {payload.get("section_count", "—")}<br/>
            <strong>Chunks:</strong> {payload.get("chunk_count", "—")}<br/>
            <strong>Qdrant points:</strong> {payload.get("qdrant_point_count", "—")}<br/>
            <strong>Latest build:</strong> {built_at_display}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Detailed corpus stats"):
        st.json(payload)


def render_filter_metadata_notes(
    *,
    drugs_result: dict[str, Any],
    families_result: dict[str, Any],
) -> None:
    if not drugs_result["ok"]:
        st.sidebar.caption(
            "Using fallback drug filters because `/drugs` is unavailable."
        )

    if not families_result["ok"]:
        st.sidebar.caption(
            "Using fallback family filters because `/families` is unavailable."
        )


def render_recent_queries() -> None:
    st.sidebar.markdown("### Recent queries")

    if not st.session_state.recent_queries:
        st.sidebar.caption("No questions asked yet.")
        return

    for item in st.session_state.recent_queries:
        status_label = item["status"]
        status_color = (
            "🟢" if status_label == "answered" else "🟡"
        )

        st.sidebar.markdown(
            f"""
            <div class="recent-query">
                <strong>{status_color} {status_label}</strong><br/>
                <span>{item["query"]}</span><br/>
                <span style="opacity:0.72;">Drug: {item["drug"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_items(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        st.caption("No evidence snippets returned.")
        return

    for item in evidence:
        with st.expander(
            f"[{item['evidence_id']}] {item['heading']}",
            expanded=False,
        ):
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-heading">
                        {item['heading']}
                    </div>
                    <div class="muted-small">
                        Drug: {item['drug']} · Family: {item['retrieval_family']}<br/>
                        Source: {item['source_label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Hybrid score", f"{item['hybrid_score']:.6f}")
            col2.metric(
                "Lexical rank",
                str(item.get("lexical_rank"))
                if item.get("lexical_rank") is not None
                else "—",
            )
            col3.metric(
                "Dense rank",
                str(item.get("dense_rank"))
                if item.get("dense_rank") is not None
                else "—",
            )

            st.markdown("**Evidence text**")
            st.write(item["chunk_text"])


def render_identity_evidence_items(
    identity_evidence: list[dict[str, Any]],
) -> None:
    if not identity_evidence:
        st.caption("No RxNorm identity evidence returned.")
        return

    for item in identity_evidence:
        selected_candidate = item.get("selected_candidate") or {}
        related_ingredients = item.get("related_ingredients") or []
        related_brands = item.get("related_brands") or []

        selected_name = selected_candidate.get("name") or "—"
        selected_tty = selected_candidate.get("tty") or "—"
        selected_rxcui = selected_candidate.get("rxcui") or "—"

        ingredient_names = [
            concept.get("name")
            for concept in related_ingredients
            if concept.get("name")
        ]

        brand_names = [
            concept.get("name")
            for concept in related_brands
            if concept.get("name")
        ]

        with st.expander(
            f"[{item['evidence_id']}] RxNorm identity evidence for {item['term']}",
            expanded=False,
        ):
            st.markdown(
                f"""
                <div class="identity-card">
                    <div class="identity-heading">
                        {item['term']}
                    </div>
                    <div class="muted-small">
                        Resolution status: {item.get("resolution_status", "—")}<br/>
                        Selected concept: {selected_name} · {selected_tty} · RxCUI {selected_rxcui}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Related ingredients**")
                st.write(comma_join_or_dash(ingredient_names))

            with col2:
                st.markdown("**Related brands**")
                st.write(comma_join_or_dash(brand_names))

            st.markdown("**Identity evidence summary**")
            st.write(item.get("summary") or "—")


def render_source_route_summary(
    *,
    payload: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> None:
    planned_source = payload.get("planned_source")
    resolved_drug = payload.get("resolved_drug")
    planned_family = payload.get("planned_family")

    source_plan = (
        diagnostics.get("source_plan")
        if diagnostics is not None
        else None
    )

    family_plan = (
        diagnostics.get("family_plan")
        if diagnostics is not None
        else None
    )

    mixed_composition = (
        diagnostics.get("mixed_source_composition")
        if diagnostics is not None
        else None
    )

    st.markdown(
        f"""
        <div class="route-card">
            <strong>Planned source:</strong> {humanize_source_name(planned_source)}<br/>
            <strong>Resolved drug:</strong> {resolved_drug or "—"}<br/>
            <strong>Planned family:</strong> {planned_family or "—"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_plan:
        st.markdown("#### Source routing")
        st.markdown(
            f"""
            <div class="diag-box">
                <strong>Status:</strong> {source_plan.get("status", "—")}<br/>
                <strong>Selected source:</strong> {humanize_source_name(source_plan.get("selected_source"))}<br/>
                <strong>Intent:</strong> {source_plan.get("intent") or "—"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if family_plan:
        st.markdown("#### Retrieval-family planning")
        st.markdown(
            f"""
            <div class="diag-box">
                <strong>Status:</strong> {family_plan.get("status", "—")}<br/>
                <strong>Intent:</strong> {family_plan.get("intent") or "—"}<br/>
                <strong>Planned family:</strong> {family_plan.get("planned_family") or "—"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if mixed_composition:
        st.markdown("#### Mixed-source composition")
        st.markdown(
            f"""
            <div class="diag-box">
                <strong>Status:</strong> {mixed_composition.get("status", "—")}<br/><br/>
                <strong>Identity subquery:</strong><br/>
                {mixed_composition.get("identity_query") or "—"}<br/><br/>
                <strong>Clinical subquery:</strong><br/>
                {mixed_composition.get("clinical_query") or "—"}<br/><br/>
                <strong>Identity intent:</strong> {mixed_composition.get("identity_intent") or "—"}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_answer_payload(payload: dict[str, Any]) -> None:
    result = payload["result"]
    evidence = payload.get("evidence") or []
    identity_evidence = payload.get("identity_evidence") or []
    diagnostics = payload.get("diagnostics")
    latency_ms = payload.get("_client_latency_ms")

    st.markdown("## Grounded answer")

    status_value = result["status"]

    pill_html = []

    if status_value == "answered":
        pill_html.append(pill("Answered", "pill-success"))
    else:
        pill_html.append(pill("Insufficient Evidence", "pill-warning"))

    planned_source = payload.get("planned_source")
    if planned_source:
        pill_html.append(
            pill(
                humanize_source_name(planned_source),
                "pill-info",
            )
        )

    if diagnostics:
        verification = diagnostics.get("verification")
        if verification:
            pill_html.append(
                pill(
                    f"Verifier: {verification['verdict'].title()}",
                    "pill-info",
                )
            )

        if diagnostics.get("guardrail_triggered"):
            pill_html.append(
                pill("Guardrail Triggered", "pill-warning")
            )

    if latency_ms is not None:
        pill_html.append(
            pill(f"{latency_ms} ms", "pill-neutral")
        )

    st.markdown("".join(pill_html), unsafe_allow_html=True)

    active_drug = payload.get("drug") or NO_DRUG_FILTER
    active_family = payload.get("family") or NO_FAMILY_FILTER
    resolved_drug = payload.get("resolved_drug") or "—"
    planned_family = payload.get("planned_family") or "—"

    st.markdown(
        f"""
        <div class="filter-row">
            <strong>Requested filters:</strong>
            Drug = {active_drug} · Section family = {active_family}<br/>
            <strong>Resolved workflow:</strong>
            Drug = {resolved_drug} · Planned family = {planned_family}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-text">{result["answer"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    citations = result.get("citations", [])
    if citations:
        st.markdown("**Citations**")
        chips = "".join(
            f'<span class="citation-chip">{citation}</span>'
            for citation in citations
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown(
            """
            <div class="citation-legend">
                Citation legend: <strong>E*</strong> = DailyMed label evidence ·
                <strong>R*</strong> = RxNorm identity evidence
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Citations: None")

    st.info(result["safety_note"])

    with st.expander("Evidence summary", expanded=False):
        st.write(result["evidence_summary"])

    with st.expander("Routing and source plan", expanded=False):
        render_source_route_summary(
            payload=payload,
            diagnostics=diagnostics,
        )

    with st.expander(
        f"DailyMed label evidence ({len(evidence)})",
        expanded=False,
    ):
        render_evidence_items(evidence)

    with st.expander(
        f"RxNorm identity evidence ({len(identity_evidence)})",
        expanded=False,
    ):
        render_identity_evidence_items(identity_evidence)

    if diagnostics is not None:
        with st.expander("Diagnostics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Evidence count",
                diagnostics.get("evidence_count", 0),
            )

            col2.metric(
                "Proposed status",
                diagnostics.get("proposed_status") or "—",
            )

            col3.metric(
                "Verifier override",
                "Yes"
                if diagnostics.get("verification_overrode_answer")
                else "No",
            )

            col4.metric(
                "Guardrail",
                "Triggered"
                if diagnostics.get("guardrail_triggered")
                else "Not triggered",
            )

            verification = diagnostics.get("verification")
            if verification:
                st.markdown("### Verification")
                st.markdown(
                    f"""
                    <div class="diag-box">
                        <strong>Verdict:</strong> {verification["verdict"]}<br/><br/>
                        <strong>Rationale:</strong> {verification["rationale"]}<br/><br/>
                        <strong>Evidence used:</strong> {", ".join(verification["cited_evidence_used"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if diagnostics.get("guardrail_triggered"):
                st.markdown("### Guardrail")
                st.warning(
                    diagnostics.get("guardrail_reason")
                    or "A guardrail was triggered."
                )

            with st.expander("Raw diagnostics JSON"):
                st.json(diagnostics)

            mixed_composition = diagnostics.get(
                "mixed_source_composition"
            )

            if mixed_composition:
                st.markdown("### Mixed-source composition")
                st.markdown(
                    f"""
                    <div class="diag-box">
                        <strong>Status:</strong> {mixed_composition.get("status", "—")}<br/><br/>
                        <strong>Identity subquery:</strong><br/>
                        {mixed_composition.get("identity_query") or "—"}<br/><br/>
                        <strong>Clinical subquery:</strong><br/>
                        {mixed_composition.get("clinical_query") or "—"}<br/><br/>
                        <strong>Identity intent:</strong>
                        {mixed_composition.get("identity_intent") or "—"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_retrieval_debug_payload(payload: dict[str, Any]) -> None:
    evidence = payload.get("evidence") or []
    latency_ms = payload.get("_client_latency_ms")

    st.markdown("## Retrieval debug")

    meta_html = pill(
        f"{len(evidence)} evidence item(s)",
        "pill-info",
    )

    if latency_ms is not None:
        meta_html += pill(f"{latency_ms} ms", "pill-neutral")

    st.markdown(meta_html, unsafe_allow_html=True)

    st.caption(
        "This view inspects the compact evidence pack returned by the retrieval layer only. No LLM answer generation is called."
    )

    render_evidence_items(evidence)


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.markdown("## MedLabelIQ Controls")

api_base_url = normalize_api_base_url(
    st.sidebar.text_input(
        "FastAPI base URL",
        value=DEFAULT_API_BASE_URL,
    )
)

if st.sidebar.button("Refresh backend metadata", use_container_width=True):
    fetch_health.clear()
    fetch_drugs.clear()
    fetch_families.clear()
    fetch_corpus_stats.clear()
    st.rerun()

render_sidebar_health(api_base_url)
render_sidebar_corpus_stats(api_base_url)

drugs_result = fetch_drugs(api_base_url)
families_result = fetch_families(api_base_url)

drug_options = drugs_result["options"]
family_options = families_result["options"]

safe_set_selected_option(
    key="selected_drug",
    allowed_options=drug_options,
    fallback=NO_DRUG_FILTER,
)

safe_set_selected_option(
    key="selected_family",
    allowed_options=family_options,
    fallback=NO_FAMILY_FILTER,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Retrieval filters")

st.sidebar.selectbox(
    "Drug concept",
    drug_options,
    key="selected_drug",
)

st.sidebar.selectbox(
    "Section family",
    family_options,
    key="selected_family",
    format_func=(
        lambda value: value
        if value == NO_FAMILY_FILTER
        else humanize_family_name(value)
    ),
)

render_filter_metadata_notes(
    drugs_result=drugs_result,
    families_result=families_result,
)

st.sidebar.slider(
    "Evidence top-k",
    min_value=1,
    max_value=10,
    key="top_k",
)

st.sidebar.checkbox(
    "Include evidence",
    key="include_evidence",
)

st.sidebar.checkbox(
    "Include diagnostics",
    key="include_diagnostics",
)

st.sidebar.markdown("---")
render_recent_queries()


# =============================================================================
# Main header
# =============================================================================

st.markdown(
    '<div class="hero-title">💊 MedLabelIQ</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        Grounded medication-label QA with DailyMed label evidence, RxNorm drug-identity reasoning,
        source-aware orchestration, mixed-source synthesis, verification, and deterministic abstention guardrails.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="summary-card">
            <div class="summary-label">Knowledge sources</div>
            <div class="summary-value">DailyMed SPL + RxNorm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="summary-card">
            <div class="summary-label">Orchestration</div>
            <div class="summary-value">Routing + Decomposition + Synthesis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="summary-card">
            <div class="summary-label">Safety layer</div>
            <div class="summary-value">Verifier + Guardrails + Abstention</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)


# =============================================================================
# Tabs
# =============================================================================

ask_tab, debug_tab = st.tabs(
    [
        "Ask MedLabelIQ",
        "Retrieval Debug",
    ]
)


# =============================================================================
# Ask tab
# =============================================================================

with ask_tab:
    st.markdown("### Try an example")

    examples_per_row = 3

    for row_start in range(0, len(EXAMPLE_QUESTIONS), examples_per_row):
        row_examples = EXAMPLE_QUESTIONS[
            row_start : row_start + examples_per_row
        ]
        example_cols = st.columns(examples_per_row)

        for offset, example in enumerate(row_examples):
            idx = row_start + offset

            with example_cols[offset]:
                st.button(
                    example["label"],
                    key=f"example_{idx}",
                    use_container_width=True,
                    on_click=load_example_question,
                    kwargs={
                        "query": example["query"],
                        "drug": example["drug"],
                        "family": example["family"],
                        "drug_options": drug_options,
                        "family_options": family_options,
                    },
                )

    st.markdown("### Ask a medication-label question")

    with st.form("qa_form"):
        query = st.text_area(
            "Medication-label question",
            key="query_input",
            height=130,
            placeholder=(
                "Example: Can metformin cause dangerous acid buildup in the blood?"
            ),
        )

        submitted = st.form_submit_button(
            "Ask MedLabelIQ",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not query.strip():
            st.error("Please enter a medication-label question.")
        else:
            try:
                with st.spinner("Generating grounded answer..."):
                    answer_payload = call_answer_api(
                        api_base_url=api_base_url,
                        query=query.strip(),
                        drug=optional_drug(st.session_state.selected_drug),
                        family=optional_family(st.session_state.selected_family),
                        top_k=st.session_state.top_k,
                        include_evidence=st.session_state.include_evidence,
                        include_diagnostics=st.session_state.include_diagnostics,
                    )

                st.session_state.last_answer_response = answer_payload

                add_recent_query(
                    query=query.strip(),
                    answer_status=answer_payload["result"]["status"],
                    drug=optional_drug(st.session_state.selected_drug),
                )

            except Exception as exc:
                st.error(
                    f"Answer request failed: {type(exc).__name__}: {exc}"
                )

    if st.session_state.last_answer_response is not None:
        render_answer_payload(st.session_state.last_answer_response)


# =============================================================================
# Retrieval debug tab
# =============================================================================

with debug_tab:
    st.markdown("### Retrieval-only debugging")
    st.caption(
        "Use this tab to inspect what the hybrid retriever sends into the answer pipeline."
    )

    with st.form("retrieval_debug_form"):
        debug_query = st.text_area(
            "Retrieval query",
            key="debug_query_input",
            height=110,
            placeholder="Example: dangerous acid buildup in the blood",
        )

        debug_top_k = st.slider(
            "Debug evidence top-k",
            min_value=1,
            max_value=10,
            value=5,
            key="debug_top_k",
        )

        debug_submitted = st.form_submit_button(
            "Run Retrieval Debug",
            use_container_width=True,
        )

    if debug_submitted:
        if not debug_query.strip():
            st.error("Please enter a retrieval query.")
        else:
            try:
                with st.spinner("Running retrieval-only debug..."):
                    debug_payload = call_retrieval_debug_api(
                        api_base_url=api_base_url,
                        query=debug_query.strip(),
                        drug=optional_drug(st.session_state.selected_drug),
                        family=optional_family(st.session_state.selected_family),
                        top_k=debug_top_k,
                    )

                st.session_state.last_debug_response = debug_payload

            except Exception as exc:
                st.error(
                    f"Retrieval debug failed: {type(exc).__name__}: {exc}"
                )

    if st.session_state.last_debug_response is not None:
        render_retrieval_debug_payload(
            st.session_state.last_debug_response
        )