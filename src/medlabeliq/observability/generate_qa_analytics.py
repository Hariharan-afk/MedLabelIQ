from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from medlabeliq.config.settings import settings
from medlabeliq.db.connection import get_connection


# =============================================================================
# Output paths
# =============================================================================

DEFAULT_CSV_DIR = (
    settings.project_root
    / "data"
    / "interim"
    / "qa_analytics"
)

DEFAULT_PLOT_DIR = (
    settings.project_root
    / "outputs"
    / "qa_analytics"
)


# =============================================================================
# Database queries
# =============================================================================

REQUEST_LOG_QUERY = """
SELECT
    request_log_id::text AS request_log_id,
    created_at,
    query_text,
    drug_filter,
    family_filter,
    top_k,
    include_evidence,
    include_diagnostics,
    final_status,
    final_answer,
    final_citations,
    final_evidence_summary,
    safety_note,
    proposed_status,
    verification_enabled,
    verification_verdict,
    verification_rationale,
    verification_evidence_used,
    verification_overrode_answer,
    guardrail_triggered,
    guardrail_reason,
    evidence_count,
    api_latency_ms
FROM qa_request_log
ORDER BY created_at ASC;
"""


EVIDENCE_LOG_QUERY = """
SELECT
    request_log_id::text AS request_log_id,
    evidence_id,
    evidence_position,
    chunk_id::text AS chunk_id,
    section_id::text AS section_id,
    concept_name,
    retrieval_family,
    canonical_section_name,
    nearest_canonical_section_name,
    heading,
    set_id,
    version_number,
    hybrid_score,
    lexical_rank,
    dense_rank,
    was_cited
FROM qa_evidence_log
ORDER BY request_log_id, evidence_position ASC;
"""


def load_query_as_dataframe(query: str) -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_analytics_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    requests_df = load_query_as_dataframe(REQUEST_LOG_QUERY)
    evidence_df = load_query_as_dataframe(EVIDENCE_LOG_QUERY)

    if not requests_df.empty:
        requests_df["created_at"] = pd.to_datetime(
            requests_df["created_at"],
            utc=True,
        )

        requests_df["request_date"] = (
            requests_df["created_at"]
            .dt.date
            .astype(str)
        )

        requests_df["status_transition"] = (
            requests_df["proposed_status"].fillna("none")
            + " → "
            + requests_df["final_status"].fillna("none")
        )

    return requests_df, evidence_df


# =============================================================================
# Summary metrics
# =============================================================================

def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def latency_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["metric", "value_ms"]
        )

    latency = requests_df["api_latency_ms"].astype(float)

    rows = [
        {"metric": "count", "value_ms": float(latency.count())},
        {"metric": "mean", "value_ms": float(latency.mean())},
        {"metric": "median", "value_ms": float(latency.median())},
        {"metric": "min", "value_ms": float(latency.min())},
        {"metric": "max", "value_ms": float(latency.max())},
        {"metric": "p95", "value_ms": float(latency.quantile(0.95))},
    ]

    return pd.DataFrame(rows)


def request_status_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["final_status", "request_count", "share"]
        )

    counts = (
        requests_df["final_status"]
        .value_counts(dropna=False)
        .rename_axis("final_status")
        .reset_index(name="request_count")
    )

    counts["share"] = (
        counts["request_count"] / len(requests_df)
    )

    return counts


def intervention_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    total = len(requests_df)

    if total == 0:
        return pd.DataFrame(
            columns=["intervention", "count", "rate"]
        )

    guardrail_count = int(
        requests_df["guardrail_triggered"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    verifier_override_count = int(
        requests_df["verification_overrode_answer"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    rows = [
        {
            "intervention": "guardrail_triggered",
            "count": guardrail_count,
            "rate": safe_rate(guardrail_count, total),
        },
        {
            "intervention": "verifier_overrode_answer",
            "count": verifier_override_count,
            "rate": safe_rate(verifier_override_count, total),
        },
    ]

    return pd.DataFrame(rows)


def verification_verdict_summary(
    requests_df: pd.DataFrame,
) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["verification_verdict", "request_count"]
        )

    filtered = requests_df[
        requests_df["verification_verdict"].notna()
    ]

    if filtered.empty:
        return pd.DataFrame(
            columns=["verification_verdict", "request_count"]
        )

    return (
        filtered["verification_verdict"]
        .value_counts()
        .rename_axis("verification_verdict")
        .reset_index(name="request_count")
    )


def transition_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["status_transition", "request_count"]
        )

    return (
        requests_df["status_transition"]
        .value_counts()
        .rename_axis("status_transition")
        .reset_index(name="request_count")
    )


def evidence_count_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["evidence_count", "request_count"]
        )

    return (
        requests_df["evidence_count"]
        .value_counts()
        .sort_index()
        .rename_axis("evidence_count")
        .reset_index(name="request_count")
    )


def requests_by_drug_filter(
    requests_df: pd.DataFrame,
) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["drug_filter", "request_count"]
        )

    temp = requests_df.copy()
    temp["drug_filter"] = temp["drug_filter"].fillna(
        "<NO_DRUG_FILTER>"
    )

    return (
        temp["drug_filter"]
        .value_counts()
        .rename_axis("drug_filter")
        .reset_index(name="request_count")
    )


def requests_by_family_filter(
    requests_df: pd.DataFrame,
) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["family_filter", "request_count"]
        )

    temp = requests_df.copy()
    temp["family_filter"] = temp["family_filter"].fillna(
        "<NO_FAMILY_FILTER>"
    )

    return (
        temp["family_filter"]
        .value_counts()
        .rename_axis("family_filter")
        .reset_index(name="request_count")
    )


def evidence_family_summary(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    if evidence_df.empty:
        return pd.DataFrame(
            columns=["retrieval_family", "evidence_rows"]
        )

    return (
        evidence_df["retrieval_family"]
        .value_counts()
        .rename_axis("retrieval_family")
        .reset_index(name="evidence_rows")
    )


def cited_evidence_family_summary(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    if evidence_df.empty:
        return pd.DataFrame(
            columns=["retrieval_family", "cited_evidence_rows"]
        )

    cited = evidence_df[
        evidence_df["was_cited"].fillna(False).astype(bool)
    ]

    if cited.empty:
        return pd.DataFrame(
            columns=["retrieval_family", "cited_evidence_rows"]
        )

    return (
        cited["retrieval_family"]
        .value_counts()
        .rename_axis("retrieval_family")
        .reset_index(name="cited_evidence_rows")
    )


def daily_volume_summary(requests_df: pd.DataFrame) -> pd.DataFrame:
    if requests_df.empty:
        return pd.DataFrame(
            columns=["request_date", "request_count"]
        )

    return (
        requests_df["request_date"]
        .value_counts()
        .sort_index()
        .rename_axis("request_date")
        .reset_index(name="request_count")
    )


# =============================================================================
# CSV exports
# =============================================================================

def export_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_all_csvs(
    *,
    csv_dir: Path,
    requests_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)

    export_dataframe(
        requests_df,
        csv_dir / "qa_request_log_export.csv",
    )

    export_dataframe(
        evidence_df,
        csv_dir / "qa_evidence_log_export.csv",
    )

    for name, df in summaries.items():
        export_dataframe(
            df,
            csv_dir / f"{name}.csv",
        )


# =============================================================================
# Plot helpers
# =============================================================================

def save_bar_plot(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    rotate_xticks: bool = False,
) -> None:
    if df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_latency_histogram(
    *,
    requests_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if requests_df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    latency = requests_df["api_latency_ms"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.hist(latency, bins=min(10, max(1, len(latency))))
    plt.title("QA API Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Request count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_daily_volume_line_plot(
    *,
    daily_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if daily_df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        daily_df["request_date"].astype(str),
        daily_df["request_count"].astype(float),
        marker="o",
    )
    plt.title("Daily Logged QA Request Volume")
    plt.xlabel("Date")
    plt.ylabel("Request count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_plots(
    *,
    plot_dir: Path,
    requests_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_bar_plot(
        df=summaries["request_status_summary"],
        x_col="final_status",
        y_col="request_count",
        title="Final QA Answer Status Counts",
        xlabel="Final status",
        ylabel="Request count",
        output_path=plot_dir / "final_answer_status_counts.png",
    )

    save_latency_histogram(
        requests_df=requests_df,
        output_path=plot_dir / "api_latency_distribution.png",
    )

    save_bar_plot(
        df=summaries["evidence_count_summary"],
        x_col="evidence_count",
        y_col="request_count",
        title="Evidence Count per QA Request",
        xlabel="Evidence count",
        ylabel="Request count",
        output_path=plot_dir / "evidence_count_distribution.png",
    )

    save_bar_plot(
        df=summaries["intervention_summary"],
        x_col="intervention",
        y_col="count",
        title="Guardrail and Verifier Intervention Counts",
        xlabel="Intervention type",
        ylabel="Count",
        output_path=plot_dir / "intervention_counts.png",
        rotate_xticks=True,
    )

    save_bar_plot(
        df=summaries["evidence_family_summary"].head(10),
        x_col="retrieval_family",
        y_col="evidence_rows",
        title="Top Logged Evidence Families",
        xlabel="Retrieval family",
        ylabel="Evidence rows",
        output_path=plot_dir / "top_evidence_families.png",
        rotate_xticks=True,
    )

    save_daily_volume_line_plot(
        daily_df=summaries["daily_volume_summary"],
        output_path=plot_dir / "daily_request_volume.png",
    )


# =============================================================================
# Terminal summary
# =============================================================================

def print_terminal_summary(
    *,
    requests_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    csv_dir: Path,
    plot_dir: Path,
) -> None:
    total_requests = len(requests_df)
    total_evidence_rows = len(evidence_df)

    status_df = summaries["request_status_summary"]
    latency_df = summaries["latency_summary"]
    intervention_df = summaries["intervention_summary"]
    verdict_df = summaries["verification_verdict_summary"]

    print("\nQA ANALYTICS REPORT")
    print("=" * 80)
    print(f"Total logged QA requests: {total_requests}")
    print(f"Total logged evidence rows: {total_evidence_rows}")

    print("\nFinal answer status:")
    if status_df.empty:
        print("  - No QA request rows available.")
    else:
        for _, row in status_df.iterrows():
            share_pct = float(row["share"]) * 100
            print(
                f"  - {row['final_status']}: "
                f"{int(row['request_count'])} "
                f"({share_pct:.1f}%)"
            )

    print("\nLatency summary:")
    if latency_df.empty:
        print("  - No latency rows available.")
    else:
        latency_map = {
            row["metric"]: row["value_ms"]
            for _, row in latency_df.iterrows()
        }

        print(f"  - Mean:   {latency_map.get('mean', 0):.2f} ms")
        print(f"  - Median: {latency_map.get('median', 0):.2f} ms")
        print(f"  - Min:    {latency_map.get('min', 0):.2f} ms")
        print(f"  - Max:    {latency_map.get('max', 0):.2f} ms")
        print(f"  - P95:    {latency_map.get('p95', 0):.2f} ms")

    print("\nInterventions:")
    if intervention_df.empty:
        print("  - No intervention rows available.")
    else:
        for _, row in intervention_df.iterrows():
            rate_pct = float(row["rate"]) * 100
            print(
                f"  - {row['intervention']}: "
                f"{int(row['count'])} "
                f"({rate_pct:.1f}%)"
            )

    print("\nVerifier verdicts:")
    if verdict_df.empty:
        print("  - No verifier verdicts logged.")
    else:
        for _, row in verdict_df.iterrows():
            print(
                f"  - {row['verification_verdict']}: "
                f"{int(row['request_count'])}"
            )

    print("\nTop logged evidence families:")
    family_df = summaries["evidence_family_summary"].head(10)

    if family_df.empty:
        print("  - No evidence rows logged.")
    else:
        for _, row in family_df.iterrows():
            print(
                f"  - {row['retrieval_family']}: "
                f"{int(row['evidence_rows'])}"
            )

    print("\nExport locations:")
    print(f"  - CSV reports: {csv_dir}")
    print(f"  - Plot outputs: {plot_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QA observability analytics CSVs and plots."
    )

    parser.add_argument(
        "--csv-dir",
        default=str(DEFAULT_CSV_DIR),
        help="Directory to write analytics CSV outputs.",
    )

    parser.add_argument(
        "--plot-dir",
        default=str(DEFAULT_PLOT_DIR),
        help="Directory to write analytics plot PNGs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_dir = Path(args.csv_dir)
    plot_dir = Path(args.plot_dir)

    requests_df, evidence_df = load_analytics_data()

    summaries = {
        "request_status_summary": request_status_summary(requests_df),
        "latency_summary": latency_summary(requests_df),
        "intervention_summary": intervention_summary(requests_df),
        "verification_verdict_summary": verification_verdict_summary(requests_df),
        "status_transition_summary": transition_summary(requests_df),
        "evidence_count_summary": evidence_count_summary(requests_df),
        "requests_by_drug_filter": requests_by_drug_filter(requests_df),
        "requests_by_family_filter": requests_by_family_filter(requests_df),
        "evidence_family_summary": evidence_family_summary(evidence_df),
        "cited_evidence_family_summary": cited_evidence_family_summary(evidence_df),
        "daily_volume_summary": daily_volume_summary(requests_df),
    }

    export_all_csvs(
        csv_dir=csv_dir,
        requests_df=requests_df,
        evidence_df=evidence_df,
        summaries=summaries,
    )

    generate_plots(
        plot_dir=plot_dir,
        requests_df=requests_df,
        summaries=summaries,
    )

    print_terminal_summary(
        requests_df=requests_df,
        evidence_df=evidence_df,
        summaries=summaries,
        csv_dir=csv_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()