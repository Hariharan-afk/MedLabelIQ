from __future__ import annotations

import pandas as pd

import medlabeliq.observability.generate_qa_analytics as module


def make_requests_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "planned_source": "dailymed_label",
                "source_plan_status": "routed_dailymed_label",
                "family_plan_status": "routed_single_family",
                "mixed_source_composition_status": "<NOT_COMPOSED>",
                "final_status": "answered",
                "api_latency_ms": 100.0,
                "identity_evidence_count": 0,
                "total_support_evidence_count": 3,
            },
            {
                "planned_source": "rxnorm_identity",
                "source_plan_status": "routed_rxnorm_identity",
                "family_plan_status": "no_route_detected",
                "mixed_source_composition_status": "<NOT_COMPOSED>",
                "final_status": "answered",
                "api_latency_ms": 50.0,
                "identity_evidence_count": 2,
                "total_support_evidence_count": 2,
            },
            {
                "planned_source": "multi_source_composed",
                "source_plan_status": "ambiguous_mixed_source",
                "family_plan_status": "routed_single_family",
                "mixed_source_composition_status": "composed_answered",
                "final_status": "answered",
                "api_latency_ms": 250.0,
                "identity_evidence_count": 2,
                "total_support_evidence_count": 7,
            },
        ]
    )


def test_planned_source_summary_counts_sources() -> None:
    summary = module.planned_source_summary(make_requests_df())

    counts = dict(
        zip(summary["planned_source"], summary["request_count"])
    )

    assert counts["dailymed_label"] == 1
    assert counts["rxnorm_identity"] == 1
    assert counts["multi_source_composed"] == 1


def test_mixed_source_composition_summary_counts_statuses() -> None:
    summary = module.mixed_source_composition_summary(
        make_requests_df()
    )

    counts = dict(
        zip(
            summary["mixed_source_composition_status"],
            summary["request_count"],
        )
    )

    assert counts["<NOT_COMPOSED>"] == 2
    assert counts["composed_answered"] == 1


def test_latency_by_planned_source_summary_computes_mean() -> None:
    summary = module.latency_by_planned_source_summary(
        make_requests_df()
    )

    rows = {
        row["planned_source"]: row
        for _, row in summary.iterrows()
    }

    assert float(rows["dailymed_label"]["mean_ms"]) == 100.0
    assert float(rows["rxnorm_identity"]["mean_ms"]) == 50.0
    assert float(rows["multi_source_composed"]["mean_ms"]) == 250.0