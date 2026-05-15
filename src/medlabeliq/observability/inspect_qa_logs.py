from __future__ import annotations

from medlabeliq.db.connection import get_connection


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total_requests
                FROM qa_request_log;
                """
            )
            total_requests = cur.fetchone()["total_requests"]

            cur.execute(
                """
                SELECT final_status, COUNT(*) AS count
                FROM qa_request_log
                GROUP BY final_status
                ORDER BY final_status;
                """
            )
            status_rows = cur.fetchall()

            cur.execute(
                """
                SELECT COUNT(*) AS guardrail_count
                FROM qa_request_log
                WHERE guardrail_triggered = TRUE;
                """
            )
            guardrail_count = cur.fetchone()["guardrail_count"]

            cur.execute(
                """
                SELECT COUNT(*) AS verifier_override_count
                FROM qa_request_log
                WHERE verification_overrode_answer = TRUE;
                """
            )
            verifier_override_count = cur.fetchone()["verifier_override_count"]

            cur.execute(
                """
                SELECT
                    ROUND(AVG(api_latency_ms)::numeric, 2) AS avg_latency_ms,
                    ROUND(MIN(api_latency_ms)::numeric, 2) AS min_latency_ms,
                    ROUND(MAX(api_latency_ms)::numeric, 2) AS max_latency_ms
                FROM qa_request_log;
                """
            )
            latency_row = cur.fetchone()

            cur.execute(
                """
                SELECT
                    created_at,
                    final_status,
                    proposed_status,
                    verification_verdict,
                    guardrail_triggered,
                    evidence_count,
                    ROUND(api_latency_ms::numeric, 2) AS api_latency_ms,
                    query_text
                FROM qa_request_log
                ORDER BY created_at DESC
                LIMIT 10;
                """
            )
            recent_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    retrieval_family,
                    COUNT(*) AS evidence_rows
                FROM qa_evidence_log
                GROUP BY retrieval_family
                ORDER BY evidence_rows DESC, retrieval_family ASC
                LIMIT 10;
                """
            )
            family_rows = cur.fetchall()

    print("\nQA OBSERVABILITY SUMMARY")
    print("=" * 80)
    print(f"Total logged QA requests: {total_requests}")

    print("\nFinal answer status counts:")
    if status_rows:
        for row in status_rows:
            print(f"  - {row['final_status']}: {row['count']}")
    else:
        print("  - No rows yet.")

    print(f"\nGuardrail-triggered requests: {guardrail_count}")
    print(f"Verifier-overridden requests: {verifier_override_count}")

    print("\nAPI latency:")
    print(f"  - Average: {latency_row['avg_latency_ms']} ms")
    print(f"  - Minimum: {latency_row['min_latency_ms']} ms")
    print(f"  - Maximum: {latency_row['max_latency_ms']} ms")

    print("\nMost common logged evidence families:")
    if family_rows:
        for row in family_rows:
            print(
                f"  - {row['retrieval_family']}: "
                f"{row['evidence_rows']} evidence row(s)"
            )
    else:
        print("  - No evidence rows yet.")

    print("\nRecent QA requests:")
    if recent_rows:
        for row in recent_rows:
            print("-" * 80)
            print(f"Timestamp: {row['created_at']}")
            print(f"Final status: {row['final_status']}")
            print(f"Proposed status: {row['proposed_status']}")
            print(f"Verifier verdict: {row['verification_verdict']}")
            print(f"Guardrail triggered: {row['guardrail_triggered']}")
            print(f"Evidence count: {row['evidence_count']}")
            print(f"API latency: {row['api_latency_ms']} ms")
            print(f"Query: {row['query_text']}")
    else:
        print("  - No rows yet.")


if __name__ == "__main__":
    main()