from __future__ import annotations

from medlabeliq.retrieval.lexical_search import search_chunks


TEST_QUERIES = [
    {
        "name": "metformin lactic acidosis warning",
        "query": "lactic acidosis",
        "drug": "metformin",
        "family": "warnings_and_precautions",
    },
    {
        "name": "apixaban bleeding warning",
        "query": "bleeding risk",
        "drug": "apixaban",
        "family": "warnings_and_precautions",
    },
    {
        "name": "acetaminophen liver warning",
        "query": "liver warning",
        "drug": "acetaminophen",
        "family": None,
    },
    {
        "name": "sertraline suicidal thoughts",
        "query": "suicidal thoughts",
        "drug": "sertraline",
        "family": "boxed_warning",
    },
    {
        "name": "albuterol bronchospasm",
        "query": "bronchospasm",
        "drug": "albuterol",
        "family": "warnings_and_precautions",
    },
    {
    "name": "omeprazole acid-mediated GERD",
    "query": "acid-mediated GERD",
    "drug": "omeprazole",
    "family": "indications_and_usage",
    },
]


def main() -> None:
    errors: list[str] = []

    print("\nLEXICAL RETRIEVAL VALIDATION")
    print("=" * 80)

    for test in TEST_QUERIES:
        results = search_chunks(
            query=test["query"],
            concept_name=test["drug"],
            retrieval_family=test["family"],
            limit=5,
        )

        print("\n" + "-" * 80)
        print(f"Test: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Drug: {test['drug']}")
        print(f"Family: {test['family'] or 'None'}")
        print(f"Results: {len(results)}")

        if not results:
            errors.append(f"{test['name']}: returned no results.")
            continue

        top = results[0]

        print(f"Top drug: {top.concept_name}")
        print(f"Top family: {top.retrieval_family}")
        print(f"Top rank: {top.rank:.4f}")
        print(f"Top heading: {' > '.join(top.heading_path)}")
        print(f"Top preview: {top.chunk_text[:300].replace(chr(10), ' ')}...")

        if top.concept_name.lower() != test["drug"].lower():
            errors.append(
                f"{test['name']}: expected top drug {test['drug']}, "
                f"got {top.concept_name}."
            )

        if test["family"] is not None and top.retrieval_family != test["family"]:
            errors.append(
                f"{test['name']}: expected top family {test['family']}, "
                f"got {top.retrieval_family}."
            )

    if errors:
        print("\nVALIDATION STATUS: FAIL")
        print("=" * 80)
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: PASS")
    print("=" * 80)
    print("PostgreSQL lexical retrieval baseline is working.")


if __name__ == "__main__":
    main()