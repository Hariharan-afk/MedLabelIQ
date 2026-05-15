from medlabeliq.ingestion.fetch_label_manifest import DrugSeed, score_candidate


def test_score_candidate_rewards_preferred_tokens() -> None:
    seed = DrugSeed(
        concept_name="metformin",
        query_name="metformin",
        name_type="generic",
        target_profile="single-ingredient oral tablet",
        prefer_title_contains=["METFORMIN", "TABLET"],
        exclude_title_contains=["SOLUTION"],
        set_id=None,
        locked_title=None,
        notes="",
    )
    score, preferred, excluded = score_candidate(seed, "METFORMIN HYDROCHLORIDE TABLET")
    assert score > 0
    assert preferred == ["METFORMIN", "TABLET"]
    assert excluded == []


def test_score_candidate_penalizes_excluded_tokens() -> None:
    seed = DrugSeed(
        concept_name="acetaminophen",
        query_name="acetaminophen",
        name_type="generic",
        target_profile="oral tablet",
        prefer_title_contains=["ACETAMINOPHEN", "TABLET"],
        exclude_title_contains=["LIQUID"],
        set_id=None,
        locked_title=None,
        notes="",
    )
    score, preferred, excluded = score_candidate(seed, "ACETAMINOPHEN LIQUID")
    assert score < 0
    assert preferred == ["ACETAMINOPHEN"]
    assert excluded == ["LIQUID"]
