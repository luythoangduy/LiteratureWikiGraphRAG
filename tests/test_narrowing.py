from literature_wiki_graphrag.config import Settings
from literature_wiki_graphrag.narrowing import (
    apply_narrowing_suggestion,
    build_narrowing_report,
    load_narrowing_history,
    save_narrowing_history,
)
from literature_wiki_graphrag.schemas import (
    NarrowingDecision,
    NarrowingSuggestion,
    PaperCandidate,
    SearchConfig,
)


def make_candidate(index: int, theme: str, year: int = 2025) -> PaperCandidate:
    return PaperCandidate(
        id=f"paper:{index}",
        source="test",
        title=f"{theme} Paper {index}",
        year=year,
        abstract=f"This paper studies {theme}.",
        citation_count=index,
        open_access=index % 2 == 0,
        fields_of_study=[theme],
    )


def test_build_narrowing_report_detects_broad_results_and_themes() -> None:
    candidates = [
        make_candidate(index, "Graph Retrieval" if index < 12 else "Evaluation")
        for index in range(25)
    ]
    config = SearchConfig(topic="graph rag", target_min_papers=10, target_max_papers=20)

    report = build_narrowing_report(
        candidates,
        config,
        settings=Settings(GOOGLE_API_KEY=None, OPENROUTER_API_KEY=None),
    )

    assert report.is_too_broad is True
    assert report.result_count == 25
    assert report.threshold == 20
    assert report.summarizer == "heuristic"
    assert "25 papers" in report.overview
    assert report.themes[0].label in {"Graph Retrieval", "Evaluation"}
    assert report.suggestions


def test_apply_narrowing_suggestion_merges_keywords() -> None:
    config = SearchConfig(topic="graph rag", must_include_keywords=["rag"])
    suggestion = NarrowingSuggestion(
        label="Focus on Graph Retrieval",
        rationale="Theme is large.",
        config_updates={"must_include_keywords": ["graph retrieval"], "from_year": 2024},
    )

    updated = apply_narrowing_suggestion(config, suggestion)

    assert updated.from_year == 2024
    assert updated.must_include_keywords == ["rag", "graph retrieval"]


def test_narrowing_history_round_trips(tmp_path) -> None:
    history = [
        NarrowingDecision(
            selected_label="Recent work",
            rationale="Keep newer papers.",
            config_updates={"from_year": 2024},
        )
    ]

    path = save_narrowing_history(tmp_path, history)
    loaded = load_narrowing_history(tmp_path)

    assert path.name == "narrowing_history.json"
    assert loaded[0].selected_label == "Recent work"
    assert loaded[0].config_updates == {"from_year": 2024}

