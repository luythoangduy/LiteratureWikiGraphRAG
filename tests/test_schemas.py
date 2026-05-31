from literature_wiki_graphrag.schemas import SearchConfig, SortMode


def test_search_config_defaults_to_balanced_review_set() -> None:
    config = SearchConfig(topic="graph rag")

    assert config.sort_mode == SortMode.BALANCED
    assert config.target_min_papers == 15
    assert config.target_max_papers == 20
