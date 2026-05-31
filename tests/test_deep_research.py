import pytest

from literature_wiki_graphrag.config import Settings
from literature_wiki_graphrag.deep_research import (
    build_deep_research_prompt,
    parse_deep_research_report,
    run_deep_research,
)
from literature_wiki_graphrag.schemas import PaperType, SearchConfig, SortMode


def test_build_deep_research_prompt() -> None:
    config = SearchConfig(
        topic="graph rag",
        from_year=2021,
        to_year=2025,
        target_min_papers=10,
        target_max_papers=15,
        sort_mode=SortMode.MOST_CITED,
        must_include_keywords=["llm", "graph"],
        exclude_keywords=["medical"],
        min_citation_count=5,
        paper_type=PaperType.PREPRINT,
        open_access_only=True,
    )

    prompt = build_deep_research_prompt(config)

    assert "graph rag" in prompt
    assert "2021 to 2025" in prompt
    assert "10 to 15 papers" in prompt
    assert "most cited" in prompt
    assert "Must include keywords" in prompt
    assert "llm, graph" in prompt
    assert "Exclude keywords" in prompt
    assert "medical" in prompt
    assert "Minimum citation count" in prompt
    assert "5" in prompt
    assert "preprint" in prompt
    assert "Open access only" in prompt


def test_parse_deep_research_report_valid() -> None:
    report_text = """
Some conversational text from the agent.
Here is the JSON report:
```json
{
  "papers": [
    {
      "title": "Deep Research Paper",
      "authors": ["Alice Smith", "Bob Jones"],
      "year": 2025,
      "arxiv_id": "2501.12345",
      "abstract": "This is a test abstract.",
      "url": "https://arxiv.org/abs/2501.12345",
      "pdf_url": "https://arxiv.org/pdf/2501.12345",
      "citation_count": 42,
      "venue": "arXiv",
      "fields_of_study": ["AI", "GraphRAG"]
    }
  ]
}
```
Some trailing thoughts.
"""
    candidates, errors = parse_deep_research_report(report_text)

    assert not errors
    assert len(candidates) == 1
    c = candidates[0]
    assert c.title == "Deep Research Paper"
    assert c.authors == ["Alice Smith", "Bob Jones"]
    assert c.year == 2025
    assert c.arxiv_id == "2501.12345"
    assert c.abstract == "This is a test abstract."
    assert str(c.url) == "https://arxiv.org/abs/2501.12345"
    assert str(c.pdf_url) == "https://arxiv.org/pdf/2501.12345"
    assert c.citation_count == 42
    assert c.venue == "arXiv"
    assert c.fields_of_study == ["AI", "GraphRAG"]
    assert c.source == "deep_research"


def test_parse_deep_research_report_invalid() -> None:
    # Test empty report
    candidates, errors = parse_deep_research_report("")
    assert len(candidates) == 0
    assert len(errors) == 1
    assert "Empty report text" in errors[0]

    # Test invalid json structure
    candidates, errors = parse_deep_research_report("No JSON here, only text.")
    assert len(candidates) == 0
    assert len(errors) == 1
    assert "Failed to parse JSON" in errors[0]

    # Test missing papers key
    candidates, errors = parse_deep_research_report("```json\n{}\n```")
    assert len(candidates) == 0
    assert len(errors) == 1
    assert "Missing 'papers' key" in errors[0]


def test_run_deep_research_missing_api_key() -> None:
    config = SearchConfig(topic="graph rag")
    settings = Settings()
    # Explicitly clear the key even if it was loaded from the env
    settings.google_api_key = None

    with pytest.raises(ValueError, match="Deep Research requires GOOGLE_API_KEY"):
        run_deep_research(config, settings=settings)
