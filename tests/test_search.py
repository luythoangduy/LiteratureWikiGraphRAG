import asyncio

import httpx

from literature_wiki_graphrag.schemas import PaperType, SearchConfig, SortMode
from literature_wiki_graphrag.search import (
    AcademicSearchError,
    AcademicSearchService,
    ArxivConnector,
    OpenAlexConnector,
    SemanticScholarConnector,
    abstract_from_inverted_index,
    arxiv_search_queries,
    build_default_search_service,
    dedupe_candidates,
    rank_candidates,
)

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <updated>2026-05-31T00:00:00Z</updated>
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <updated>2025-01-03T00:00:00Z</updated>
    <published>2025-01-02T00:00:00Z</published>
    <title>ReConPatch: Representation Learning for Industrial Anomaly Detection</title>
    <summary>Self-supervised representation learning for industrial anomaly detection.</summary>
    <author><name>A. Researcher</name></author>
    <category term="cs.CV" />
    <link href="http://arxiv.org/abs/2501.00001v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2501.00001v1"
          rel="related" type="application/pdf" />
    <arxiv:doi>10.48550/arXiv.2501.00001</arxiv:doi>
  </entry>
</feed>
"""


def test_default_search_service_is_arxiv_only() -> None:
    service = build_default_search_service()

    assert [connector.source for connector in service._connectors] == ["arxiv"]


def test_arxiv_connector_normalizes_atom_feed_and_query() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, text=ARXIV_FEED)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = ArxivConnector(client=client)
    result = asyncio.run(
        connector.search(
            SearchConfig(
                topic="idustrial anomaly detection, representation learning",
                from_year=2020,
                to_year=2026,
                paper_type=PaperType.PREPRINT,
            ),
            limit=5,
        )
    )
    asyncio.run(client.aclose())

    assert captured["search_query"] == (
        'all:industrial AND all:"anomaly detection" AND all:"representation learning"'
    )
    assert captured["max_results"] == "5"
    assert captured["sortBy"] == "relevance"

    candidate = result.candidates[0]
    assert candidate.source == "arxiv"
    assert candidate.title == "ReConPatch: Representation Learning for Industrial Anomaly Detection"
    assert candidate.authors == ["A. Researcher"]
    assert candidate.year == 2025
    assert candidate.arxiv_id == "2501.00001v1"
    assert candidate.doi == "10.48550/arxiv.2501.00001"
    assert candidate.open_access is True
    assert candidate.fields_of_study == ["cs.CV"]


def test_arxiv_search_queries_fix_common_industrial_typo() -> None:
    assert arxiv_search_queries("idustrial anomaly detection, representation learning")[0] == (
        'all:industrial AND all:"anomaly detection" AND all:"representation learning"'
    )


def test_openalex_connector_normalizes_work_and_query_params() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "Graph RAG for Literature Review",
                        "authorships": [
                            {"author": {"display_name": "Ada Researcher"}},
                        ],
                        "publication_year": 2025,
                        "publication_date": "2025-04-10",
                        "primary_location": {
                            "landing_page_url": "https://example.org/paper",
                            "source": {"display_name": "Journal of Graphs"},
                        },
                        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
                        "open_access": {"is_oa": True},
                        "doi": "https://doi.org/10.1234/GRAPH.1",
                        "cited_by_count": 42,
                        "abstract_inverted_index": {
                            "Graph": [0],
                            "RAG": [1],
                            "works": [2],
                        },
                        "concepts": [{"display_name": "Information retrieval"}],
                        "keywords": [{"display_name": "GraphRAG"}],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = OpenAlexConnector(client=client, mailto="researcher@example.com")
    config = SearchConfig(
        topic="graph rag",
        from_year=2020,
        to_year=2026,
        min_citation_count=5,
        paper_type=PaperType.JOURNAL_ARTICLE,
        open_access_only=True,
        sort_mode=SortMode.MOST_CITED,
    )

    result = asyncio.run(connector.search(config, limit=10))
    asyncio.run(client.aclose())

    assert captured["search"] == "graph rag"
    assert "from_publication_date:2020-01-01" in captured["filter"]
    assert "to_publication_date:2026-12-31" in captured["filter"]
    assert "cited_by_count:>4" in captured["filter"]
    assert "open_access.is_oa:true" in captured["filter"]
    assert "type:article" in captured["filter"]
    assert captured["sort"] == "cited_by_count:desc"
    assert captured["mailto"] == "researcher@example.com"

    candidate = result.candidates[0]
    assert candidate.source == "openalex"
    assert candidate.title == "Graph RAG for Literature Review"
    assert candidate.authors == ["Ada Researcher"]
    assert candidate.doi == "10.1234/graph.1"
    assert candidate.abstract == "Graph RAG works"
    assert candidate.citation_count == 42
    assert candidate.open_access is True


def test_openalex_connector_falls_back_when_original_topic_has_no_results() -> None:
    searched_topics: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        topic = request.url.params["search"]
        searched_topics.append(topic)
        if topic != "representation learning":
            return httpx.Response(200, json={"meta": {"count": 0}, "results": []})
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/Wfallback",
                        "display_name": "Representation Learning for Anomaly Detection",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = OpenAlexConnector(client=client)
    result = asyncio.run(
        connector.search(
            SearchConfig(topic="idustrial anomaly detection, representation learning"),
            limit=5,
        )
    )
    asyncio.run(client.aclose())

    assert searched_topics == [
        "idustrial anomaly detection, representation learning",
        "idustrial anomaly detection representation learning",
        "idustrial anomaly detection",
        "representation learning",
    ]
    assert result.raw_response["_lwgrag_fallback_search"]["used_topic"] == "representation learning"
    assert result.candidates[0].title == "Representation Learning for Anomaly Detection"


def test_semantic_scholar_connector_normalizes_paper_and_auth_header() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "abc123",
                        "externalIds": {"DOI": "10.48550/arXiv.2501.00001", "ArXiv": "2501.00001"},
                        "title": "Semantic Search for Papers",
                        "authors": [{"name": "B. Scholar"}],
                        "year": 2025,
                        "publicationDate": "2025-01-02",
                        "venue": "ACL",
                        "abstract": "A paper search abstract.",
                        "url": "https://www.semanticscholar.org/paper/abc123",
                        "citationCount": 7,
                        "openAccessPdf": {"url": "https://example.org/open.pdf"},
                        "fieldsOfStudy": ["Computer Science"],
                        "s2FieldsOfStudy": [{"category": "Computer Science"}],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SemanticScholarConnector(client=client, api_key="secret")
    config = SearchConfig(
        topic="paper search",
        from_year=2024,
        to_year=2026,
        paper_type=PaperType.CONFERENCE_PAPER,
        open_access_only=True,
    )

    result = asyncio.run(connector.search(config, limit=5))
    asyncio.run(client.aclose())

    assert captured["query"] == "paper search"
    assert captured["year"] == "2024-2026"
    assert captured["publicationTypes"] == "Conference"
    assert "openAccessPdf" in captured

    candidate = result.candidates[0]
    assert candidate.source == "semantic_scholar"
    assert candidate.id == "semantic_scholar:abc123"
    assert candidate.arxiv_id == "2501.00001"
    assert candidate.pdf_url is not None
    assert candidate.fields_of_study == ["Computer Science"]


def test_semantic_scholar_retries_rate_limit_before_success() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "retry-ok",
                        "title": "Recovered Semantic Scholar Result",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SemanticScholarConnector(
        client=client,
        max_retries=1,
        retry_sleep_seconds=0,
    )

    result = asyncio.run(connector.search(SearchConfig(topic="rate limit"), limit=1))
    asyncio.run(client.aclose())

    assert calls == 2
    assert result.candidates[0].title == "Recovered Semantic Scholar Result"


def test_semantic_scholar_rate_limit_error_mentions_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(429)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SemanticScholarConnector(
        client=client,
        max_retries=0,
        retry_sleep_seconds=0,
    )

    try:
        asyncio.run(connector.search(SearchConfig(topic="rate limit"), limit=1))
    except AcademicSearchError as exc:
        message = str(exc)
    else:
        message = ""
    finally:
        asyncio.run(client.aclose())

    assert "SEMANTIC_SCHOLAR_API_KEY" in message


def test_search_service_keeps_results_when_one_connector_fails(tmp_path) -> None:
    class FailingConnector:
        source = "broken"

        async def search(self, config: SearchConfig, limit: int):  # noqa: ARG002
            raise AcademicSearchError("rate limited")

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Surviving Source",
                        "publication_year": 2024,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = AcademicSearchService(
        connectors=[OpenAlexConnector(client=client), FailingConnector()],
        raw_responses_dir=tmp_path,
    )

    report = asyncio.run(service.search(SearchConfig(topic="graph rag"), limit_per_source=1))
    asyncio.run(client.aclose())

    assert [candidate.title for candidate in report.candidates] == ["Surviving Source"]
    assert report.errors == {"broken": "rate limited"}
    assert len(list(tmp_path.glob("*_openalex.json"))) == 1


def test_search_service_keeps_broad_result_set_for_narrowing() -> None:
    class ManyConnector:
        source = "many"

        async def search(self, config: SearchConfig, limit: int):  # noqa: ARG002
            return type(
                "Result",
                (),
                {
                    "source": self.source,
                    "raw_response": {"ok": True},
                    "candidates": [
                        OpenAlexConnector()._normalize_work(
                            {
                                "id": f"https://openalex.org/W{index}",
                                "display_name": f"Paper {index}",
                                "publication_year": 2025,
                            }
                        )
                        for index in range(25)
                    ],
                },
            )()

    service = AcademicSearchService(connectors=[ManyConnector()])
    report = asyncio.run(
        service.search(SearchConfig(topic="graph rag", target_max_papers=20), limit_per_source=25)
    )

    assert len(report.candidates) == 25


def test_helpers_reconstruct_abstract_and_dedupe_by_doi() -> None:
    assert abstract_from_inverted_index({"hello": [0], "world": [1]}) == "hello world"

    config = SearchConfig(topic="graph rag")
    first = OpenAlexConnector()._normalize_work(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Same Paper",
            "doi": "https://doi.org/10.1000/example",
        }
    )
    second = SemanticScholarConnector()._normalize_paper(
        {
            "paperId": "S1",
            "title": "Same Paper",
            "externalIds": {"DOI": "10.1000/EXAMPLE"},
        }
    )

    assert first is not None
    assert second is not None
    assert len(dedupe_candidates([first, second])) == 1
    assert config.target_max_papers == 20


def test_dedupe_merges_arxiv_title_citations_and_better_abstract() -> None:
    first = OpenAlexConnector()._normalize_work(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Graph RAG for Literature Reviews",
            "publication_year": 2024,
            "cited_by_count": 3,
            "abstract_inverted_index": {"Short": [0]},
            "concepts": [{"display_name": "Information Retrieval"}],
        }
    )
    second = SemanticScholarConnector()._normalize_paper(
        {
            "paperId": "S1",
            "title": "Graph-RAG for literature review",
            "year": 2025,
            "citationCount": 25,
            "abstract": "A longer abstract with more complete metadata.",
            "externalIds": {"ArXiv": "2501.00001"},
            "fieldsOfStudy": ["Computer Science"],
        }
    )
    third = ArxivConnector()._normalize_entry(
        {
            "id": "http://arxiv.org/abs/2501.00001v1",
            "title": "Graph RAG for Literature Reviews",
            "published": "2025-01-01T00:00:00Z",
            "summary": "arXiv abstract.",
        }
    )

    assert first is not None
    assert second is not None
    assert third is not None

    deduped = dedupe_candidates([first, second, third])

    assert len(deduped) == 1
    assert deduped[0].citation_count == 25
    assert deduped[0].abstract == "A longer abstract with more complete metadata."
    assert "openalex" in deduped[0].source
    assert "semantic_scholar" in deduped[0].source
    assert deduped[0].arxiv_id in {"2501.00001", "2501.00001v1"}


def test_rank_candidates_supports_newest_most_cited_and_balanced_reasons() -> None:
    old_classic = OpenAlexConnector()._normalize_work(
        {
            "id": "https://openalex.org/Wold",
            "display_name": "Classic Graph Retrieval",
            "publication_year": 2018,
            "publication_date": "2018-01-01",
            "cited_by_count": 900,
            "abstract_inverted_index": {"classic": [0]},
            "concepts": [{"display_name": "Information Retrieval"}],
        }
    )
    recent = OpenAlexConnector()._normalize_work(
        {
            "id": "https://openalex.org/Wnew",
            "display_name": "Fresh Graph RAG",
            "publication_year": 2026,
            "publication_date": "2026-02-01",
            "cited_by_count": 5,
            "abstract_inverted_index": {"fresh": [0]},
            "concepts": [{"display_name": "Graph RAG"}],
        }
    )

    assert old_classic is not None
    assert recent is not None

    assert rank_candidates([old_classic, recent], SortMode.NEWEST)[0].title == "Fresh Graph RAG"
    assert (
        rank_candidates([old_classic, recent], SortMode.MOST_CITED)[0].title
        == "Classic Graph Retrieval"
    )

    balanced = rank_candidates([old_classic, recent], SortMode.BALANCED)

    assert all(candidate.ranking_reason for candidate in balanced)
    assert "balanced" in balanced[0].ranking_reason
