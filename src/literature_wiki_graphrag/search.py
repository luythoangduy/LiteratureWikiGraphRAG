from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree as ET

import httpx
from rapidfuzz import fuzz

from literature_wiki_graphrag.config import Settings, get_settings
from literature_wiki_graphrag.schemas import (
    PaperCandidate,
    PaperType,
    SearchConfig,
    SortMode,
)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


class AcademicSearchError(RuntimeError):
    """Raised when an academic source cannot complete a search request."""


@dataclass(frozen=True)
class ConnectorSearchResult:
    source: str
    candidates: list[PaperCandidate]
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class AcademicSearchReport:
    candidates: list[PaperCandidate]
    source_counts: dict[str, int]
    errors: dict[str, str] = field(default_factory=dict)


class AcademicSearchConnector(Protocol):
    source: str

    async def search(self, config: SearchConfig, limit: int) -> ConnectorSearchResult: ...


class ArxivConnector:
    source = "arxiv"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def search(self, config: SearchConfig, limit: int) -> ConnectorSearchResult:
        if config.paper_type not in {PaperType.ALL, PaperType.PREPRINT}:
            return ConnectorSearchResult(
                source=self.source,
                candidates=[],
                raw_response={
                    "skipped": True,
                    "reason": "arXiv-only stage supports all/preprint paper types only.",
                },
            )

        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20)

        try:
            payload = await self._search_with_fallbacks(client, config, limit)
        except httpx.HTTPStatusError as exc:
            raise AcademicSearchError(f"arXiv returned HTTP {exc.response.status_code}") from exc
        except ET.ParseError as exc:
            raise AcademicSearchError("arXiv returned invalid Atom XML.") from exc
        except httpx.HTTPError as exc:
            raise AcademicSearchError(f"arXiv request failed: {exc}") from exc
        finally:
            if close_client:
                await client.aclose()

        candidates = [
            candidate
            for entry in payload.get("entries", [])
            if (candidate := self._normalize_entry(entry)) is not None
        ]
        return ConnectorSearchResult(
            source=self.source,
            candidates=filter_candidates(candidates, config),
            raw_response=payload,
        )

    async def _search_with_fallbacks(
        self,
        client: httpx.AsyncClient,
        config: SearchConfig,
        limit: int,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        last_payload: dict[str, Any] | None = None

        for query in arxiv_search_queries(config.topic):
            params = self._build_params(config, limit, query)
            response = await client.get(ARXIV_QUERY_URL, params=params)
            response.raise_for_status()
            payload = parse_arxiv_feed(response.text)
            attempts.append({"query": query, "count": len(payload.get("entries", []))})

            if payload.get("entries"):
                payload["_lwgrag_search"] = {
                    "original_topic": config.topic,
                    "used_query": query,
                    "attempts": attempts,
                }
                return payload

            last_payload = payload

        if last_payload is not None:
            last_payload["_lwgrag_search"] = {
                "original_topic": config.topic,
                "used_query": None,
                "attempts": attempts,
            }
            return last_payload

        raise AcademicSearchError("arXiv search did not return a response.")

    def _build_params(
        self,
        config: SearchConfig,
        limit: int,
        query: str,
    ) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "search_query": query,
            "start": 0,
            "max_results": min(max(limit, 1), 100),
        }

        if config.sort_mode == SortMode.NEWEST:
            params["sortBy"] = "submittedDate"
            params["sortOrder"] = "descending"
        else:
            params["sortBy"] = "relevance"
            params["sortOrder"] = "descending"

        return params

    def _normalize_entry(self, entry: dict[str, Any]) -> PaperCandidate | None:
        title = normalize_whitespace(entry.get("title"))
        arxiv_id = extract_arxiv_id(entry.get("id"))
        if not title or not arxiv_id:
            return None

        published = parse_datetime_date(entry.get("published"))
        categories = [str(category) for category in entry.get("categories", []) if category]

        return PaperCandidate(
            id=f"arxiv:{arxiv_id}",
            source=self.source,
            title=title,
            authors=[
                author
                for author in entry.get("authors", [])
                if author
            ],
            year=published.year if published else None,
            publication_date=published,
            venue=entry.get("journal_ref") or "arXiv",
            abstract=normalize_whitespace(entry.get("summary")),
            doi=normalize_doi(entry.get("doi")),
            arxiv_id=arxiv_id,
            url=clean_url(entry.get("id")),
            pdf_url=clean_url(entry.get("pdf_url")),
            citation_count=None,
            open_access=True,
            fields_of_study=categories,
            keywords=categories,
        )


class OpenAlexConnector:
    source = "openalex"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        mailto: str | None = None,
    ) -> None:
        self._client = client
        self._mailto = mailto

    async def search(self, config: SearchConfig, limit: int) -> ConnectorSearchResult:
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20)

        try:
            payload = await self._search_with_fallbacks(client, config, limit)
        except httpx.HTTPStatusError as exc:
            raise AcademicSearchError(
                f"OpenAlex returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AcademicSearchError(f"OpenAlex request failed: {exc}") from exc
        finally:
            if close_client:
                await client.aclose()

        works = payload.get("results") or []
        candidates = [
            candidate
            for item in works
            if (candidate := self._normalize_work(item)) is not None
        ]
        return ConnectorSearchResult(
            source=self.source,
            candidates=filter_candidates(candidates, config),
            raw_response=payload,
        )

    async def _search_with_fallbacks(
        self,
        client: httpx.AsyncClient,
        config: SearchConfig,
        limit: int,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        last_payload: dict[str, Any] | None = None

        for topic in openalex_search_topics(config.topic):
            params = self._build_params(config, limit, topic)
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            attempts.append(
                {
                    "topic": topic,
                    "count": (payload.get("meta") or {}).get("count"),
                }
            )

            if payload.get("results"):
                if topic != config.topic:
                    payload["_lwgrag_fallback_search"] = {
                        "original_topic": config.topic,
                        "used_topic": topic,
                        "attempts": attempts,
                    }
                return payload

            last_payload = payload

        if last_payload is not None:
            last_payload["_lwgrag_fallback_search"] = {
                "original_topic": config.topic,
                "used_topic": None,
                "attempts": attempts,
            }
            return last_payload

        raise AcademicSearchError("OpenAlex search did not return a response.")

    def _build_params(
        self,
        config: SearchConfig,
        limit: int,
        topic: str | None = None,
    ) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "search": topic or config.topic,
            "per-page": min(max(limit, 1), 200),
        }

        filters: list[str] = []
        if config.from_year:
            filters.append(f"from_publication_date:{config.from_year}-01-01")
        if config.to_year:
            filters.append(f"to_publication_date:{config.to_year}-12-31")
        if config.min_citation_count is not None:
            min_count = max(config.min_citation_count - 1, 0)
            filters.append(f"cited_by_count:>{min_count}")
        if config.open_access_only:
            filters.append("open_access.is_oa:true")

        openalex_type = {
            PaperType.JOURNAL_ARTICLE: "article",
            PaperType.CONFERENCE_PAPER: "proceedings-article",
            PaperType.PREPRINT: "preprint",
        }.get(config.paper_type)
        if openalex_type:
            filters.append(f"type:{openalex_type}")

        if filters:
            params["filter"] = ",".join(filters)

        if config.sort_mode == SortMode.NEWEST:
            params["sort"] = "publication_date:desc"
        elif config.sort_mode == SortMode.MOST_CITED:
            params["sort"] = "cited_by_count:desc"
        else:
            params["sort"] = "relevance_score:desc"

        if self._mailto:
            params["mailto"] = self._mailto

        return params

    def _normalize_work(self, item: dict[str, Any]) -> PaperCandidate | None:
        title = item.get("display_name") or item.get("title")
        if not title:
            return None

        primary_location = item.get("primary_location") or {}
        best_oa_location = item.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        open_access = item.get("open_access") or {}
        ids = item.get("ids") or {}

        doi = normalize_doi(item.get("doi") or ids.get("doi"))
        url = (
            primary_location.get("landing_page_url")
            or item.get("id")
            or ids.get("openalex")
        )
        pdf_url = best_oa_location.get("pdf_url") or primary_location.get("pdf_url")
        arxiv_id = (
            extract_arxiv_id(ids.get("arxiv"))
            or extract_arxiv_id(url)
            or extract_arxiv_id(pdf_url)
            or extract_arxiv_id(doi)
        )

        concepts = item.get("concepts") or []
        keywords = item.get("keywords") or []

        return PaperCandidate(
            id=f"openalex:{item.get('id') or title}",
            source=self.source,
            title=str(title),
            authors=[
                author_name
                for authorship in item.get("authorships") or []
                if (author_name := ((authorship.get("author") or {}).get("display_name")))
            ],
            year=item.get("publication_year"),
            publication_date=parse_date(item.get("publication_date")),
            venue=source.get("display_name"),
            abstract=abstract_from_inverted_index(item.get("abstract_inverted_index")),
            doi=doi,
            arxiv_id=arxiv_id,
            url=clean_url(url),
            pdf_url=clean_url(pdf_url),
            citation_count=item.get("cited_by_count"),
            open_access=open_access.get("is_oa"),
            fields_of_study=[
                concept_name
                for concept in concepts
                if (concept_name := concept.get("display_name"))
            ],
            keywords=[
                keyword_name
                for keyword in keywords
                if (keyword_name := (keyword.get("display_name") or keyword.get("keyword")))
            ],
        )


class SemanticScholarConnector:
    source = "semantic_scholar"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        api_key: str | None = None,
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_sleep_seconds = retry_sleep_seconds

    async def search(self, config: SearchConfig, limit: int) -> ConnectorSearchResult:
        params = self._build_params(config, limit)
        headers = {"x-api-key": self._api_key} if self._api_key else None
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20)

        try:
            response = await self._get_with_retries(client, params, headers)
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise AcademicSearchError(
                    "Semantic Scholar rate limit hit (HTTP 429). "
                    "Wait a bit or set SEMANTIC_SCHOLAR_API_KEY for a higher quota."
                ) from exc
            raise AcademicSearchError(
                f"Semantic Scholar returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AcademicSearchError(f"Semantic Scholar request failed: {exc}") from exc
        finally:
            if close_client:
                await client.aclose()

        papers = payload.get("data") or []
        candidates = [
            candidate
            for item in papers
            if (candidate := self._normalize_paper(item)) is not None
        ]
        return ConnectorSearchResult(
            source=self.source,
            candidates=filter_candidates(candidates, config),
            raw_response=payload,
        )

    async def _get_with_retries(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str | int],
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        last_response: httpx.Response | None = None

        for attempt in range(self._max_retries + 1):
            response = await client.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=params,
                headers=headers,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response

            last_response = response
            if attempt >= self._max_retries:
                break

            retry_after = retry_after_seconds(response.headers.get("Retry-After"))
            await asyncio.sleep(retry_after or self._retry_sleep_seconds)

        if last_response is not None:
            last_response.raise_for_status()
        raise AcademicSearchError("Semantic Scholar request failed before receiving a response.")

    def _build_params(self, config: SearchConfig, limit: int) -> dict[str, str | int]:
        fields = [
            "paperId",
            "externalIds",
            "title",
            "authors",
            "year",
            "publicationDate",
            "venue",
            "abstract",
            "url",
            "citationCount",
            "openAccessPdf",
            "fieldsOfStudy",
            "s2FieldsOfStudy",
            "publicationTypes",
        ]
        params: dict[str, str | int] = {
            "query": config.topic,
            "limit": min(max(limit, 1), 100),
            "offset": 0,
            "fields": ",".join(fields),
        }

        year_filter = semantic_scholar_year_filter(config)
        if year_filter:
            params["year"] = year_filter
        if config.min_citation_count is not None:
            params["minCitationCount"] = config.min_citation_count
        if config.open_access_only:
            params["openAccessPdf"] = ""

        publication_type = {
            PaperType.JOURNAL_ARTICLE: "JournalArticle",
            PaperType.CONFERENCE_PAPER: "Conference",
            PaperType.PREPRINT: "Preprint",
        }.get(config.paper_type)
        if publication_type:
            params["publicationTypes"] = publication_type

        return params

    def _normalize_paper(self, item: dict[str, Any]) -> PaperCandidate | None:
        title = item.get("title")
        paper_id = item.get("paperId")
        if not title or not paper_id:
            return None

        external_ids = item.get("externalIds") or {}
        open_access_pdf = item.get("openAccessPdf") or {}
        s2_fields = item.get("s2FieldsOfStudy") or []
        fields_of_study = item.get("fieldsOfStudy") or []

        doi = normalize_doi(external_ids.get("DOI"))
        arxiv_id = extract_arxiv_id(external_ids.get("ArXiv")) or extract_arxiv_id(
            item.get("url")
        )

        return PaperCandidate(
            id=f"semantic_scholar:{paper_id}",
            source=self.source,
            title=str(title),
            authors=[
                author_name
                for author in item.get("authors") or []
                if (author_name := author.get("name"))
            ],
            year=item.get("year"),
            publication_date=parse_date(item.get("publicationDate")),
            venue=item.get("venue"),
            abstract=item.get("abstract"),
            doi=doi,
            arxiv_id=arxiv_id,
            url=clean_url(item.get("url")),
            pdf_url=clean_url(open_access_pdf.get("url")),
            citation_count=item.get("citationCount"),
            open_access=bool(open_access_pdf.get("url")) if open_access_pdf else False,
            fields_of_study=[
                str(field)
                for field in fields_of_study
                if field
            ],
            keywords=[
                category
                for field in s2_fields
                if (category := field.get("category"))
            ],
        )


class AcademicSearchService:
    def __init__(
        self,
        connectors: list[AcademicSearchConnector],
        *,
        raw_responses_dir: Path | None = None,
    ) -> None:
        self._connectors = connectors
        self._raw_responses_dir = raw_responses_dir

    async def search(
        self,
        config: SearchConfig,
        limit_per_source: int | None = None,
    ) -> AcademicSearchReport:
        limit = limit_per_source or max(config.target_max_papers * 3, 25)
        results = await asyncio.gather(
            *[self._safe_search(connector, config, limit) for connector in self._connectors],
        )

        candidates: list[PaperCandidate] = []
        source_counts: dict[str, int] = {}
        errors: dict[str, str] = {}

        for connector, result in zip(self._connectors, results, strict=True):
            if isinstance(result, Exception):
                errors[connector.source] = str(result)
                continue

            self._save_raw_response(config, result)
            source_counts[result.source] = len(result.candidates)
            candidates.extend(result.candidates)

        return AcademicSearchReport(
            candidates=rank_candidates(
                dedupe_candidates(candidates),
                config.sort_mode,
            ),
            source_counts=source_counts,
            errors=errors,
        )

    async def _safe_search(
        self,
        connector: AcademicSearchConnector,
        config: SearchConfig,
        limit: int,
    ) -> ConnectorSearchResult | Exception:
        try:
            return await connector.search(config, limit)
        except Exception as exc:  # noqa: BLE001 - one failed source should not stop search.
            return exc

    def _save_raw_response(self, config: SearchConfig, result: ConnectorSearchResult) -> None:
        if not self._raw_responses_dir:
            return

        self._raw_responses_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        topic_slug = slugify(config.topic)[:60] or "search"
        path = self._raw_responses_dir / f"{timestamp}_{topic_slug}_{result.source}.json"
        payload = {
            "source": result.source,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "config": config.model_dump(mode="json"),
            "response": result.raw_response,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_default_search_service(settings: Settings | None = None) -> AcademicSearchService:
    settings = settings or get_settings()
    return AcademicSearchService(
        connectors=[ArxivConnector()],
        raw_responses_dir=settings.raw_responses_dir,
    )


async def search_academic_sources(
    config: SearchConfig,
    *,
    settings: Settings | None = None,
    limit_per_source: int | None = None,
) -> AcademicSearchReport:
    service = build_default_search_service(settings)
    return await service.search(config, limit_per_source=limit_per_source)


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None

    max_position = max(
        (position for positions in index.values() for position in positions),
        default=-1,
    )
    if max_position < 0:
        return None

    words = [""] * (max_position + 1)
    for word, positions in index.items():
        for position in positions:
            if 0 <= position <= max_position:
                words[position] = word

    return " ".join(word for word in words if word).strip() or None


def semantic_scholar_year_filter(config: SearchConfig) -> str | None:
    if config.from_year and config.to_year:
        return f"{config.from_year}-{config.to_year}"
    if config.from_year:
        return f"{config.from_year}-"
    if config.to_year:
        return f"-{config.to_year}"
    return None


def parse_arxiv_feed(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, Any]] = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        links = []
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            links.append(link.attrib)

        pdf_url = next(
            (
                link.get("href")
                for link in links
                if link.get("title") == "pdf" or link.get("type") == "application/pdf"
            ),
            None,
        )

        entries.append(
            {
                "id": text_from_xml(entry, f"{{{ATOM_NS}}}id"),
                "title": text_from_xml(entry, f"{{{ATOM_NS}}}title"),
                "summary": text_from_xml(entry, f"{{{ATOM_NS}}}summary"),
                "published": text_from_xml(entry, f"{{{ATOM_NS}}}published"),
                "updated": text_from_xml(entry, f"{{{ATOM_NS}}}updated"),
                "authors": [
                    name
                    for author in entry.findall(f"{{{ATOM_NS}}}author")
                    if (name := text_from_xml(author, f"{{{ATOM_NS}}}name"))
                ],
                "categories": [
                    category.get("term")
                    for category in entry.findall(f"{{{ATOM_NS}}}category")
                    if category.get("term")
                ],
                "doi": text_from_xml(entry, f"{{{ARXIV_NS}}}doi"),
                "journal_ref": text_from_xml(entry, f"{{{ARXIV_NS}}}journal_ref"),
                "pdf_url": pdf_url,
            }
        )

    return {
        "feed_title": text_from_xml(root, f"{{{ATOM_NS}}}title"),
        "updated": text_from_xml(root, f"{{{ATOM_NS}}}updated"),
        "entries": entries,
    }


def text_from_xml(element: ET.Element, path: str) -> str | None:
    found = element.find(path)
    if found is None or found.text is None:
        return None
    return normalize_whitespace(found.text)


def arxiv_search_queries(topic: str) -> list[str]:
    cleaned = normalize_whitespace(correct_topic_typos(topic))
    phrase_queries = arxiv_phrase_queries(cleaned)
    token_query = arxiv_token_query(cleaned)

    queries: list[str] = []
    add_unique(queries, phrase_queries)
    add_unique(queries, token_query)
    add_unique(queries, f'all:"{escape_arxiv_query(cleaned)}"')
    return queries


def arxiv_phrase_queries(topic: str) -> str:
    lowered = topic.lower()
    parts: list[str] = []

    if "industrial" in lowered:
        parts.append("all:industrial")
    for phrase in ["anomaly detection", "representation learning", "fault detection"]:
        if phrase in lowered:
            parts.append(f'all:"{phrase}"')

    if parts:
        return " AND ".join(parts)

    chunks = [
        normalize_whitespace(chunk)
        for chunk in re.split(r"[,;:]+", topic)
        if normalize_whitespace(chunk)
    ]
    return " AND ".join(f'all:"{escape_arxiv_query(chunk)}"' for chunk in chunks[:3])


def arxiv_token_query(topic: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", topic.lower())
        if len(token) > 2 and token not in {"and", "the", "for", "with"}
    ]
    return " AND ".join(f"all:{escape_arxiv_query(token)}" for token in tokens[:6])


def correct_topic_typos(topic: str) -> str:
    corrections = {
        "idustrial": "industrial",
    }
    corrected = topic
    for typo, replacement in corrections.items():
        corrected = re.sub(rf"\b{typo}\b", replacement, corrected, flags=re.IGNORECASE)
    return corrected


def escape_arxiv_query(value: str) -> str:
    return value.replace('"', "")


def openalex_search_topics(topic: str) -> list[str]:
    cleaned = " ".join(topic.split())
    topics: list[str] = []
    add_unique(topics, cleaned)

    without_punctuation = re.sub(r"[,;:]+", " ", cleaned)
    without_punctuation = " ".join(without_punctuation.split())
    add_unique(topics, without_punctuation)

    for part in re.split(r"[,;:]+", cleaned):
        part = " ".join(part.split())
        if len(part) >= 4:
            add_unique(topics, part)

    return topics[:4]


def add_unique(values: list[str], value: str) -> None:
    if value and value.lower() not in {item.lower() for item in values}:
        values.append(value)


def normalize_whitespace(value: str | None) -> str:
    return " ".join(str(value or "").split())


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_datetime_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = str(value).strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower() or None


def extract_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([^?#\s]+)",
        r"arxiv:([^?#\s]+)",
        r"10\.48550/arxiv\.([^?#\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).removesuffix(".pdf")
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", text, flags=re.IGNORECASE):
        return text
    return None


def clean_url(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text if text.startswith(("http://", "https://")) else None


def filter_candidates(
    candidates: list[PaperCandidate],
    config: SearchConfig,
) -> list[PaperCandidate]:
    return [candidate for candidate in candidates if candidate_matches_config(candidate, config)]


def candidate_matches_config(candidate: PaperCandidate, config: SearchConfig) -> bool:
    if config.from_year and candidate.year and candidate.year < config.from_year:
        return False
    if config.to_year and candidate.year and candidate.year > config.to_year:
        return False
    if config.min_citation_count is not None and candidate.citation_count is not None:
        if candidate.citation_count < config.min_citation_count:
            return False
    if config.open_access_only and candidate.open_access is False:
        return False

    searchable = " ".join(
        [
            candidate.title,
            candidate.abstract or "",
            " ".join(candidate.keywords),
            " ".join(candidate.fields_of_study),
        ]
    ).lower()

    if any(keyword.lower() not in searchable for keyword in config.must_include_keywords):
        return False
    if any(keyword.lower() in searchable for keyword in config.exclude_keywords):
        return False

    return True


def dedupe_candidates(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    key_to_index: dict[str, int] = {}
    deduped: list[PaperCandidate] = []

    for candidate in candidates:
        keys = candidate_keys(candidate)
        duplicate_index = next(
            (key_to_index[key] for key in keys if key in key_to_index),
            None,
        )
        if duplicate_index is None:
            duplicate_index = fuzzy_duplicate_index(candidate, deduped)

        if duplicate_index is not None:
            merged = merge_candidates(deduped[duplicate_index], candidate)
            deduped[duplicate_index] = merged
            for key in candidate_keys(merged):
                key_to_index[key] = duplicate_index
            continue

        deduped.append(candidate)
        candidate_index = len(deduped) - 1
        for key in keys:
            key_to_index[key] = candidate_index

    return deduped


def fuzzy_duplicate_index(
    candidate: PaperCandidate,
    existing_candidates: list[PaperCandidate],
) -> int | None:
    normalized_title = normalize_title(candidate.title)
    if not normalized_title:
        return None

    for index, existing in enumerate(existing_candidates):
        existing_title = normalize_title(existing.title)
        if existing_title and fuzz.token_set_ratio(normalized_title, existing_title) >= 96:
            return index

    return None


def merge_candidates(primary: PaperCandidate, duplicate: PaperCandidate) -> PaperCandidate:
    citation_count = max(
        primary.citation_count or 0,
        duplicate.citation_count or 0,
    )
    citation_count = citation_count if citation_count > 0 else None

    updates = {
        "source": merge_unique_strings([primary.source, duplicate.source]),
        "authors": merge_unique_list(primary.authors, duplicate.authors),
        "year": primary.year or duplicate.year,
        "publication_date": primary.publication_date or duplicate.publication_date,
        "venue": primary.venue or duplicate.venue,
        "abstract": better_text(primary.abstract, duplicate.abstract),
        "doi": primary.doi or duplicate.doi,
        "arxiv_id": primary.arxiv_id or duplicate.arxiv_id,
        "url": primary.url or duplicate.url,
        "pdf_url": primary.pdf_url or duplicate.pdf_url,
        "citation_count": citation_count,
        "open_access": (
            primary.open_access
            if primary.open_access is not None
            else duplicate.open_access
        ),
        "fields_of_study": merge_unique_list(primary.fields_of_study, duplicate.fields_of_study),
        "keywords": merge_unique_list(primary.keywords, duplicate.keywords),
    }
    return primary.model_copy(update=updates)


def better_text(first: str | None, second: str | None) -> str | None:
    if not first:
        return second
    if not second:
        return first
    return second if len(second) > len(first) else first


def merge_unique_strings(values: list[str]) -> str:
    return ", ".join(merge_unique_list(values, []))


def merge_unique_list(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*first, *second]:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(value)
    return merged


def rank_candidates(
    candidates: list[PaperCandidate],
    sort_mode: SortMode,
) -> list[PaperCandidate]:
    theme_counts = count_candidate_themes(candidates)

    def sort_key(candidate: PaperCandidate) -> tuple[float, date, int, str]:
        score = ranking_score(candidate, sort_mode, theme_counts)
        pub_date = candidate.publication_date or date(candidate.year or 1, 1, 1)
        citations = candidate.citation_count or 0
        return (score, pub_date, citations, candidate.title.lower())

    ranked = sorted(candidates, key=sort_key, reverse=True)
    return [
        candidate.model_copy(
            update={"ranking_reason": ranking_reason(candidate, sort_mode, theme_counts)}
        )
        for candidate in ranked
    ]


def ranking_score(
    candidate: PaperCandidate,
    sort_mode: SortMode,
    theme_counts: dict[str, int],
) -> float:
    if sort_mode == SortMode.NEWEST:
        return float(candidate_date_ordinal(candidate))
    if sort_mode == SortMode.MOST_CITED:
        return float(candidate.citation_count or 0)

    recency = recency_score(candidate)
    citations = citation_score(candidate)
    metadata = metadata_score(candidate)
    theme = theme_score(candidate, theme_counts)
    emerging = (
        0.15
        if (candidate.year or 0) >= date.today().year - 2
        and (candidate.citation_count or 0) < 10
        else 0
    )
    return (0.35 * recency) + (0.35 * citations) + (0.15 * metadata) + (0.10 * theme) + emerging


def ranking_reason(
    candidate: PaperCandidate,
    sort_mode: SortMode,
    theme_counts: dict[str, int],
) -> str:
    citations = candidate.citation_count or 0
    year = candidate.year or "unknown year"
    if sort_mode == SortMode.NEWEST:
        return f"Newest-first: published in {year}."
    if sort_mode == SortMode.MOST_CITED:
        return f"Most-cited: {citations} citations."

    parts = [
        f"balanced recency={recency_score(candidate):.2f}",
        f"citations={citation_score(candidate):.2f} ({citations})",
        f"metadata={metadata_score(candidate):.2f}",
    ]
    theme = primary_theme(candidate)
    if theme:
        parts.append(f"theme='{theme}'")
    if (candidate.year or 0) >= date.today().year - 2 and citations < 10:
        parts.append("emerging recent paper")
    return "; ".join(parts) + "."


def candidate_date_ordinal(candidate: PaperCandidate) -> int:
    if candidate.publication_date:
        return candidate.publication_date.toordinal()
    if candidate.year:
        return date(candidate.year, 1, 1).toordinal()
    return 0


def recency_score(candidate: PaperCandidate) -> float:
    current_year = date.today().year
    if not candidate.year:
        return 0.0
    age = max(current_year - candidate.year, 0)
    return max(0.0, 1.0 - min(age, 10) / 10)


def citation_score(candidate: PaperCandidate) -> float:
    citations = candidate.citation_count or 0
    return min(citations / 500, 1.0)


def metadata_score(candidate: PaperCandidate) -> float:
    fields = [
        candidate.abstract,
        candidate.doi,
        candidate.arxiv_id,
        candidate.url,
        candidate.pdf_url,
        candidate.venue,
    ]
    present = sum(1 for value in fields if value)
    return present / len(fields)


def count_candidate_themes(candidates: list[PaperCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        theme = primary_theme(candidate)
        if theme:
            counts[theme] = counts.get(theme, 0) + 1
    return counts


def primary_theme(candidate: PaperCandidate) -> str | None:
    for value in [*candidate.fields_of_study, *candidate.keywords]:
        normalized = normalize_title(value)
        if normalized:
            return normalized
    return None


def theme_score(candidate: PaperCandidate, theme_counts: dict[str, int]) -> float:
    theme = primary_theme(candidate)
    if not theme:
        return 0.0
    return 1 / max(theme_counts.get(theme, 1), 1)


def candidate_keys(candidate: PaperCandidate) -> set[str]:
    keys: set[str] = set()
    if candidate.doi:
        keys.add(f"doi:{normalize_doi(candidate.doi)}")
    if candidate.arxiv_id:
        keys.add(f"arxiv:{candidate.arxiv_id.lower()}")

    normalized_title = normalize_title(candidate.title)
    if normalized_title:
        keys.add(f"title:{normalized_title}")

    return keys


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
