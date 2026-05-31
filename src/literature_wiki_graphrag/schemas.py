from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class SortMode(StrEnum):
    NEWEST = "newest"
    MOST_CITED = "most_cited"
    BALANCED = "balanced"


class PaperType(StrEnum):
    ALL = "all"
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"


class SearchConfig(BaseModel):
    topic: str
    from_year: int | None = None
    to_year: int | None = None
    sort_mode: SortMode = SortMode.BALANCED
    target_min_papers: int = 15
    target_max_papers: int = 20
    min_citation_count: int | None = None
    must_include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    paper_type: PaperType = PaperType.ALL
    open_access_only: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PaperCandidate(BaseModel):
    id: str
    source: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    publication_date: date | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None
    citation_count: int | None = None
    open_access: bool | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    ranking_reason: str | None = None


class ApprovedPaper(BaseModel):
    """A paper candidate that the researcher has explicitly included."""

    candidate: PaperCandidate
    approved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExcludedPaper(BaseModel):
    """A paper candidate that the researcher has explicitly excluded."""

    candidate: PaperCandidate
    reason: str = ""
    excluded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PaperChunk(BaseModel):
    id: str
    paper_id: str
    section: str | None = None
    text: str
    token_estimate: int
    metadata: dict[str, object] = Field(default_factory=dict)
    embedding: list[float] | None = None


class PaperEvidence(BaseModel):
    id: str
    candidate_id: str
    source: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None
    bibtex: str | None = None
    pdf_text: str | None = None
    extraction_error: str | None = None
    chunks: list[PaperChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ThemeSummary(BaseModel):
    label: str
    summary: str
    paper_ids: list[str] = Field(default_factory=list)
    representative_titles: list[str] = Field(default_factory=list)


class NarrowingSuggestion(BaseModel):
    label: str
    rationale: str
    config_updates: dict[str, object] = Field(default_factory=dict)
    estimated_paper_count: int | None = None


class NarrowingDecision(BaseModel):
    selected_label: str
    rationale: str | None = None
    config_updates: dict[str, object] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NarrowingReport(BaseModel):
    is_too_broad: bool
    result_count: int
    threshold: int
    summarizer: str = "heuristic"
    fallback_reason: str | None = None
    overview: str
    themes: list[ThemeSummary] = Field(default_factory=list)
    suggestions: list[NarrowingSuggestion] = Field(default_factory=list)
    history: list[NarrowingDecision] = Field(default_factory=list)
