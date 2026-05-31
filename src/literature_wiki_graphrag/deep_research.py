"""Gemini Deep Research connector for autonomous arXiv paper discovery.

Uses the Interactions API (``client.interactions.create``) to run a
long-running Deep Research task that searches arXiv, reads papers, and
returns a structured JSON list of candidates.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from google import genai

from literature_wiki_graphrag.config import Settings, get_settings
from literature_wiki_graphrag.narrowing import parse_json_object
from literature_wiki_graphrag.schemas import PaperCandidate, SearchConfig
from literature_wiki_graphrag.search import AcademicSearchReport

logger = logging.getLogger(__name__)

DEFAULT_DEEP_RESEARCH_MODEL = "deep-research-preview-04-2026"


@dataclass(frozen=True)
class DeepResearchSearchReport(AcademicSearchReport):
    """Search report for Deep Research containing synthesized report text."""

    deep_research_report: str | None = None
    interaction_id: str | None = None


DEEP_RESEARCH_PROMPT_TEMPLATE = """\
You are a literature search assistant. Your task is to find academic papers
on arXiv that match the researcher's query.

**Research topic**: {topic}
**Year range**: {from_year} to {to_year}
**Target paper count**: {target_min} to {target_max} papers
**Sort preference**: {sort_mode}
{extra_constraints}

Search arXiv thoroughly for papers matching this topic. For each paper you
find, extract the following metadata. Return your results as a JSON object
with a single key "papers" containing a list of paper objects. Each paper
object must have these keys:

- "title": the full paper title
- "authors": list of author name strings
- "year": publication year as integer
- "arxiv_id": the arXiv identifier (e.g. "2501.12345")
- "abstract": the paper abstract (first 500 characters if very long)
- "url": the arXiv abstract page URL
- "pdf_url": the arXiv PDF URL
- "citation_count": estimated citation count if known, otherwise null
- "venue": publication venue if known, otherwise null
- "fields_of_study": list of research field/category strings

Return ONLY the JSON object. Do not include any other text before or after
the JSON. Target {target_min}-{target_max} high-quality, relevant papers.
Prefer recent, well-cited, and diverse papers covering different aspects of
the topic.
"""


@dataclass
class DeepResearchResult:
    """Result from a Deep Research interaction."""

    candidates: list[PaperCandidate]
    raw_report: str
    interaction_id: str
    model: str
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)


def build_deep_research_prompt(config: SearchConfig) -> str:
    """Build the prompt for the Deep Research agent."""
    extra_lines: list[str] = []
    if config.must_include_keywords:
        extra_lines.append(
            f"**Must include keywords**: {', '.join(config.must_include_keywords)}"
        )
    if config.exclude_keywords:
        extra_lines.append(
            f"**Exclude keywords**: {', '.join(config.exclude_keywords)}"
        )
    if config.min_citation_count:
        extra_lines.append(
            f"**Minimum citation count**: {config.min_citation_count}"
        )
    if config.open_access_only:
        extra_lines.append("**Open access only**: yes")
    if config.paper_type.value != "all":
        extra_lines.append(
            f"**Paper type**: {config.paper_type.value.replace('_', ' ')}"
        )

    return DEEP_RESEARCH_PROMPT_TEMPLATE.format(
        topic=config.topic,
        from_year=config.from_year or 2015,
        to_year=config.to_year or 2026,
        target_min=config.target_min_papers,
        target_max=config.target_max_papers,
        sort_mode=config.sort_mode.value.replace("_", " "),
        extra_constraints="\n".join(extra_lines),
    )


def extract_steps_info(interaction: Any) -> list[str]:
    """Extract human-readable reasoning or processing summaries from the steps."""
    steps_info = []
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        thought = getattr(step, "thought", None)
        summary = None
        if thought:
            summary = getattr(thought, "summary", None)
        if not summary:
            summary = getattr(step, "thought_summary", None)
        if summary:
            steps_info.append(summary)
        else:
            step_type = getattr(step, "type", None)
            if step_type:
                steps_info.append(f"Step: {step_type}")
    return steps_info


def stream_deep_research(
    config: SearchConfig,
    *,
    settings: Settings | None = None,
    poll_interval: float = 10.0,
    max_wait_seconds: float = 600.0,
) -> Generator[dict[str, Any], None, DeepResearchSearchReport]:
    """Generator that runs a Deep Research interaction and yields progress dicts,
    returning the final DeepResearchSearchReport.
    """
    settings = settings or get_settings()
    if not settings.google_api_key:
        msg = (
            "Deep Research requires GOOGLE_API_KEY. "
            "Set it in .env or fall back to arXiv search."
        )
        raise ValueError(msg)

    client = genai.Client(api_key=settings.google_api_key)
    model = settings.google_deep_research_model or DEFAULT_DEEP_RESEARCH_MODEL
    prompt = build_deep_research_prompt(config)

    logger.info("Starting Deep Research with model=%s", model)
    start_time = time.monotonic()

    interaction = client.interactions.create(
        input=prompt,
        agent=model,
        background=True,
    )

    interaction_id = interaction.id
    logger.info("Deep Research started: id=%s", interaction_id)

    # Poll for completion.
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait_seconds:
            msg = (
                f"Deep Research timed out after {elapsed:.0f}s. "
                f"Interaction ID: {interaction_id}"
            )
            raise TimeoutError(msg)

        interaction = client.interactions.get(interaction_id)

        # Extract steps info and yield progress
        steps = extract_steps_info(interaction)
        yield {"elapsed": elapsed, "steps": steps}

        if interaction.status == "completed":
            break
        elif interaction.status == "failed":
            error_detail = getattr(interaction, "error", "Unknown error")
            msg = f"Deep Research failed: {error_detail}"
            raise RuntimeError(msg)

        time.sleep(poll_interval)

    elapsed_total = time.monotonic() - start_time
    raw_report = interaction.output_text or ""
    logger.info(
        "Deep Research completed in %.1fs, report length=%d",
        elapsed_total,
        len(raw_report),
    )

    candidates, errors = parse_deep_research_report(raw_report)

    return DeepResearchSearchReport(
        candidates=candidates,
        source_counts={"deep_research": len(candidates)},
        errors={"deep_research": "; ".join(errors)} if errors else {},
        deep_research_report=raw_report,
        interaction_id=interaction_id,
    )


def run_deep_research(
    config: SearchConfig,
    *,
    settings: Settings | None = None,
    poll_interval: float = 10.0,
    max_wait_seconds: float = 600.0,
) -> DeepResearchSearchReport:
    """Run a Deep Research interaction and return parsed candidates wrapped
    in DeepResearchSearchReport (synchronous wrapper around stream_deep_research).
    """
    generator = stream_deep_research(
        config,
        settings=settings,
        poll_interval=poll_interval,
        max_wait_seconds=max_wait_seconds,
    )
    while True:
        try:
            next(generator)
        except StopIteration as exc:
            return exc.value


def parse_deep_research_report(
    report_text: str,
) -> tuple[list[PaperCandidate], list[str]]:
    """Extract paper candidates from a Deep Research report.

    Returns a tuple of (candidates, errors).
    """
    errors: list[str] = []
    candidates: list[PaperCandidate] = []

    if not report_text.strip():
        errors.append("Empty report text from Deep Research.")
        return candidates, errors

    try:
        payload = parse_json_object(report_text)
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Failed to parse JSON from report: {exc}")
        return candidates, errors

    if "papers" not in payload:
        errors.append("Missing 'papers' key in Deep Research output.")
        return candidates, errors

    papers = payload["papers"]
    if not isinstance(papers, list):
        errors.append(
            "Expected 'papers' key to be a list in Deep Research output."
        )
        return candidates, errors

    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            errors.append(f"Paper at index {index} is not a dict.")
            continue
        try:
            candidate = normalize_deep_research_paper(paper, index)
            candidates.append(candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to normalize paper {index}: {exc}")

    return candidates, errors


def normalize_deep_research_paper(
    paper: dict[str, Any],
    index: int,
) -> PaperCandidate:
    """Normalize a single paper dict from Deep Research into a
    PaperCandidate.
    """
    arxiv_id = str(paper.get("arxiv_id") or "")
    candidate_id = (
        f"deep_research:{arxiv_id}" if arxiv_id else f"deep_research:{index}"
    )

    url = paper.get("url")
    if not url and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"

    pdf_url = paper.get("pdf_url")
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return PaperCandidate(
        id=candidate_id,
        source="deep_research",
        title=str(paper.get("title") or f"Untitled paper {index}"),
        authors=[
            str(a) for a in (paper.get("authors") or []) if a
        ],
        year=int(paper["year"]) if paper.get("year") else None,
        arxiv_id=arxiv_id or None,
        abstract=str(paper.get("abstract") or ""),
        url=url,
        pdf_url=pdf_url,
        citation_count=paper.get("citation_count"),
        venue=paper.get("venue"),
        open_access=True,
        fields_of_study=[
            str(f) for f in (paper.get("fields_of_study") or []) if f
        ],
        ranking_reason="deep_research",
    )
