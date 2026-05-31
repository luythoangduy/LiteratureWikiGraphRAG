from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from google import genai

from literature_wiki_graphrag.config import Settings, get_settings
from literature_wiki_graphrag.schemas import (
    NarrowingDecision,
    NarrowingReport,
    NarrowingSuggestion,
    PaperCandidate,
    SearchConfig,
    ThemeSummary,
)
from literature_wiki_graphrag.search import normalize_title

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_narrowing_report(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    *,
    settings: Settings | None = None,
    history: list[NarrowingDecision] | None = None,
) -> NarrowingReport:
    settings = settings or get_settings()
    history = history or []
    threshold = max(config.target_max_papers, config.target_min_papers)
    is_too_broad = len(candidates) > threshold

    if settings.google_api_key and candidates:
        try:
            return gemini_narrowing_report(
                candidates,
                config,
                settings=settings,
                history=history,
                threshold=threshold,
                is_too_broad=is_too_broad,
            )
        except Exception as exc:  # noqa: BLE001 - fallback keeps the narrowing loop usable.
            return heuristic_narrowing_report(
                candidates,
                config,
                history=history,
                threshold=threshold,
                is_too_broad=is_too_broad,
                fallback_reason=f"Gemini fallback: {exc}",
            )

    if settings.openrouter_api_key and candidates:
        try:
            return openrouter_narrowing_report(
                candidates,
                config,
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                history=history,
                threshold=threshold,
                is_too_broad=is_too_broad,
            )
        except Exception as exc:  # noqa: BLE001 - fallback keeps the narrowing loop usable.
            return heuristic_narrowing_report(
                candidates,
                config,
                history=history,
                threshold=threshold,
                is_too_broad=is_too_broad,
                fallback_reason=f"OpenRouter fallback: {exc}",
            )

    return heuristic_narrowing_report(
        candidates,
        config,
        history=history,
        threshold=threshold,
        is_too_broad=is_too_broad,
        fallback_reason=(
            "No Google or OpenRouter API key is configured."
            if not settings.google_api_key and not settings.openrouter_api_key
            else None
        ),
    )


def openrouter_narrowing_report(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    *,
    api_key: str,
    model: str,
    history: list[NarrowingDecision],
    threshold: int,
    is_too_broad: bool,
) -> NarrowingReport:
    prompt = build_gemini_prompt(candidates, config, threshold)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "LiteratureWikiGraphRAG",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON for literature search narrowing.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=60) as client:
        response = client.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = parse_json_object(content)
    return parsed_narrowing_report(
        parsed,
        candidates=candidates,
        history=history,
        threshold=threshold,
        is_too_broad=is_too_broad,
        summarizer=f"openrouter:{model}",
    )


def gemini_narrowing_report(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    *,
    settings: Settings,
    history: list[NarrowingDecision],
    threshold: int,
    is_too_broad: bool,
) -> NarrowingReport:
    client = genai.Client(api_key=settings.google_api_key)
    prompt = build_gemini_prompt(candidates, config, threshold)
    model = settings.google_gemini_model or DEFAULT_GEMINI_MODEL
    response = client.models.generate_content(model=model, contents=prompt)
    payload = parse_json_object(response.text or "")

    return parsed_narrowing_report(
        payload,
        candidates=candidates,
        history=history,
        threshold=threshold,
        is_too_broad=is_too_broad,
        summarizer=f"gemini:{model}",
    )


def parsed_narrowing_report(
    payload: dict[str, Any],
    *,
    candidates: list[PaperCandidate],
    history: list[NarrowingDecision],
    threshold: int,
    is_too_broad: bool,
    summarizer: str,
) -> NarrowingReport:
    return NarrowingReport(
        is_too_broad=is_too_broad,
        result_count=len(candidates),
        threshold=threshold,
        summarizer=summarizer,
        overview=str(payload.get("overview") or ""),
        themes=[
            ThemeSummary(
                label=str(theme.get("label") or "Theme"),
                summary=str(theme.get("summary") or ""),
                paper_ids=[str(item) for item in theme.get("paper_ids", [])],
                representative_titles=[
                    str(item) for item in theme.get("representative_titles", [])
                ],
            )
            for theme in payload.get("themes", [])
            if isinstance(theme, dict)
        ],
        suggestions=[
            NarrowingSuggestion(
                label=str(suggestion.get("label") or "Narrow focus"),
                rationale=str(suggestion.get("rationale") or ""),
                config_updates=dict(suggestion.get("config_updates") or {}),
                estimated_paper_count=suggestion.get("estimated_paper_count"),
            )
            for suggestion in payload.get("suggestions", [])
            if isinstance(suggestion, dict)
        ],
        history=history,
    )


def build_gemini_prompt(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    threshold: int,
) -> str:
    papers = [
        {
            "id": candidate.id,
            "title": candidate.title,
            "year": candidate.year,
            "citations": candidate.citation_count,
            "abstract": truncate(candidate.abstract or "", 800),
            "fields": candidate.fields_of_study,
            "keywords": candidate.keywords,
        }
        for candidate in candidates[:60]
    ]
    return (
        "You are helping narrow a broad literature search. Return strict JSON with keys "
        "overview, themes, suggestions. Each theme has label, summary, paper_ids, "
        "representative_titles. Each suggestion has label, rationale, config_updates, "
        "estimated_paper_count. Prefer practical filters that reduce the set to about "
        f"{config.target_min_papers}-{config.target_max_papers} papers; broad threshold is "
        f"{threshold}. Search config: {config.model_dump(mode='json')}. Papers: "
        f"{json.dumps(papers, ensure_ascii=False)}"
    )


def heuristic_narrowing_report(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    *,
    history: list[NarrowingDecision],
    threshold: int,
    is_too_broad: bool,
    fallback_reason: str | None = None,
) -> NarrowingReport:
    themes = cluster_candidates_by_theme(candidates)
    overview = build_overview(candidates, themes)
    suggestions = build_narrowing_suggestions(candidates, config, themes)

    return NarrowingReport(
        is_too_broad=is_too_broad,
        result_count=len(candidates),
        threshold=threshold,
        summarizer="heuristic",
        fallback_reason=fallback_reason,
        overview=overview,
        themes=themes,
        suggestions=suggestions,
        history=history,
    )


def cluster_candidates_by_theme(candidates: list[PaperCandidate]) -> list[ThemeSummary]:
    grouped: dict[str, list[PaperCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate_theme(candidate)].append(candidate)

    themes: list[ThemeSummary] = []
    for label, papers in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        titles = [paper.title for paper in papers[:3]]
        themes.append(
            ThemeSummary(
                label=label,
                summary=summarize_theme(label, papers),
                paper_ids=[paper.id for paper in papers],
                representative_titles=titles,
            )
        )
    return themes[:8]


def candidate_theme(candidate: PaperCandidate) -> str:
    for value in [*candidate.fields_of_study, *candidate.keywords]:
        normalized = " ".join(normalize_title(value).split()[:4])
        if normalized:
            return normalized.title()

    title_tokens = [
        token
        for token in normalize_title(candidate.title).split()
        if len(token) > 3 and token not in STOPWORDS
    ]
    if title_tokens:
        return " ".join(title_tokens[:3]).title()
    return "Uncategorized"


def summarize_theme(label: str, papers: list[PaperCandidate]) -> str:
    years = [paper.year for paper in papers if paper.year]
    citations = [paper.citation_count or 0 for paper in papers]
    year_span = f"{min(years)}-{max(years)}" if years else "unknown years"
    top_citations = max(citations) if citations else 0
    return (
        f"{label} contains {len(papers)} papers from {year_span}; "
        f"top citation count is {top_citations}."
    )


def build_overview(candidates: list[PaperCandidate], themes: list[ThemeSummary]) -> str:
    if not candidates:
        return "No papers matched the current search constraints."

    years = [candidate.year for candidate in candidates if candidate.year]
    year_span = f"{min(years)}-{max(years)}" if years else "unknown years"
    theme_labels = ", ".join(theme.label for theme in themes[:4]) or "no clear themes"
    return (
        f"The result set has {len(candidates)} papers across {year_span}. "
        f"The largest visible themes are {theme_labels}."
    )


def build_narrowing_suggestions(
    candidates: list[PaperCandidate],
    config: SearchConfig,
    themes: list[ThemeSummary],
) -> list[NarrowingSuggestion]:
    suggestions: list[NarrowingSuggestion] = []

    for theme in themes[:3]:
        suggestions.append(
            NarrowingSuggestion(
                label=f"Focus on {theme.label}",
                rationale=f"Theme has {len(theme.paper_ids)} related candidates.",
                config_updates={"must_include_keywords": [theme.label.lower()]},
                estimated_paper_count=len(theme.paper_ids),
            )
        )

    recent_year = max((candidate.year or 0 for candidate in candidates), default=0) - 2
    if recent_year > 0:
        recent_count = sum(1 for candidate in candidates if (candidate.year or 0) >= recent_year)
        suggestions.append(
            NarrowingSuggestion(
                label=f"Recent work since {recent_year}",
                rationale="Keeps the newest part of the result set.",
                config_updates={"from_year": recent_year},
                estimated_paper_count=recent_count,
            )
        )

    citation_counts = [candidate.citation_count or 0 for candidate in candidates]
    if citation_counts:
        citation_floor = max(10, sorted(citation_counts)[len(citation_counts) // 2])
        suggestions.append(
            NarrowingSuggestion(
                label=f"Highly cited baselines ({citation_floor}+ citations)",
                rationale="Keeps stronger baseline papers before GraphRAG indexing.",
                config_updates={"min_citation_count": citation_floor},
                estimated_paper_count=sum(
                    1
                    for candidate in candidates
                    if (candidate.citation_count or 0) >= citation_floor
                ),
            )
        )

    if not config.open_access_only:
        oa_count = sum(1 for candidate in candidates if candidate.open_access or candidate.pdf_url)
        suggestions.append(
            NarrowingSuggestion(
                label="Open abstracts or PDFs only",
                rationale="Prioritizes candidates with accessible text for downstream review.",
                config_updates={"open_access_only": True},
                estimated_paper_count=oa_count,
            )
        )

    return suggestions[:6]


def apply_narrowing_suggestion(
    config: SearchConfig,
    suggestion: NarrowingSuggestion,
) -> SearchConfig:
    updates = dict(suggestion.config_updates)
    if "must_include_keywords" in updates:
        current = list(config.must_include_keywords)
        for keyword in updates["must_include_keywords"]:
            if keyword not in current:
                current.append(str(keyword))
        updates["must_include_keywords"] = current
    return config.model_copy(update=updates)


def format_narrowing_report_text(report: NarrowingReport) -> str:
    lines = [
        f"Result set: {report.result_count} papers. Target threshold: {report.threshold}.",
        f"Summarizer: {report.summarizer}.",
        "",
        "Overview",
        report.overview or "No overview available.",
        "",
        "Research themes",
    ]
    for index, theme in enumerate(report.themes, start=1):
        titles = "; ".join(theme.representative_titles[:3]) or "No representative titles."
        lines.extend(
            [
                f"{index}. {theme.label} ({len(theme.paper_ids)} papers)",
                f"   Summary: {theme.summary}",
                f"   Representative papers: {titles}",
            ]
        )

    lines.extend(["", "Suggested narrowing options"])
    for index, suggestion in enumerate(report.suggestions, start=1):
        estimated = (
            f" Estimated candidates: {suggestion.estimated_paper_count}."
            if suggestion.estimated_paper_count is not None
            else ""
        )
        lines.extend(
            [
                f"{index}. {suggestion.label}",
                f"   Why: {suggestion.rationale}{estimated}",
                f"   Config updates: {json.dumps(suggestion.config_updates, ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def chat_narrowing_suggestion(
    user_message: str,
    report: NarrowingReport,
    config: SearchConfig,
    *,
    settings: Settings | None = None,
) -> NarrowingSuggestion:
    settings = settings or get_settings()
    prompt = (
        "You are helping a researcher narrow a literature search. The user will describe "
        "which area they want to focus on. Return strict JSON with keys label, rationale, "
        "config_updates, estimated_paper_count. config_updates may use from_year, to_year, "
        "min_citation_count, must_include_keywords, exclude_keywords, paper_type, "
        "open_access_only. Prefer simple keyword filters that can be rerun by the app.\n\n"
        f"Current search config: {config.model_dump(mode='json')}\n\n"
        f"Narrowing report:\n{format_narrowing_report_text(report)}\n\n"
        f"User preference: {user_message}"
    )

    if settings.google_api_key:
        client = genai.Client(api_key=settings.google_api_key)
        model = settings.google_gemini_model or DEFAULT_GEMINI_MODEL
        response = client.models.generate_content(model=model, contents=prompt)
        payload = parse_json_object(response.text or "")
    elif settings.openrouter_api_key:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "LiteratureWikiGraphRAG",
        }
        payload_data = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload_data,
            )
            response.raise_for_status()
        payload = parse_json_object(response.json()["choices"][0]["message"]["content"])
    else:
        return NarrowingSuggestion(
            label="Manual narrowing request",
            rationale="No LLM key is configured, so the user message was converted to keywords.",
            config_updates={"must_include_keywords": [user_message]},
        )

    return NarrowingSuggestion(
        label=str(payload.get("label") or "LLM narrowing suggestion"),
        rationale=str(payload.get("rationale") or ""),
        config_updates=dict(payload.get("config_updates") or {}),
        estimated_paper_count=payload.get("estimated_paper_count"),
    )


def save_narrowing_history(
    output_dir: Path,
    history: list[NarrowingDecision],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "narrowing_history.json"
    payload = [decision.model_dump(mode="json") for decision in history]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_narrowing_history(output_dir: Path) -> list[NarrowingDecision]:
    path = output_dir / "narrowing_history.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [NarrowingDecision.model_validate(item) for item in payload]


def parse_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def truncate(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else f"{value[: max_chars - 3]}..."


STOPWORDS = {
    "with",
    "from",
    "using",
    "based",
    "paper",
    "study",
    "review",
    "towards",
    "through",
    "learning",
}
