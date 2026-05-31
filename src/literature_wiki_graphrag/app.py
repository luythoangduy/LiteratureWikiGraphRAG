import asyncio
import json
from datetime import date
from pathlib import Path

import streamlit as st

from literature_wiki_graphrag.config import get_settings
from literature_wiki_graphrag.narrowing import (
    apply_narrowing_suggestion,
    build_narrowing_report,
    load_narrowing_history,
    save_narrowing_history,
)
from literature_wiki_graphrag.schemas import NarrowingDecision, PaperType, SearchConfig, SortMode
from literature_wiki_graphrag.search import AcademicSearchReport, search_academic_sources
from literature_wiki_graphrag.storage import save_model


def parse_keywords(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_academic_search(
    config: SearchConfig,
    limit_per_source: int | None = None,
) -> AcademicSearchReport:
    return asyncio.run(
        search_academic_sources(
            config,
            settings=get_settings(),
            limit_per_source=limit_per_source,
        )
    )


def candidate_rows(report: AcademicSearchReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report.candidates:
        rows.append(
            {
                "title": candidate.title,
                "year": candidate.year,
                "citations": candidate.citation_count,
                "source": candidate.source,
                "venue": candidate.venue,
                "doi": candidate.doi,
                "arxiv_id": candidate.arxiv_id,
                "open_access": candidate.open_access,
                "ranking_reason": candidate.ranking_reason,
                "url": str(candidate.url) if candidate.url else None,
                "abstract": candidate.abstract,
            }
        )
    return rows


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="LiteratureWikiGraphRAG", layout="wide")
    st.session_state.setdefault("narrowing_history", load_narrowing_history(settings.output_dir))

    st.title("LiteratureWikiGraphRAG")
    st.caption("Concept-centric GraphRAG workspace for literature review discovery.")

    with st.sidebar:
        st.header("Search Intake")
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. graph retrieval augmented generation",
        )

        current_year = date.today().year
        year_range = st.slider("Year range", 1990, current_year, (2020, current_year))
        sort_mode = st.selectbox(
            "Sort mode",
            options=list(SortMode),
            format_func=lambda mode: {
                SortMode.NEWEST: "Newest first",
                SortMode.MOST_CITED: "Most cited first",
                SortMode.BALANCED: "Balanced review set",
            }[mode],
            index=2,
        )
        target_range = st.slider("Target paper count", 5, 50, (15, 20))
        source_result_limit = st.slider(
            "Source result limit",
            min_value=10,
            max_value=100,
            value=max(target_range[1] * 3, 25),
            step=5,
            help=(
                "Number of raw results to request from each academic source "
                "before dedupe and ranking."
            ),
        )
        min_citations = st.number_input("Minimum citation count", min_value=0, value=0)
        paper_type = st.selectbox(
            "Paper type",
            options=[PaperType.ALL, PaperType.PREPRINT],
            format_func=lambda item: item.value.replace("_", " ").title(),
        )
        open_access_only = st.checkbox("Open access only")
        must_include = st.text_input("Must-include keywords", placeholder="comma-separated")
        exclude = st.text_input("Exclude keywords", placeholder="comma-separated")

        save_clicked = st.button("Save Search Config", type="primary", use_container_width=True)
        search_clicked = st.button("Search Papers", use_container_width=True)

    if not topic:
        st.info("Enter a research topic in the sidebar to create the first search config.")
        return

    config = SearchConfig(
        topic=topic,
        from_year=year_range[0],
        to_year=year_range[1],
        sort_mode=sort_mode,
        target_min_papers=target_range[0],
        target_max_papers=target_range[1],
        min_citation_count=min_citations or None,
        must_include_keywords=parse_keywords(must_include),
        exclude_keywords=parse_keywords(exclude),
        paper_type=paper_type,
        open_access_only=open_access_only,
    )

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Structured Search Config")
        st.json(config.model_dump(mode="json"))

    with right:
        st.subheader("MVP Status")
        st.write("Project setup is ready.")
        st.write("arXiv search connector is active for this stage.")

    if save_clicked:
        output_path = Path(settings.output_dir) / "search_config.json"
        save_model(output_path, config)
        st.success(f"Saved search config to {output_path}")

    if search_clicked:
        with st.spinner("Searching arXiv..."):
            st.session_state["search_report"] = run_academic_search(
                config,
                limit_per_source=source_result_limit,
            )
            st.session_state["active_search_config"] = config
            st.session_state["active_source_result_limit"] = source_result_limit

    active_config = st.session_state.get("active_search_config", config)
    active_source_result_limit = st.session_state.get(
        "active_source_result_limit",
        source_result_limit,
    )

    report = st.session_state.get("search_report")
    if report:
        st.subheader("Paper Candidates")
        st.caption(f"Ranking mode: {active_config.sort_mode.value.replace('_', ' ')}")
        st.caption(f"Source result limit: {active_source_result_limit}")
        source_summary = ", ".join(
            f"{source}: {count}" for source, count in report.source_counts.items()
        )
        st.caption(source_summary or "No candidates returned.")

        if report.errors:
            for source, error in report.errors.items():
                st.warning(f"{source}: {error}")

        narrowing_report = build_narrowing_report(
            report.candidates,
            active_config,
            settings=settings,
            history=st.session_state["narrowing_history"],
        )
        if narrowing_report.is_too_broad:
            st.warning(
                "Result set is still broad. Review themes and choose a narrowing focus "
                "before moving on to GraphRAG."
            )
            st.caption(f"Summarizer: {narrowing_report.summarizer}")
            if narrowing_report.fallback_reason:
                st.info(narrowing_report.fallback_reason)
            st.write(narrowing_report.overview)

            st.subheader("Research Themes")
            for theme in narrowing_report.themes:
                with st.expander(f"{theme.label} ({len(theme.paper_ids)} papers)"):
                    st.write(theme.summary)
                    st.write(", ".join(theme.representative_titles))

            st.subheader("Narrowing Choices")
            suggestion_labels = [suggestion.label for suggestion in narrowing_report.suggestions]
            if suggestion_labels:
                selected_label = st.radio(
                    "Choose focus",
                    suggestion_labels,
                    label_visibility="collapsed",
                )
                selected = next(
                    suggestion
                    for suggestion in narrowing_report.suggestions
                    if suggestion.label == selected_label
                )
                st.caption(selected.rationale)
                if selected.estimated_paper_count is not None:
                    st.caption(f"Estimated candidates: {selected.estimated_paper_count}")
                if st.button("Apply Focus And Re-run", use_container_width=True):
                    next_config = apply_narrowing_suggestion(active_config, selected)
                    decision = NarrowingDecision(
                        selected_label=selected.label,
                        rationale=selected.rationale,
                        config_updates=selected.config_updates,
                    )
                    st.session_state["narrowing_history"].append(decision)
                    save_narrowing_history(
                        settings.output_dir,
                        st.session_state["narrowing_history"],
                    )
                    with st.spinner("Re-running with narrowed focus..."):
                        st.session_state["search_report"] = run_academic_search(
                            next_config,
                            limit_per_source=active_source_result_limit,
                        )
                        st.session_state["active_search_config"] = next_config
                    st.rerun()

            if narrowing_report.history:
                st.subheader("Narrowing History")
                st.dataframe(
                    [
                        decision.model_dump(mode="json")
                        for decision in narrowing_report.history
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        display_candidates = (
            report.candidates
            if narrowing_report.is_too_broad
            else report.candidates[: active_config.target_max_papers]
        )
        display_report = AcademicSearchReport(
            candidates=display_candidates,
            source_counts=report.source_counts,
            errors=report.errors,
        )
        rows = candidate_rows(display_report)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            output_path = Path(settings.output_dir) / "paper_candidates.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [candidate.model_dump(mode="json") for candidate in display_candidates]
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"Saved normalized candidates to {output_path}")
        elif not report.errors:
            st.info("No papers matched the current constraints.")


if __name__ == "__main__":
    main()
