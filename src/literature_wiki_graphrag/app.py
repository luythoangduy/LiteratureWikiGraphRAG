import asyncio
import json
from datetime import date
from pathlib import Path

import streamlit as st

from literature_wiki_graphrag.config import get_settings
from literature_wiki_graphrag.deep_research import stream_deep_research
from literature_wiki_graphrag.ingestion import (
    ingest_approved_papers,
    save_approved_papers,
    save_excluded_papers,
)
from literature_wiki_graphrag.narrowing import (
    apply_narrowing_suggestion,
    build_narrowing_report,
    chat_narrowing_suggestion,
    format_narrowing_report_text,
    load_narrowing_history,
    save_narrowing_history,
)
from literature_wiki_graphrag.schemas import (
    ApprovedPaper,
    ExcludedPaper,
    NarrowingDecision,
    PaperType,
    SearchConfig,
    SortMode,
)
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


def inject_custom_css() -> None:
    """Inject premium CSS to style Streamlit app components."""
    st.markdown(
        """
        <style>
        /* Modern font and background colors */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
        
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Premium sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0;
            padding-top: 2rem;
        }
        
        /* Expander card decoration */
        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                        0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
            background-color: white !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* Typography refinements */
        h1, h2, h3 {
            color: #0f172a !important;
            font-family: 'Outfit', 'Inter', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }
        
        .stCaption {
            color: #64748b !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Buttons design */
        button[kind="primary"] {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            border: none !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[kind="primary"]:hover {
            background-color: #1d4ed8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        }
        
        button[kind="secondary"] {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[kind="secondary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="LiteratureWikiGraphRAG", layout="wide")
    inject_custom_css()
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

        st.markdown("---")
        has_google_key = bool(settings.google_api_key)
        use_deep_research = st.checkbox(
            "Use Gemini Deep Research",
            value=has_google_key,
            disabled=not has_google_key,
            help=(
                "Requires a valid GOOGLE_API_KEY. "
                "Polls asynchronously for a comprehensive arXiv search report."
            ),
        )

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
        if use_deep_research:
            status_label = "Initializing Gemini Deep Research..."
            with st.status(status_label, expanded=True) as status_box:
                try:
                    import time
                    start_time = time.monotonic()
                    generator = stream_deep_research(config, settings=settings)
                    displayed_steps = set()
                    while True:
                        try:
                            progress = next(generator)
                            elapsed = progress.get("elapsed", 0.0)
                            steps = progress.get("steps", [])

                            # Update status label with time
                            lbl = (
                                "Gemini Deep Research active... "
                                f"({elapsed:.0f}s elapsed)"
                            )
                            status_box.update(label=lbl, state="running")

                            # Render new thoughts
                            for step_str in steps:
                                if step_str not in displayed_steps:
                                    st.write(f"✔️ {step_str}")
                                    displayed_steps.add(step_str)
                        except StopIteration as exc:
                            report = exc.value
                            break

                    elapsed_total = time.monotonic() - start_time
                    st.session_state["search_report"] = report
                    st.session_state["active_search_config"] = config
                    st.session_state["active_source_result_limit"] = (
                        source_result_limit
                    )

                    lbl = (
                        "Gemini Deep Research completed in "
                        f"{elapsed_total:.0f}s!"
                    )
                    status_box.update(label=lbl, state="complete")
                except Exception as exc:
                    status_box.update(
                        label="Gemini Deep Research failed!",
                        state="error",
                    )
                    st.error(
                        f"Gemini Deep Research failed: {exc}. "
                        "Falling back to standard arXiv search..."
                    )
                    with st.spinner("Searching arXiv fallback..."):
                        st.session_state["search_report"] = (
                            run_academic_search(
                                config,
                                limit_per_source=source_result_limit,
                            )
                        )
                        st.session_state["active_search_config"] = config
                        st.session_state["active_source_result_limit"] = (
                            source_result_limit
                        )
        else:
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
        if getattr(report, "deep_research_report", None):
            with st.expander("📄 View Gemini Deep Research Report", expanded=True):
                st.markdown(report.deep_research_report)
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
            st.subheader("Narrowing Brief")
            st.text_area(
                "LLM-readable narrowing context",
                value=format_narrowing_report_text(narrowing_report),
                height=360,
            )

            st.subheader("Chat To Choose Narrowing Focus")
            st.caption(
                "Tell the LLM which area you want, then apply its suggested filter and rerun."
            )
            st.session_state.setdefault("narrowing_chat", [])
            for message in st.session_state["narrowing_chat"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            user_message = st.chat_input(
                "e.g. focus on image anomaly detection with representation learning"
            )
            if user_message:
                st.session_state["narrowing_chat"].append(
                    {"role": "user", "content": user_message}
                )
                with st.spinner("Asking LLM for a narrowing filter..."):
                    try:
                        suggestion = chat_narrowing_suggestion(
                            user_message,
                            narrowing_report,
                            active_config,
                            settings=settings,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.session_state["narrowing_chat"].append(
                            {"role": "assistant", "content": f"Narrowing failed: {exc}"}
                        )
                    else:
                        st.session_state["pending_narrowing_suggestion"] = suggestion
                        reply = (
                            f"{suggestion.label}\n\n"
                            f"{suggestion.rationale}\n\n"
                            "Config updates:\n"
                            f"{json.dumps(suggestion.config_updates, ensure_ascii=False, indent=2)}"
                        )
                        if suggestion.estimated_paper_count is not None:
                            reply += (
                                f"\n\nEstimated candidates: "
                                f"{suggestion.estimated_paper_count}"
                            )
                        st.session_state["narrowing_chat"].append(
                            {"role": "assistant", "content": reply}
                        )
                st.rerun()

            pending_suggestion = st.session_state.get("pending_narrowing_suggestion")
            if pending_suggestion:
                st.info(f"Pending focus: {pending_suggestion.label}")
                st.json(pending_suggestion.config_updates)
                if st.button("Apply LLM Focus And Re-run", use_container_width=True):
                    next_config = apply_narrowing_suggestion(
                        active_config,
                        pending_suggestion,
                    )
                    decision = NarrowingDecision(
                        selected_label=pending_suggestion.label,
                        rationale=pending_suggestion.rationale,
                        config_updates=pending_suggestion.config_updates,
                    )
                    st.session_state["narrowing_history"].append(decision)
                    save_narrowing_history(
                        settings.output_dir,
                        st.session_state["narrowing_history"],
                    )
                    st.session_state.pop("pending_narrowing_suggestion", None)
                    with st.spinner("Re-running with LLM-selected focus..."):
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
            # Always persist the raw candidate snapshot for audit.
            output_path = Path(settings.output_dir) / "paper_candidates.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [c.model_dump(mode="json") for c in display_candidates]
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"Saved normalized candidates to {output_path}")
        elif not report.errors:
            st.info("No papers matched the current constraints.")

        # ── Approval Review Table ─────────────────────────────────────
        if not narrowing_report.is_too_broad and display_candidates:
            st.subheader("Paper Approval Review")
            st.caption(
                "Include or exclude each candidate. Excluded papers will be "
                "saved with your reason. Only approved papers will be ingested."
            )

            # Header row
            header_cols = st.columns([1, 4, 1, 1, 3])
            header_cols[0].write("**Include?**")
            header_cols[1].write("**Title & Authors**")
            header_cols[2].write("**Year**")
            header_cols[3].write("**Citations**")
            header_cols[4].write("**Exclusion Reason / Note**")

            include_flags: list[bool] = []
            exclude_reasons: list[str] = []
            for idx, candidate in enumerate(display_candidates):
                cols = st.columns([1, 4, 1, 1, 3])
                with cols[0]:
                    included = st.checkbox(
                        "Include",
                        value=True,
                        key=f"approve_{idx}",
                        label_visibility="collapsed",
                    )
                    include_flags.append(included)
                with cols[1]:
                    authors_str = (
                        ", ".join(candidate.authors)
                        if candidate.authors
                        else "Unknown Authors"
                    )
                    st.markdown(f"**{candidate.title}**  \n*{authors_str}*")
                with cols[2]:
                    st.write(str(candidate.year) if candidate.year else "-")
                with cols[3]:
                    citations = (
                        str(candidate.citation_count)
                        if candidate.citation_count is not None
                        else "0"
                    )
                    st.write(citations)
                with cols[4]:
                    reason = st.text_input(
                        "Exclusion reason" if not included else "Note (optional)",
                        value="",
                        key=f"reason_{idx}",
                        disabled=included,
                        label_visibility="collapsed",
                    )
                    exclude_reasons.append(reason)

            if st.button(
                "Confirm Approved Papers",
                type="primary",
                use_container_width=True,
            ):
                approved: list[ApprovedPaper] = []
                excluded: list[ExcludedPaper] = []
                for idx, candidate in enumerate(display_candidates):
                    if include_flags[idx]:
                        approved.append(ApprovedPaper(candidate=candidate))
                    else:
                        excluded.append(
                            ExcludedPaper(
                                candidate=candidate,
                                reason=exclude_reasons[idx],
                            )
                        )
                approved_path = save_approved_papers(
                    Path(settings.output_dir), approved
                )
                excluded_path = save_excluded_papers(
                    Path(settings.output_dir), excluded
                )
                st.session_state["approval_done"] = True
                st.success(
                    f"Approved {len(approved)} paper(s) → {approved_path}\n\n"
                    f"Excluded {len(excluded)} paper(s) → {excluded_path}"
                )

    # ── Paper Ingestion (gated to approved papers only) ───────────────
    st.subheader("Paper Ingestion")
    approved_path = Path(settings.output_dir) / "approved_papers.json"
    report_exists = st.session_state.get("search_report") is not None
    is_broad = False
    if report_exists:
        nr = build_narrowing_report(
            st.session_state["search_report"].candidates,
            st.session_state.get("active_search_config", config),
            settings=settings,
            history=st.session_state["narrowing_history"],
        )
        is_broad = nr.is_too_broad

    if is_broad:
        st.info(
            "Narrow the search results before approving and ingesting papers."
        )
    elif not approved_path.exists():
        st.info("Approve your paper selection first to enable ingestion.")
    else:
        fetch_pdfs = st.checkbox("Try PDF extraction when pdf_url is available")
        if st.button("Ingest Approved Papers", use_container_width=True):
            try:
                result = ingest_approved_papers(
                    Path(settings.output_dir),
                    target_max_papers=active_config.target_max_papers,
                    fetch_pdfs=fetch_pdfs,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Paper ingestion failed: {exc}")
            else:
                chunk_count = sum(len(item.chunks) for item in result.evidence)
                failed_pdfs = sum(
                    1 for item in result.evidence if item.extraction_error
                )
                st.success(
                    f"Ingested {len(result.evidence)} papers and "
                    f"{chunk_count} chunks to {result.output_path}"
                )
                if failed_pdfs:
                    st.warning(
                        f"{failed_pdfs} PDF extraction attempt(s) failed, "
                        "but metadata and abstract chunks were still saved."
                    )


if __name__ == "__main__":
    main()
