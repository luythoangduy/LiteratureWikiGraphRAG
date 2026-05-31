from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from literature_wiki_graphrag.schemas import (
    ApprovedPaper,
    ExcludedPaper,
    PaperCandidate,
    PaperChunk,
    PaperEvidence,
)


@dataclass(frozen=True)
class IngestionResult:
    evidence: list[PaperEvidence]
    output_path: Path


def load_paper_candidates(path: Path) -> list[PaperCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = f"Expected a list of paper candidates in {path}"
        raise ValueError(msg)
    return [PaperCandidate.model_validate(item) for item in payload]


def load_approved_papers(path: Path) -> list[PaperCandidate]:
    """Load approved papers and return the inner ``PaperCandidate`` objects."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = f"Expected a list of approved papers in {path}"
        raise ValueError(msg)
    approved = [ApprovedPaper.model_validate(item) for item in payload]
    return [item.candidate for item in approved]


def save_approved_papers(output_dir: Path, approved: list[ApprovedPaper]) -> Path:
    """Persist the researcher-approved paper set."""
    path = output_dir / "approved_papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in approved]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_excluded_papers(output_dir: Path, excluded: list[ExcludedPaper]) -> Path:
    """Persist excluded papers with their exclusion reasons."""
    path = output_dir / "excluded_papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in excluded]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ingest_papers(
    candidates: list[PaperCandidate],
    *,
    output_dir: Path,
    bibtex_by_id: dict[str, str] | None = None,
    fetch_pdfs: bool = False,
    generate_embeddings: bool = True,
) -> IngestionResult:
    evidence = [
        ingest_paper(
            candidate,
            bibtex=(bibtex_by_id or {}).get(candidate.id),
            fetch_pdf=fetch_pdfs,
            generate_embeddings=generate_embeddings,
        )
        for candidate in candidates
    ]
    output_path = output_dir / "paper_evidence.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in evidence]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return IngestionResult(evidence=evidence, output_path=output_path)


def ingest_approved_papers(
    output_dir: Path,
    *,
    target_max_papers: int = 20,
    bibtex_by_id: dict[str, str] | None = None,
    fetch_pdfs: bool = False,
    generate_embeddings: bool = True,
) -> IngestionResult:
    """Ingest papers from approved_papers.json, validating that the approved paper count
    is not above the target limit and that the approved papers file exists.
    """
    approved_path = output_dir / "approved_papers.json"
    if not approved_path.exists():
        msg = (
            f"No approved papers file found at {approved_path}. "
            "Please approve papers first."
        )
        raise ValueError(msg)

    candidates = load_approved_papers(approved_path)
    if len(candidates) > target_max_papers:
        msg = (
            f"Cannot ingest: approved papers count ({len(candidates)}) "
            f"exceeds target maximum ({target_max_papers})."
        )
        raise ValueError(msg)

    return ingest_papers(
        candidates,
        output_dir=output_dir,
        bibtex_by_id=bibtex_by_id,
        fetch_pdfs=fetch_pdfs,
        generate_embeddings=generate_embeddings,
    )


def ingest_paper(
    candidate: PaperCandidate,
    *,
    bibtex: str | None = None,
    fetch_pdf: bool = False,
    generate_embeddings: bool = True,
) -> PaperEvidence:
    pdf_text: str | None = None
    extraction_error: str | None = None
    if fetch_pdf and candidate.pdf_url:
        try:
            pdf_text = extract_pdf_text(str(candidate.pdf_url))
        except Exception as exc:  # noqa: BLE001
            extraction_error = str(exc)

    evidence = PaperEvidence(
        id=stable_id("evidence", candidate.id),
        candidate_id=candidate.id,
        source=candidate.source,
        title=candidate.title,
        authors=candidate.authors,
        year=candidate.year,
        venue=candidate.venue,
        abstract=candidate.abstract,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        url=candidate.url,
        pdf_url=candidate.pdf_url,
        bibtex=bibtex,
        pdf_text=pdf_text,
        extraction_error=extraction_error,
    )
    evidence.chunks = chunk_evidence(evidence, generate_embeddings=generate_embeddings)
    return evidence


def extract_pdf_text(pdf_url: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        msg = "Install the pdf extra to extract PDFs: pip install -e .[pdf]"
        raise RuntimeError(msg) from exc

    response = httpx.get(pdf_url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(response.content)
    try:
        reader = PdfReader(str(temp_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    finally:
        temp_path.unlink(missing_ok=True)


def chunk_evidence(
    evidence: PaperEvidence,
    *,
    max_words: int = 220,
    overlap_words: int = 35,
    generate_embeddings: bool = True,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for section, text in (("abstract", evidence.abstract), ("full_text", evidence.pdf_text)):
        section_chunks = split_text(text or "", max_words, overlap_words)
        for index, chunk_text in enumerate(section_chunks, start=1):
            metadata = {
                "paper_title": evidence.title,
                "doi": evidence.doi,
                "arxiv_id": evidence.arxiv_id,
                "section": section,
            }
            chunk = PaperChunk(
                id=stable_id("chunk", evidence.id, section, str(index), chunk_text[:80]),
                paper_id=evidence.id,
                section=section,
                text=chunk_text,
                token_estimate=estimate_tokens(chunk_text),
                metadata=metadata,
                embedding=hash_embedding(chunk_text) if generate_embeddings else None,
            )
            chunks.append(chunk)
    return chunks


def split_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def estimate_tokens(text: str) -> int:
    return max(1, round(len(re.findall(r"\S+", text)) * 1.3))


def hash_embedding(text: str, dimensions: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        byte = digest[index % len(digest)]
        values.append(round((byte / 127.5) - 1.0, 6))
    return values


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{parts[0]}:{digest}"
