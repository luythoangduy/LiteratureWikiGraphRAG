from literature_wiki_graphrag.ingestion import (
    ingest_approved_papers,
    ingest_papers,
    save_approved_papers,
    split_text,
)
from literature_wiki_graphrag.schemas import ApprovedPaper, PaperCandidate


def test_ingest_papers_creates_traceable_evidence_chunks(tmp_path) -> None:
    candidate = PaperCandidate(
        id="semantic_scholar:abc",
        source="semantic_scholar",
        title="Graph RAG for Literature Review",
        authors=["Ada Lovelace"],
        year=2026,
        abstract="Graph retrieval augmented generation links claims to evidence.",
        doi="10.1234/example",
        arxiv_id="2605.12345",
    )

    result = ingest_papers(
        [candidate],
        output_dir=tmp_path,
        bibtex_by_id={candidate.id: "@article{example,title={Graph RAG}}"},
        fetch_pdfs=False,
    )

    evidence = result.evidence[0]
    assert evidence.candidate_id == candidate.id
    assert evidence.bibtex
    assert evidence.extraction_error is None
    assert result.output_path.exists()
    assert evidence.chunks
    assert evidence.chunks[0].section == "abstract"
    assert evidence.chunks[0].metadata["paper_title"] == candidate.title
    assert evidence.chunks[0].metadata["doi"] == candidate.doi
    assert evidence.chunks[0].embedding


def test_split_text_overlaps_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(10))

    chunks = split_text(text, max_words=5, overlap_words=2)

    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7 word8 word9",
    ]


def test_ingest_approved_papers_success(tmp_path) -> None:
    candidate = PaperCandidate(
        id="test:1",
        source="test",
        title="Test Paper",
        year=2026,
        abstract="This is a test paper abstract.",
    )
    approved = [ApprovedPaper(candidate=candidate)]
    save_approved_papers(tmp_path, approved)

    result = ingest_approved_papers(tmp_path, target_max_papers=5, generate_embeddings=False)
    assert len(result.evidence) == 1
    assert result.evidence[0].title == "Test Paper"
    assert result.output_path.exists()


def test_ingest_approved_papers_fails_when_no_approved_file(tmp_path) -> None:
    import pytest
    with pytest.raises(ValueError, match="No approved papers file found"):
        ingest_approved_papers(tmp_path)


def test_ingest_approved_papers_fails_when_exceeds_target_max(tmp_path) -> None:
    import pytest
    candidates = [
        PaperCandidate(
            id=f"test:{i}",
            source="test",
            title=f"Test Paper {i}",
            year=2026,
            abstract=f"Abstract {i}",
        )
        for i in range(5)
    ]
    approved = [ApprovedPaper(candidate=c) for c in candidates]
    save_approved_papers(tmp_path, approved)

    with pytest.raises(ValueError, match="exceeds target maximum"):
        ingest_approved_papers(tmp_path, target_max_papers=3)
