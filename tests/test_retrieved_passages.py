"""Tests for retrieve_relevant_passages (per-message passage retrieval)."""

from unittest.mock import AsyncMock, patch

import pytest

from open_notebook.utils.context_builder import retrieve_relevant_passages


def _result(rid, parent, title, content, sim=0.9):
    return {
        "id": rid,
        "parent_id": parent,
        "title": title,
        "matches": content,
        "similarity": sim,
    }


def _context(source_ids):
    return {"sources": [{"id": sid} for sid in source_ids], "notes": []}


@pytest.mark.asyncio
async def test_keeps_only_selected_sources():
    """Results from sources outside the context are dropped."""
    context = _context(["source:selected"])
    results = [
        _result("source_embedding:a", "source:selected", "S", "keep me"),
        _result("source_embedding:b", "source:other", "O", "drop me"),
        _result("source_embedding:c", "source:selected", "S", "also keep"),
    ]
    with patch(
        "open_notebook.domain.notebook.vector_search_chunks",
        new=AsyncMock(return_value=results),
    ):
        passages = await retrieve_relevant_passages("hypnotics", context)

    assert [p["id"] for p in passages] == [
        "source_embedding:a",
        "source_embedding:c",
    ]
    assert all(p["parent_id"] == "source:selected" for p in passages)


@pytest.mark.asyncio
async def test_dedupes_and_caps_at_default():
    """Duplicate ids are dropped and output is capped at the default 30."""
    context = _context(["source:s1"])
    results = []
    for i in range(40):
        results.append(_result(f"source_embedding:c{i}", "source:s1", "S", f"c{i}"))
    results.append(_result("source_embedding:c0", "source:s1", "S", "dup"))
    with patch(
        "open_notebook.domain.notebook.vector_search_chunks",
        new=AsyncMock(return_value=results),
    ):
        passages = await retrieve_relevant_passages("question", context)

    ids = [p["id"] for p in passages]
    assert len(ids) == 30
    assert len(set(ids)) == len(ids)
    assert "source_embedding:c0" in ids


@pytest.mark.asyncio
async def test_cap_respects_env_override():
    """OPEN_NOTEBOOK_RETRIEVED_PASSAGES overrides the default cap."""
    context = _context(["source:s1"])
    results = [
        _result(f"source_embedding:c{i}", "source:s1", "S", f"c{i}") for i in range(15)
    ]
    with patch(
        "open_notebook.domain.notebook.vector_search_chunks",
        new=AsyncMock(return_value=results),
    ), patch.dict("os.environ", {"OPEN_NOTEBOOK_RETRIEVED_PASSAGES": "5"}):
        passages = await retrieve_relevant_passages("question", context)

    assert len(passages) == 5


@pytest.mark.asyncio
async def test_empty_context_and_query_return_empty():
    assert await retrieve_relevant_passages("", {"sources": []}) == []
    assert await retrieve_relevant_passages("q", None) == []
    assert await retrieve_relevant_passages("q", {"sources": []}) == []


@pytest.mark.asyncio
async def test_retrieval_failure_is_tolerated():
    """A retrieval failure never breaks the chat."""
    with patch(
        "open_notebook.domain.notebook.vector_search_chunks",
        new=AsyncMock(side_effect=Exception("db hiccup")),
    ):
        passages = await retrieve_relevant_passages("q", _context(["source:s1"]))

    assert passages == []


@pytest.mark.asyncio
async def test_multi_query_unions_results():
    """Query expansion surfaces passages a single query would miss."""
    context = _context(["source:s1"])
    porphyria = _result("source_embedding:porph1", "source:s1", "S", "porphyria", 0.4)
    mechanisms = _result("source_embedding:mt2x", "source:s1", "S", "MT2 receptor", 0.5)
    base = _result("source_embedding:base1", "source:s1", "S", "hypnotics", 0.9)

    def side_effect(query, results, source, note, minimum_score=0.2):
        if "porphyria" in query:
            return [porphyria]
        if "receptor subtypes" in query:
            return [mechanisms]
        return [base]

    with patch(
        "open_notebook.domain.notebook.vector_search_chunks",
        new=AsyncMock(side_effect=side_effect),
    ):
        passages = await retrieve_relevant_passages("hypnotic drugs", context)

    ids = [p["id"] for p in passages]
    # union of all three variants, ranked by similarity (base first)
    assert set(ids) == {"source_embedding:base1", "source_embedding:porph1", "source_embedding:mt2x"}
    assert ids[0] == "source_embedding:base1"


def test_build_citation_references():
    from api.routers._chat_shared import build_citation_references

    passages = [
        {"number": 1, "id": "source_embedding:aaa", "parent_id": "source:x", "title": "T1"},
        {"number": 2, "id": "source_embedding:bbb", "parent_id": "source:y", "title": "T2"},
        {"id": "source_embedding:nonumber", "parent_id": "source:z"},  # skipped
    ]
    refs = build_citation_references(passages)
    assert refs == [
        {"number": 1, "type": "source_embedding", "id": "aaa", "parent_id": "source:x", "title": "T1"},
        {"number": 2, "type": "source_embedding", "id": "bbb", "parent_id": "source:y", "title": "T2"},
    ]
    assert build_citation_references(None) == []
    assert build_citation_references([]) == []
