"""Tests for the two-pass essay refinement (grounding check + revision)."""

from unittest.mock import AsyncMock, patch

import pytest

from open_notebook.utils.essay_revision import (
    drop_unsupported_citations,
    extract_citation_claims,
    is_grounding_enabled,
    refine_chat_answer,
)


def test_extract_citation_claims_single_and_grouped():
    text = (
        "Benzodiazepines increase chloride opening [3]. "
        "Barbiturates are less selective [7, 8]. "
        "No citation here. "
        "Melatonin acts outside GABA [16]."
    )
    claims = extract_citation_claims(text)
    assert claims == [
        {"number": 3, "claim": "Benzodiazepines increase chloride opening"},
        {"number": 7, "claim": "Barbiturates are less selective"},
        {"number": 8, "claim": "Barbiturates are less selective"},
        {"number": 16, "claim": "Melatonin acts outside GABA"},
    ]


def test_extract_citation_claims_empty():
    assert extract_citation_claims("Plain text without citations.") == []
    assert extract_citation_claims("") == []


def test_drop_unsupported_citations():
    text = "A [1]. B [2, 3]. C [4, 5]."
    out = drop_unsupported_citations(text, [2, 4])
    assert out == "A [1]. B [3]. C [5]."
    # all numbers unsupported -> marker removed
    out = drop_unsupported_citations("X [6, 7]", [6, 7])
    assert out == "X "
    # no-op when nothing to drop
    assert drop_unsupported_citations("A [1]", []) == "A [1]"


@pytest.mark.asyncio
async def test_refine_disabled_returns_draft():
    with patch.dict("os.environ", {"OPEN_NOTEBOOK_CHAT_GROUNDING": "off"}):
        out = await refine_chat_answer("q", "draft [1]", [], None)
    assert out == "draft [1]"


@pytest.mark.asyncio
async def test_refine_pipeline_runs_and_cleans_unsupported():
    """Grounding judge flags a citation; revise runs; backstop drops it."""
    draft = "Claim one [1]. Claim two [2]."
    passages = [
        {"number": 1, "id": "source_embedding:a", "content": "supports one"},
        {"number": 2, "id": "source_embedding:b", "content": "does not"},
    ]
    judge_payload = '{"results": [{"number": 1, "supported": true}, {"number": 2, "supported": false, "reason": "not in passage"}]}'
    revised = "Claim one [1]. Claim two."  # editor removes unsupported cite

    judge = AsyncMock(return_value=judge_payload)
    revise = AsyncMock(return_value=revised)

    with patch(
        "open_notebook.utils.essay_revision._run_llm", side_effect=[judge_payload, revised]
    ) as run_llm:
        out = await refine_chat_answer("q", draft, passages, "model-x")

    assert out == "Claim one [1]. Claim two."
    # two LLM calls: grounding check then revision
    assert run_llm.await_count == 2
    templates = [call.args[0] for call in run_llm.await_args_list]
    assert templates == ["chat/grounding_check", "chat/revise"]
    # revise prompt received the unsupported verdict
    revise_data = run_llm.await_args_list[1].args[1]
    assert revise_data["unsupported_citations"] == [
        {"number": 2, "supported": False, "reason": "not in passage"}
    ]
    assert revise_data["draft"] == draft
    assert revise_data["question"] == "q"


@pytest.mark.asyncio
async def test_refine_failure_keeps_draft():
    with patch(
        "open_notebook.utils.essay_revision._run_llm",
        new=AsyncMock(side_effect=Exception("boom")),
    ):
        out = await refine_chat_answer("q", "draft [1]", [{"number": 1, "content": "c"}], None)
    assert out == "draft [1]"


def test_grounding_env_defaults():
    with patch.dict("os.environ", {}, clear=False):
        pass  # default is enabled when unset
    assert is_grounding_enabled() is True
    for off in ("0", "false", "off", "no", "False"):
        with patch.dict("os.environ", {"OPEN_NOTEBOOK_CHAT_GROUNDING": off}):
            assert is_grounding_enabled() is False
