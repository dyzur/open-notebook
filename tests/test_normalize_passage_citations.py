"""Tests for normalize_passage_citations."""

from open_notebook.utils.text_utils import normalize_passage_citations


def test_restores_prefix_for_known_ids():
    content = (
        "Benzodiazepines increase chloride opening [ggy6omvsw9zisgtqj3e3] "
        "and are reversed by flumazenil [zxm4ukz1b817p4q2hljk]."
    )
    retrieved = [
        {"id": "source_embedding:ggy6omvsw9zisgtqj3e3", "content": "x"},
        {"id": "source_embedding:zxm4ukz1b817p4q2hljk", "content": "y"},
    ]
    out = normalize_passage_citations(content, retrieved)
    assert "[source_embedding:ggy6omvsw9zisgtqj3e3]" in out
    assert "[source_embedding:zxm4ukz1b817p4q2hljk]" in out
    assert "[ggy6omvsw9zisgtqj3e3]" not in out


def test_leaves_unknown_and_prefixed_ids_untouched():
    content = (
        "Prefixed stays [source_embedding:abcd1234efgh5678ijkl90] "
        "and unknown bare [unknownid1234567890abcdefg] stays bare."
    )
    retrieved = [{"id": "source_embedding:abcd1234efgh5678ijkl90", "content": "x"}]
    out = normalize_passage_citations(content, retrieved)
    assert "[source_embedding:abcd1234efgh5678ijkl90]" in out
    assert "[unknownid1234567890abcdefg]" in out


def test_no_retrieved_or_empty_content_is_noop():
    assert normalize_passage_citations("text [ggy6omvsw9zisgtqj3e3]", []) == (
        "text [ggy6omvsw9zisgtqj3e3]"
    )
    assert normalize_passage_citations("", [{"id": "source_embedding:x"}]) == ""
