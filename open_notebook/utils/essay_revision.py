"""Two-pass answer refinement for notebook chat.

After the chat model produces a draft answer, this module:
1. **Grounding check** — a batched judge verifies each numbered citation
   against its cited passage (support is decided from passage text only).
2. **Revision** — the draft is revised with the grounding verdicts, fixing or
   removing unsupported claims, filling gaps from the retrieved passages, and
   enforcing citation discipline.

Both passes are best-effort: any failure falls back to the original draft so
chat never breaks. Enable/disable with ``OPEN_NOTEBOOK_CHAT_GROUNDING``
(default: enabled).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from ai_prompter import Prompter
from loguru import logger

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.utils.text_utils import clean_thinking_content, extract_text_content

# Matches [1], [7, 8], [1, 9, 12] at the end of a sentence (optional trailing
# punctuation after the closing bracket, e.g. "[3].").
_CITATION_AT_END = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]\s*[.!?]?\s*$")
# Matches any [n] / [n, m] citation anywhere (for cleanup).
_CITATION_ANYWHERE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def is_grounding_enabled() -> bool:
    """Whether the grounding+revision pipeline is enabled (env override)."""
    raw = os.getenv("OPEN_NOTEBOOK_CHAT_GROUNDING")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


def extract_citation_claims(text: str) -> List[Dict[str, Any]]:
    """Split a draft into (citation number, claim) pairs.

    Claims are sentences ending in a citation marker like ``[3]`` or
    ``[1, 3, 5]``; each number in a grouped marker yields one entry.
    """
    claims: List[Dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        match = _CITATION_AT_END.search(sentence)
        if not match:
            continue
        claim = sentence[: match.start()].strip()
        if not claim:
            continue
        for raw in match.group(1).split(","):
            claims.append({"number": int(raw.strip()), "claim": claim})
    return claims


def drop_unsupported_citations(
    text: str, unsupported_numbers: List[int]
) -> str:
    """Remove citation markers for unsupported numbers (deterministic cleanup).

    A marker whose numbers all become unsupported is removed entirely; a
    grouped marker keeps only its supported numbers.
    """
    unsupported = set(unsupported_numbers)
    if not unsupported:
        return text

    def _replace(match: "re.Match") -> str:
        numbers = [int(raw.strip()) for raw in match.group(1).split(",")]
        kept = [str(n) for n in numbers if n not in unsupported]
        return f"[{', '.join(kept)}]" if kept else ""

    return _CITATION_ANYWHERE.sub(_replace, text)


def _parse_grounding_json(content: str) -> List[Dict[str, Any]]:
    """Robustly parse the judge's JSON response (may be wrapped in fences)."""
    content = clean_thinking_content(content)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    return [
        {"number": int(r.get("number", 0)), "supported": bool(r.get("supported", False)), "reason": str(r.get("reason", ""))}
        for r in results
        if isinstance(r, dict) and r.get("number") is not None
    ]


async def _run_llm(
    prompt_template: str,
    data: Dict[str, Any],
    model_id: Optional[str],
    max_tokens: int = 4000,
) -> str:
    system_prompt = Prompter(prompt_template=prompt_template).render(data=data)
    model = await provision_langchain_model(
        system_prompt, model_id, "chat", max_tokens=max_tokens
    )
    from langchain_core.messages import SystemMessage

    message = await model.ainvoke([SystemMessage(content=system_prompt)])
    return extract_text_content(message.content)


async def refine_chat_answer(
    question: str,
    draft: str,
    retrieved_passages: Optional[List[Dict[str, Any]]],
    model_id: Optional[str],
) -> str:
    """Grounding-check + revise a draft answer. Best-effort: returns the
    original draft unchanged on any failure or when disabled."""
    if not is_grounding_enabled() or not draft or not retrieved_passages:
        return draft

    try:
        claims = extract_citation_claims(draft)
        if not claims:
            logger.info("Answer refinement skipped: no citable claims found")
            return draft

        cited_numbers = sorted({claim["number"] for claim in claims})
        cited_passages = [
            p
            for p in retrieved_passages
            if (p.get("number") or 0) in set(cited_numbers)
        ]
        if not cited_passages:
            logger.info("Answer refinement skipped: no retrieved passages cited")
            return draft

        verdicts = _parse_grounding_json(
            await _run_llm(
                "chat/grounding_check",
                {"claims": claims, "cited_passages": cited_passages},
                model_id,
                max_tokens=2000,
            )
        )
        unsupported = [v for v in verdicts if not v.get("supported", True)]

        revised = await _run_llm(
            "chat/revise",
            {
                "question": question,
                "draft": draft,
                "retrieved_passages": retrieved_passages,
                "unsupported_citations": unsupported,
            },
            model_id,
            max_tokens=8192,
        )

        # Deterministic backstop: remove any citation the judge flagged and the
        # editor failed to drop (e.g. it re-cited without fixing the claim).
        unsupported_numbers = [v.get("number") for v in unsupported if v.get("number")]
        revised = drop_unsupported_citations(revised, unsupported_numbers)
        final = revised.strip() if revised.strip() else draft

        logger.info(
            "Answer refined: %d claims checked, %d unsupported, %s",
            len(claims),
            len(unsupported),
            "revised" if final != draft else "kept draft (judge passed)",
        )
        return final
    except Exception:
        logger.exception("Chat answer refinement failed; keeping draft")
        return draft
