"""Context building for chat and podcast generation.

This is the single implementation behind:

- ``POST /api/chat/context`` (`api/routers/chat.py`) — assembles notebook
  context from a source/note inclusion config, via
  :func:`build_notebook_context`.
- the source-chat graph (`open_notebook/graphs/source_chat.py`) — assembles
  a single source plus its insights under a token budget, via
  :func:`build_source_context`.

The inclusion config uses string matching on human-readable status values
("not in context", "insights", "full content"). That protocol is shared with
the frontend — do not change it here.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from open_notebook.domain.notebook import (
    Note,
    Notebook,
    Source,
    SourceInsight,
)
from open_notebook.exceptions import DatabaseOperationError, NotFoundError

from .token_utils import token_count

SOURCE_TRUNCATION_NOTICE = (
    "\n\n[Source content truncated to fit the context token budget.]"
)
SOURCE_INSIGHT_BUDGET_RATIO = 0.2
TOKEN_PREFIX_VALIDATION_WINDOW = 16
_TOKENIZER_UNSET = object()


def format_source_context(context_data: Dict[str, Any]) -> str:
    """Render the exact source/insight Markdown supplied to Source Chat."""
    context_parts = []

    if context_data.get("sources"):
        context_parts.append("## SOURCE CONTENT")
        for source in context_data["sources"]:
            if isinstance(source, dict):
                context_parts.append(f"**Source ID:** {source.get('id', 'Unknown')}")
                context_parts.append(f"**Title:** {source.get('title', 'No title')}")
                full_text = source.get("full_text")
                if isinstance(full_text, str) and full_text.strip():
                    context_parts.append(f"**Content:**\n{full_text}")
                else:
                    context_parts.append(
                        "**Content:**\n[Source text is unavailable in this context.]"
                    )
                context_parts.append("")

    if context_data.get("insights"):
        context_parts.append("## SOURCE INSIGHTS")
        for insight in context_data["insights"]:
            if isinstance(insight, dict):
                context_parts.append(
                    f"**Insight ID:** {insight.get('id', 'Unknown')}"
                )
                context_parts.append(
                    f"**Type:** {insight.get('insight_type', 'Unknown')}"
                )
                context_parts.append(
                    f"**Content:** {insight.get('content', 'No content')}"
                )
                context_parts.append("")

    return "\n".join(context_parts)


def _rendered_source_context_tokens(
    source: Optional[Dict[str, Any]],
    insights: list[Dict[str, Any]],
    encoding: Any = _TOKENIZER_UNSET,
) -> int:
    """Count the same Markdown payload that Source Chat sends to its prompt."""
    rendered = format_source_context(
        {
            "sources": [source] if source is not None else [],
            "insights": insights,
        }
    )
    if encoding is _TOKENIZER_UNSET:
        return token_count(rendered)
    if encoding is None:
        return int(len(rendered.split()) * 1.3)
    return len(encoding.encode(rendered, disallowed_special=()))


def _ensure_prefix(table: str, record_id: str) -> str:
    """Ensure a record ID carries its table prefix (`table:id`)."""
    prefix = f"{table}:"
    return record_id if record_id.startswith(prefix) else f"{prefix}{record_id}"


def _truncate_source_to_token_budget(
    source_context: Dict[str, Any],
    max_tokens: int,
    insights: Optional[list[Dict[str, Any]]] = None,
    *,
    encoding: Any = _TOKENIZER_UNSET,
    source_tokens: Optional[list[int]] = None,
    source_is_over_budget: bool = False,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Truncate source text to a token budget while retaining source metadata.

    A near-maximal token-aligned prefix is retained with bounded validation for
    local BPE non-monotonicity. The truncation notice is part of the budget so
    downstream formatters never need a second size policy.

    Args:
        source_context: Long source context containing ``full_text``.
        max_tokens: Maximum tokens available for rendered source context.
        insights: Already selected insights that share the rendered budget.
        encoding: Reusable tokenizer, ``None`` for the word-count fallback, or
            the internal sentinel when the helper should resolve it itself.
        source_tokens: Precomputed tokens for ``full_text``.
        source_is_over_budget: Skip recounting the full rendered source when
            the caller has already established that it exceeds ``max_tokens``.

    Returns:
        The budgeted source context (or ``None`` when even an explicit notice
        cannot fit) and whether its text was truncated.
    """
    selected_insights = insights or []
    if not source_is_over_budget and (
        _rendered_source_context_tokens(
            source_context,
            selected_insights,
            encoding,
        )
        <= max_tokens
    ):
        return source_context, False

    full_text = source_context.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        return None, False

    def candidate(prefix: str) -> Dict[str, Any]:
        return {
            **source_context,
            "full_text": prefix + SOURCE_TRUNCATION_NOTICE,
        }

    # A very small budget may not even fit source metadata plus the notice.
    # Omit the item; the caller records an explicit status in context metadata.
    notice_only = candidate("")
    if (
        _rendered_source_context_tokens(
            notice_only,
            selected_insights,
            encoding,
        )
        > max_tokens
    ):
        return None, True

    def find_fitting_prefix(
        upper_bound: int,
        prefix_for_count,
        validation_window: int,
    ) -> Optional[Dict[str, Any]]:
        """Find a large fitting prefix with bounded candidate re-encoding."""
        evaluated: Dict[int, tuple[Dict[str, Any], bool]] = {}

        def evaluate(prefix_count: int) -> tuple[Dict[str, Any], bool]:
            if prefix_count not in evaluated:
                prefix = prefix_for_count(prefix_count)
                truncated = candidate(prefix)
                evaluated[prefix_count] = (
                    truncated,
                    bool(prefix.strip())
                    and _rendered_source_context_tokens(
                        truncated,
                        selected_insights,
                        encoding,
                    )
                    <= max_tokens,
                )
            return evaluated[prefix_count]

        low = 1
        high = upper_bound
        best_count = 0
        best: Optional[Dict[str, Any]] = None
        while low <= high:
            midpoint = (low + high) // 2
            truncated, fits = evaluate(midpoint)
            if fits:
                best_count = midpoint
                best = truncated
                low = midpoint + 1
            else:
                high = midpoint - 1

        # BPE counts can be locally non-monotonic. Check a bounded window past
        # the binary-search result without stopping at an over-budget candidate.
        validation_start = best_count + 1 if best_count else 1
        validation_end = min(
            upper_bound,
            best_count + validation_window,
        )
        for prefix_count in range(validation_start, validation_end + 1):
            truncated, fits = evaluate(prefix_count)
            if fits and prefix_count > best_count:
                best_count = prefix_count
                best = truncated

        return best

    try:
        if encoding is _TOKENIZER_UNSET:
            import tiktoken

            encoding = tiktoken.get_encoding("o200k_base")
        if encoding is None:
            raise ImportError
        if source_tokens is None:
            source_tokens = encoding.encode(full_text, disallowed_special=())

        def token_prefix(prefix_token_count: int) -> str:
            return encoding.decode_bytes(
                source_tokens[:prefix_token_count]
            ).decode("utf-8", errors="ignore")

        best = find_fitting_prefix(
            min(len(source_tokens), max_tokens),
            token_prefix,
            TOKEN_PREFIX_VALIDATION_WINDOW,
        )
        if best is not None:
            return best, True
    except (ImportError, OSError):
        # Match token_count's offline fallback with deterministic word-boundary
        # prefixes. Its split-based estimate is monotonic, so binary search does
        # not need the tokenizer path's validation window.
        word_ends = [match.end() for match in re.finditer(r"\S+", full_text)]

        def word_prefix(word_count: int) -> str:
            return full_text[: word_ends[word_count - 1]]

        best = find_fitting_prefix(
            min(len(word_ends), max_tokens),
            word_prefix,
            0,
        )
        if best is not None:
            return best, True

    # A notice without any source characters is not useful source context.
    # Omit it so callers can report ``omitted_budget`` honestly.
    return None, True


async def build_notebook_context(
    notebook: Notebook,
    context_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, list], str]:
    """Assemble source/note context for a notebook.

    With a config, each entry's status string decides inclusion: "not in"
    skips it, "insights" includes the short source context, "full content"
    includes the long context (notes only support "full content"). Without a
    config, every source and note is included with its short context.

    Failures on individual items are logged and skipped — one broken record
    never fails the whole request.

    Returns:
        ({"sources": [...], "notes": [...]}, concatenated str() of every
        included context dict — used for token/char counting).
    """
    context_data: Dict[str, list] = {"sources": [], "notes": []}
    total_content = ""

    if context_config:
        for source_id, status in context_config.get("sources", {}).items():
            if "not in" in status:
                continue

            try:
                full_source_id = _ensure_prefix("source", source_id)

                try:
                    source = await Source.get(full_source_id)
                except Exception:
                    continue

                if "insights" in status:
                    source_context = await source.get_context(context_size="short")
                    chunks = await _source_citable_chunks(
                        source, limit=_chunks_per_source_limit("insights")
                    )
                    if chunks:
                        source_context["chunks"] = chunks
                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
                elif "full content" in status:
                    source_context = await source.get_context(context_size="long")
                    chunks = await _source_citable_chunks(
                        source, limit=_chunks_per_source_limit("full")
                    )
                    if chunks:
                        # Chunks carry the same text as citable passages; drop
                        # the uncitable blob so content isn't duplicated.
                        source_context["chunks"] = chunks
                        source_context["full_text"] = None
                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
            except Exception as e:
                logger.warning(f"Error processing source {source_id}: {str(e)}")
                continue

        for note_id, status in context_config.get("notes", {}).items():
            if "not in" in status:
                continue

            try:
                full_note_id = _ensure_prefix("note", note_id)
                note = await Note.get(full_note_id)
                if not note:
                    continue

                if "full content" in status:
                    note_context = note.get_context(context_size="long")
                    context_data["notes"].append(note_context)
                    total_content += str(note_context)
            except Exception as e:
                logger.warning(f"Error processing note {note_id}: {str(e)}")
                continue
    else:
        # Default behavior - include all sources and notes with short context
        sources = await notebook.get_sources()
        try:
            insights_by_source = await SourceInsight.get_for_sources(
                [source.id for source in sources if source.id]
            )
        except Exception as e:
            # Match the per-source fallback below: a hiccup fetching
            # insights shouldn't fail the whole context request.
            logger.warning(f"Error batch-fetching source insights: {str(e)}")
            insights_by_source = {}
        for source in sources:
            try:
                source_context = await source.get_context(
                    context_size="short",
                    insights=insights_by_source.get(source.id or "", []),
                )
                chunks = await _source_citable_chunks(
                    source, limit=_chunks_per_source_limit("insights")
                )
                if chunks:
                    source_context["chunks"] = chunks
                context_data["sources"].append(source_context)
                total_content += str(source_context)
            except Exception as e:
                logger.warning(f"Error processing source {source.id}: {str(e)}")
                continue

        notes = await notebook.get_notes()
        for note in notes:
            try:
                note_context = note.get_context(context_size="short")
                context_data["notes"].append(note_context)
                total_content += str(note_context)
            except Exception as e:
                logger.warning(f"Error processing note {note.id}: {str(e)}")
                continue

    return context_data, total_content


async def _source_citable_chunks(source: Any, limit: int) -> list:
    """Best-effort per-passage chunks for a source; ``[]`` when unavailable.

    Uses ``getattr`` so callers/tests that stub sources without the method
    (or a source with no embeddings) keep the previous context shape.
    """
    loader = getattr(source, "get_citable_chunks", None)
    if loader is None:
        return []
    try:
        chunks = await loader(limit=limit)
    except Exception as e:
        logger.warning(f"Failed to load citable chunks for source {source.id}: {e}")
        return []
    return chunks or []


def _chunks_per_source_limit(context_size: str) -> int:
    """Max citable passages included per source for notebook chat context.

    Configurable so the context stays inside the chat model's window:
    - ``OPEN_NOTEBOOK_CHUNKS_PER_SOURCE`` (default 12) for "insights" mode.
    - ``OPEN_NOTEBOOK_CHUNKS_PER_SOURCE_FULL`` (default 1000) for "full
      content" mode, which replaces the uncitable full_text blob.
    """
    if context_size == "full":
        env_name, default = "OPEN_NOTEBOOK_CHUNKS_PER_SOURCE_FULL", 1000
    else:
        env_name, default = "OPEN_NOTEBOOK_CHUNKS_PER_SOURCE", 12
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


async def retrieve_relevant_passages(
    query: str, context: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Vector-retrieve the source passages most relevant to ``query``.

    Notebook chat context is static (insights + prefix chunks), but the
    passages a specific question needs may live anywhere in a long source.
    This runs a chunk-level vector search and keeps only results whose parent
    source is part of the current context, so citations always point at
    sources the user selected. Best-effort: any failure returns ``[]``.

    A single embedding of the question can miss high-yield facts (e.g. a
    question about "hypnotic drugs" won't surface "porphyria" or "CYP1A2"),
    so the query is expanded into several variants and the results are
    unioned by passage id before capping.
    """
    if not query or not context:
        return []
    selected_ids: set = set()
    for source in context.get("sources") or []:
        sid = source.get("id") if isinstance(source, dict) else None
        if sid:
            selected_ids.add(str(sid).split(":")[-1])
    if not selected_ids:
        return []

    try:
        from open_notebook.domain.notebook import vector_search_chunks
    except Exception:
        return []

    # Query variants: the question itself, plus expansions that pull in
    # high-yield safety/kinetics facts and mechanism/selection detail that a
    # plain question embedding often misses.
    query_variants = _retrieval_query_variants(query)

    scored: Dict[str, tuple] = {}
    for variant in query_variants:
        try:
            results = await vector_search_chunks(variant, 50, source=True, note=False)
        except Exception as e:
            logger.warning(f"Passage retrieval failed for chat message: {e}")
            continue
        for result in results or []:
            parent = str(result.get("parent_id") or "").split(":")[-1]
            if parent not in selected_ids:
                continue
            rid = str(result.get("id") or "")
            if not rid:
                continue
            matches = result.get("matches")
            content = (
                matches
                if isinstance(matches, str)
                else " ".join(matches or [])
            )
            if not content:
                continue
            similarity = result.get("similarity") or 0
            if rid not in scored or similarity > scored[rid][0]:
                scored[rid] = (
                    similarity,
                    {
                        "id": rid,
                        "parent_id": result.get("parent_id"),
                        "title": result.get("title"),
                        "content": content,
                    },
                )

    # How many relevant passages to hand the model (configurable; more passages
    # = more distinct citable units to spread citations across).
    raw_limit = os.getenv("OPEN_NOTEBOOK_RETRIEVED_PASSAGES")
    try:
        max_passages = max(1, int(raw_limit)) if raw_limit else 30
    except ValueError:
        max_passages = 30

    ranked = [entry for _, entry in sorted(scored.values(), key=lambda kv: kv[0], reverse=True)]
    return ranked[:max_passages]


_HIGH_YIELD_EXPANSION = (
    " Also cover: contraindications, porphyria, overdose, toxicity, "
    "respiratory depression, drug interactions, metabolism (CYP enzymes), "
    "half-lives, active metabolites, withdrawal, dependence."
)

_MECHANISM_EXPANSION = (
    " Also cover: receptor subtypes (alpha1, alpha2/3, MT1, MT2, OX1, OX2), "
    "mechanism of action, GABA-A receptor pharmacology, clinical selection, "
    "adverse effects."
)


def _retrieval_query_variants(query: str) -> List[str]:
    """Deterministic query expansion for passage retrieval."""
    return [
        query,
        f"{query}{_HIGH_YIELD_EXPANSION}",
        f"{query}{_MECHANISM_EXPANSION}",
    ]


async def build_source_context(
    source_id: str, max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """Assemble a single source's full text plus its insights.

    Used by the source-chat graph. If ``max_tokens`` is given, the source text
    is kept in full when it fits, then insights are retained in fetch order
    while space remains. When the source alone exceeds the budget, a bounded
    share is reserved for insights and the source is explicitly truncated into
    the remaining space instead of being dropped.

    Returns a dict with "sources", "notes" (always empty), "insights",
    "total_tokens", "total_items" and per-type counts in "metadata".
    """
    try:
        sources: list = []
        insights: list = []
        source_truncated = False
        source_text_status = "not_found"

        try:
            full_source_id = _ensure_prefix("source", source_id)
            source = await Source.get(full_source_id)
        except NotFoundError:
            source = None

        if source:
            insight_objects = await source.get_insights()
            source_context = await source.get_context(
                context_size="long",
                insights=insight_objects,
            )
            # Insights have their own budgeted items below. Keeping the nested
            # copy would double-count them while the formatter ignores it.
            source_context = {**source_context, "insights": []}
            budgeted_source: Optional[Dict[str, Any]] = source_context

            insight_items: list[Dict[str, Any]] = []
            for insight in insight_objects:
                insight_content = {
                    "id": insight.id,
                    "source_id": source.id,
                    "insight_type": insight.insight_type,
                    "content": insight.content,
                }
                insight_items.append(insight_content)

            try:
                import tiktoken

                encoding = tiktoken.get_encoding("o200k_base")
            except (ImportError, OSError):
                encoding = None

            rendered_source_tokens = _rendered_source_context_tokens(
                source_context,
                [],
                encoding,
            )

            if max_tokens is None:
                insights.extend(insight_items)
            elif rendered_source_tokens > max_tokens:
                # Large documents would otherwise consume the entire budget and
                # silently remove every insight. Reserve a bounded share for
                # insights, then give all unused space back to the source text.
                insight_budget = int(max_tokens * SOURCE_INSIGHT_BUDGET_RATIO)
                for insight_content in insight_items:
                    candidate_insights = [*insights, insight_content]
                    if (
                        _rendered_source_context_tokens(
                            None,
                            candidate_insights,
                            encoding,
                        )
                        > insight_budget
                    ):
                        continue
                    insights.append(insight_content)

                full_text = source_context.get("full_text")
                encoded_source = (
                    encoding.encode(full_text, disallowed_special=())
                    if encoding is not None and isinstance(full_text, str)
                    else None
                )

                budgeted_source, source_truncated = _truncate_source_to_token_budget(
                    source_context,
                    max_tokens,
                    insights,
                    encoding=encoding,
                    source_tokens=encoded_source,
                    source_is_over_budget=True,
                )
            else:
                for insight_content in insight_items:
                    candidate_insights = [*insights, insight_content]
                    if _rendered_source_context_tokens(
                        source_context,
                        candidate_insights,
                        encoding,
                    ) > max_tokens:
                        continue
                    insights.append(insight_content)

            if budgeted_source is None:
                source_text_status = "omitted_budget"
            else:
                full_text = budgeted_source.get("full_text")
                if isinstance(full_text, str) and full_text.strip():
                    source_text_status = (
                        "truncated" if source_truncated else "available"
                    )
                else:
                    source_text_status = "missing"

            if budgeted_source is not None:
                sources.append(budgeted_source)
            total_tokens = _rendered_source_context_tokens(
                budgeted_source,
                insights,
                encoding,
            )
        else:
            logger.warning(f"Source {source_id} not found")
            total_tokens = 0

        total_items = len(sources) + len(insights)
        logger.info(f"Built context with {total_items} items, {total_tokens} tokens")

        return {
            "sources": sources,
            "notes": [],
            "insights": insights,
            "total_tokens": total_tokens,
            "total_items": total_items,
            "metadata": {
                "source_count": len(sources),
                "note_count": 0,
                "insight_count": len(insights),
                "source_text_status": source_text_status,
                "source_truncated": source_truncated,
            },
        }
    except Exception as e:
        logger.error(f"Error building context: {str(e)}")
        raise DatabaseOperationError(f"Failed to build context: {str(e)}")
