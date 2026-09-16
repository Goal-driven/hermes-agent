"""Immutable, session-start capability routing.

Routing is an optimization layer, never an authorization layer: callers pass the
already-authorized tool definitions, and every definition is partitioned into a
direct or deferred set.  The deferred set remains reachable through Tool Search.
The plan is deterministic and serializable so a resumed/compressed session can
restore the exact same model-visible tool prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


CAPABILITY_PLAN_VERSION = 1
_DEFAULT_KERNEL_TOOLS = (
    "clarify",
    "skills_list",
    "skill_view",
    "tool_search",
    "tool_describe",
    "tool_call",
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from",
    "help", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
    "that", "the", "this", "to", "we", "with", "you",
})


def _tool_name(tool_def: Mapping[str, Any]) -> str:
    fn = tool_def.get("function")
    return str(fn.get("name") or "") if isinstance(fn, Mapping) else ""


def _schema_tokens(tool_def: Mapping[str, Any]) -> int:
    try:
        encoded = json.dumps(tool_def, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(tool_def)
    return max(1, math.ceil(len(encoded) / 4))


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token for token in _TOKEN_RE.findall(str(value or "").lower())
        if token not in _STOPWORDS
    )


def _manifest_text(manifest: Mapping[str, Any]) -> str:
    parts: list[str] = [str(manifest.get("description") or "")]
    for field in ("routing_keywords", "routing_examples"):
        value = manifest.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Sequence):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _manifest_score(query: str, manifest: Mapping[str, Any]) -> int:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0
    text = _manifest_text(manifest).lower()
    manifest_tokens = _tokens(text)
    score = 2 * len(query_tokens & manifest_tokens)
    keywords = manifest.get("routing_keywords") or ()
    if isinstance(keywords, str):
        keywords = (keywords,)
    lowered = query.lower()
    for keyword in keywords if isinstance(keywords, Sequence) else ():
        phrase = str(keyword).strip().lower()
        if phrase and phrase in lowered:
            score += 4 if " " in phrase else 2
    examples = manifest.get("routing_examples") or ()
    if isinstance(examples, str):
        examples = (examples,)
    for example in examples if isinstance(examples, Sequence) else ():
        overlap = query_tokens & _tokens(example)
        if overlap:
            score += len(overlap)
    try:
        score += int(manifest.get("routing_priority") or 0)
    except (TypeError, ValueError):
        pass
    return max(0, score)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _default_budget(context_length: int | None) -> int:
    """A provider-agnostic schema budget, not a fixed tool-count ceiling."""
    if isinstance(context_length, int) and context_length > 0:
        return max(2_000, min(8_000, int(context_length * 0.03)))
    return 6_000


@dataclass(frozen=True)
class CapabilityPlan:
    """Versioned partition of an already-authorized tool surface."""

    version: int
    direct_tools: tuple[str, ...]
    deferred_tools: tuple[str, ...]
    matched_toolsets: tuple[str, ...]
    direct_schema_tokens: int
    intent_hash: str
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "direct_tools": list(self.direct_tools),
            "deferred_tools": list(self.deferred_tools),
            "matched_toolsets": list(self.matched_toolsets),
            "direct_schema_tokens": self.direct_schema_tokens,
            "intent_hash": self.intent_hash,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityPlan":
        if int(value.get("version") or 0) != CAPABILITY_PLAN_VERSION:
            raise ValueError("unsupported capability plan version")
        return cls(
            version=CAPABILITY_PLAN_VERSION,
            direct_tools=tuple(str(name) for name in value.get("direct_tools") or ()),
            deferred_tools=tuple(str(name) for name in value.get("deferred_tools") or ()),
            matched_toolsets=tuple(str(name) for name in value.get("matched_toolsets") or ()),
            direct_schema_tokens=int(value.get("direct_schema_tokens") or 0),
            intent_hash=str(value.get("intent_hash") or ""),
            manifest_hash=str(value.get("manifest_hash") or ""),
        )


def build_capability_plan(
    user_message: Any,
    *,
    tool_defs: Iterable[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    direct_token_budget: int | None = None,
    context_length: int | None = None,
    kernel_tools: Sequence[str] = _DEFAULT_KERNEL_TOOLS,
) -> CapabilityPlan:
    """Build a deterministic direct/deferred partition.

    ``tool_defs`` is the authorization boundary. Manifests can rank only tools
    present in that input, so plugin metadata cannot grant a capability. Unknown
    or unranked tools are deferred, never discarded.
    """
    query = str(user_message or "")
    definitions = [dict(tool_def) for tool_def in tool_defs if _tool_name(tool_def)]
    names = [_tool_name(tool_def) for tool_def in definitions]
    by_name = dict(zip(names, definitions))
    allowed = frozenset(names)
    costs = {name: _schema_tokens(by_name[name]) for name in names}
    budget = max(1, int(direct_token_budget or _default_budget(context_length)))

    scored: list[tuple[int, str, Mapping[str, Any]]] = []
    for manifest_name, manifest in manifests.items():
        if not isinstance(manifest, Mapping):
            continue
        score = _manifest_score(query, manifest)
        if score > 0:
            scored.append((score, str(manifest_name), manifest))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: set[str] = {name for name in kernel_tools if name in allowed}
    used = sum(costs[name] for name in selected)
    matched: list[str] = []
    for _score, manifest_name, manifest in scored:
        candidates = [str(name) for name in manifest.get("tools") or () if str(name) in allowed]
        if not candidates:
            continue
        accepted = False
        for name in candidates:
            if name in selected:
                accepted = True
                continue
            cost = costs[name]
            if used + cost <= budget:
                selected.add(name)
                used += cost
                accepted = True
        if accepted:
            matched.append(manifest_name)

    direct = tuple(name for name in names if name in selected)
    deferred = tuple(name for name in names if name not in selected)
    manifest_projection = {
        str(name): {
            "description": manifest.get("description"),
            "tools": list(manifest.get("tools") or ()),
            "routing_keywords": list(manifest.get("routing_keywords") or ())
                if not isinstance(manifest.get("routing_keywords"), str)
                else [manifest.get("routing_keywords")],
            "routing_examples": list(manifest.get("routing_examples") or ())
                if not isinstance(manifest.get("routing_examples"), str)
                else [manifest.get("routing_examples")],
            "routing_priority": manifest.get("routing_priority", 0),
        }
        for name, manifest in sorted(manifests.items()) if isinstance(manifest, Mapping)
    }
    return CapabilityPlan(
        version=CAPABILITY_PLAN_VERSION,
        direct_tools=direct,
        deferred_tools=deferred,
        matched_toolsets=tuple(matched),
        direct_schema_tokens=sum(costs[name] for name in direct),
        intent_hash=_canonical_hash({"query": query}),
        manifest_hash=_canonical_hash(manifest_projection),
    )


__all__ = ["CAPABILITY_PLAN_VERSION", "CapabilityPlan", "build_capability_plan"]
