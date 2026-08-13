"""Typed contracts for Phase A1 skill extraction.

Everything the Gemini extractor emits must pass through these models before it
is allowed anywhere near the linker or the graph. Malformed model output is
rejected here, not silently repaired downstream.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

# Bounds are part of the contract in CLAUDE.md Step A1.2.
WEIGHT_MIN = 0.5
WEIGHT_MAX = 1.5
WEIGHT_DEFAULT = 1.0

_WHITESPACE = re.compile(r"\s+")


class ExtractedSkill(BaseModel):
    """A single skill mention extracted from one chunk of source text."""

    skill: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="The skill as named in the source text, e.g. 'Kubernetes'.",
    )
    weight: float = Field(
        ...,
        ge=WEIGHT_MIN,
        le=WEIGHT_MAX,
        description=(
            "Proficiency / emphasis signal. 0.5 = passing mention or nice-to-have, "
            "1.0 = ordinary working competence, 1.5 = core, heavily evidenced skill."
        ),
    )
    context: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Short verbatim-ish span from the source that justifies this skill.",
    )

    @field_validator("skill", "context", mode="before")
    @classmethod
    def _normalize_whitespace(cls, v: object) -> object:
        if isinstance(v, str):
            return _WHITESPACE.sub(" ", v).strip()
        return v

    @property
    def key(self) -> str:
        """Case-folded key used for pre-canonicalization deduplication."""
        return self.skill.casefold()


class ExtractionResult(BaseModel):
    """All skills extracted from one document, with provenance for auditing."""

    source_id: str
    doc_type: str = Field(default="unknown", description="'resume' | 'jd' | 'unknown'")
    skills: list[ExtractedSkill] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    chunk_count: int = 0
    failed_chunks: list[int] = Field(
        default_factory=list,
        description="Indices of chunks that never produced schema-valid output.",
    )

    @property
    def is_complete(self) -> bool:
        return not self.failed_chunks


def merge_skills(skills: Iterable[ExtractedSkill]) -> list[ExtractedSkill]:
    """Collapse duplicate mentions across chunks.

    Dedup is by case-folded surface string only. Real synonym resolution
    ('K8s' -> 'Kubernetes') is Step A2's job, not this function's.

    On collision we keep the highest weight, on the reasoning that the strongest
    evidence anywhere in the document is the best estimate of proficiency, and
    we keep that same mention's context so weight and justification stay paired.
    """
    best: dict[str, ExtractedSkill] = {}
    for s in skills:
        current = best.get(s.key)
        if current is None or s.weight > current.weight:
            best[s.key] = s
    return sorted(best.values(), key=lambda s: (-s.weight, s.skill.casefold()))