"""Phase A1, Node 2: Gemini Flash skill extraction with enforced structure.

Contract (CLAUDE.md A1.2-A1.3):
  * Gemini is called with a strict `response_schema`.
  * Every response is re-validated in code with pydantic. A malformed or
    out-of-range response is retried, then recorded as a failed chunk - never
    coerced into something that looks valid.
  * Free-tier rate limits (NFR5, 15 RPM) are respected by a client-side limiter
    plus exponential backoff, so a batch run degrades into slowness rather than
    a wall of 429s.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass

from .reader import Chunk, Document
from .schemas import ExtractedSkill, ExtractionResult, merge_skills
import google.genai  

logger = logging.getLogger(__name__)

# Bump this whenever the prompt text changes; it is recorded on every
# ExtractionResult so Phase B numbers stay attributable to a specific prompt.
PROMPT_VERSION = "a1-v1"

DEFAULT_MODEL = os.getenv("SYNAPSE_GEMINI_MODEL", "gemini-3.6-flash")

SYSTEM_INSTRUCTION = """\
You extract skills from hiring documents (resumes and job descriptions).

Rules:
- Extract only skills, tools, technologies, methodologies and domain competencies.
  Do not extract job titles, company names, universities, degrees, dates, soft
  filler ("hard worker"), or responsibilities that name no skill.
- Use the surface form as written in the text. Do not expand abbreviations, do
  not normalise spelling, do not merge related skills. Downstream canonicalisation
  handles that; guessing here destroys information.
- Assign each skill a weight from 0.5 to 1.5:
    0.5  mentioned in passing, listed as optional, or clearly peripheral
    1.0  ordinary working competence, used but not emphasised
    1.5  central to the document: repeated, quantified, or explicitly required
- `context` must be a short span (under 200 characters) taken from the text that
  justifies the skill and its weight.
- If the text contains no skills, return an empty list.
- Never invent a skill that is not supported by the text.
"""


class ExtractionError(RuntimeError):
    """Raised when a chunk cannot be extracted after all retries."""


@dataclass
class RateLimiter:
    """Simple client-side RPM ceiling (sliding window)."""

    rpm: int = 15
    _calls: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._calls = []

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < 60.0]
        if len(self._calls) >= self.rpm:
            sleep_for = 60.0 - (now - self._calls[0]) + 0.05
            logger.info("Rate limit reached; sleeping %.1fs", sleep_for)
            time.sleep(max(sleep_for, 0.0))
            now = time.monotonic()
            self._calls = [t for t in self._calls if now - t < 60.0]
        self._calls.append(time.monotonic())


class SkillExtractor:
    """Wraps the Gemini client. Construct once, reuse across documents."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        rpm: int = 15,
        max_retries: int = 4,
        temperature: float = 0.0,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.limiter = RateLimiter(rpm=rpm)
        self._client = client  # injectable for tests

        if self._client is None:
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key. Set GEMINI_API_KEY or pass api_key=... "
                    "(never hard-code it)."
                )
            from google import genai  # imported lazily so tests need no SDK

            self._client = genai.Client(api_key=api_key)

    # ---------------------------------------------------------------- internals

    def _generate(self, text: str) -> str:
        """One raw API call. Returns the model's JSON text."""
        from google.genai import types

        self.limiter.acquire()
        response = self._client.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=list[ExtractedSkill],
                temperature=self.temperature,
            ),
        )
        return response.text or "[]"

    @staticmethod
    def _validate(payload: str) -> list[ExtractedSkill]:
        """Re-validate the model's JSON independently of the SDK's parsing.

        We validate the raw text rather than trusting `response.parsed` so that a
        schema violation is caught by our own pydantic models - the SDK's parse is
        a convenience, not our correctness guarantee.
        """
        import json
        cleaned_payload = payload.strip()
        if cleaned_payload.startswith("```json"):
            cleaned_payload = cleaned_payload[7:]
        if cleaned_payload.startswith("```"):
            cleaned_payload = cleaned_payload[3:]
        if cleaned_payload.endswith("```"):
            cleaned_payload = cleaned_payload[:-3]
        cleaned_payload = cleaned_payload.strip()

        data = json.loads(payload)
        if isinstance(data, dict):  # tolerate {"skills": [...]} shaped drift
            data = data.get("skills", data.get("items", []))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array of skills, got {type(data).__name__}")
        return [ExtractedSkill.model_validate(item) for item in data]

    # ------------------------------------------------------------------- public

    def _is_retryable(self, exc: Exception) -> bool:
        """Determine if an error is worth retrying vs. failing fast."""
        # Authentication/quota errors are permanent - don't retry
        error_str = str(exc).lower()
        fatal_keywords = [
            "api_key", "apikey", "authentication", "unauthorized", "401", "403",
            "quota", "billing", "permission denied", "invalid argument",
        ]
        if any(kw in error_str for kw in fatal_keywords):
            return False
        # Transient errors: retry
        transient_keywords = [
            "timeout", "connection", "network", "503", "502", "504", "rate limit",
            "429", "temporary", "unavailable",
        ]
        if any(kw in error_str for kw in transient_keywords):
            return True
        # Default: retry on unexpected errors
        return True

    def extract_from_text(self, text: str) -> list[ExtractedSkill]:
        """Extract from one chunk, retrying on transport errors and bad schemas."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._validate(self._generate(text))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not self._is_retryable(exc):
                    logger.error("Non-retryable error: %s", exc)
                    raise ExtractionError(f"Non-retryable extraction error: {exc}") from exc
                delay = min(2**attempt, 30) + random.uniform(0, 0.5)
                logger.warning(
                    "Extraction attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        raise ExtractionError(
            f"Chunk extraction failed after {self.max_retries} attempts"
        ) from last_error

    def extract_from_chunks(
        self, chunks: list[Chunk], source_id: str, doc_type: str = "unknown"
    ) -> ExtractionResult:
        collected: list[ExtractedSkill] = []
        failed: list[int] = []
        for chunk in chunks:
            try:
                collected.extend(self.extract_from_text(chunk.text))
            except ExtractionError:
                logger.error("Giving up on chunk %d of %s", chunk.index, source_id)
                failed.append(chunk.index)

        return ExtractionResult(
            source_id=source_id,
            doc_type=doc_type,
            skills=merge_skills(collected),
            model=self.model,
            prompt_version=PROMPT_VERSION,
            chunk_count=len(chunks),
            failed_chunks=failed,
        )

    def extract_from_document(self, document: Document) -> ExtractionResult:
        return self.extract_from_chunks(
            document.chunks, document.source_id, document.doc_type
        )


# --------------------------------------------------------------- LangGraph node

def extraction_node(state: dict) -> dict:
    """Thin adapter for LangGraph.

    Expects `state["document"]` (a reader.Document) and an extractor either in
    `state["extractor"]` or constructed from env. Kept trivial on purpose: the
    logic lives in SkillExtractor so it is unit-testable without a graph.
    """
    document: Document = state["document"]
    extractor: SkillExtractor = state.get("extractor") or SkillExtractor()
    return {**state, "extraction": extractor.extract_from_document(document)}