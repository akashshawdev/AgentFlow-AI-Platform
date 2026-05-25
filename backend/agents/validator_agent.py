"""
AgentFlow - Validator Agent
Verifies the Summarizer's output for factual grounding and consistency.

Key problem this solves:
  The Summarizer can produce confident-sounding text that is not supported
  by retrieved chunks. The Validator cross-checks each claim in the summary
  against the source context and returns a structured verdict.

Retry protocol:
  If the verdict is FAIL, the orchestrator calls the Summarizer again with
  the validator's feedback injected into the prompt. Max 3 retries.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from llama_index.llms.openai import OpenAI

from backend.core.config import get_settings
from backend.agents.summarizer_agent import SummaryResult

logger = logging.getLogger(__name__)
settings = get_settings()


class ValidationVerdict(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


VALIDATOR_SYSTEM_PROMPT = """You are a strict factual grounding validator for an AI system.
Your job: given a summary and its source context, determine whether the summary's claims
are supported by the source. Be conservative — mark FAIL if any key claim is unsupported."""

VALIDATOR_USER_TEMPLATE = """Source context (ground truth):
{context}

Summary to validate:
{summary}

Key points to verify:
{key_points}

Respond ONLY with a JSON object:
{{
  "verdict": "<PASS|PARTIAL|FAIL>",
  "unsupported_claims": ["<claim>", ...],
  "supported_claims": ["<claim>", ...],
  "grounding_score": <float 0.0-1.0>,
  "feedback": "<concise feedback for re-generation if FAIL/PARTIAL, else 'ok'>"
}}"""


@dataclass
class ValidationResult:
    verdict: ValidationVerdict
    grounding_score: float
    unsupported_claims: list[str]
    supported_claims: list[str]
    feedback: str
    summary_ref: str  # first 100 chars of summary for traceability
    retry_count: int = 0


class ValidatorAgent:
    """
    Validates SummaryResult against source context.

    The structured JSON contract between Summarizer and Validator is the
    critical design decision. Without it, the Validator would need to parse
    free-text — which is brittle and introduces its own hallucination risk.
    """

    def __init__(self):
        self.llm = OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0,  # deterministic validation
        )

    def validate(
        self,
        summary_result: SummaryResult,
        source_texts: list[str],
        retry_count: int = 0,
    ) -> ValidationResult:
        """
        Validate a SummaryResult against its source chunks.

        Args:
            summary_result: Output from SummarizerAgent
            source_texts: The original retrieved text chunks
            retry_count: How many retries have been attempted

        Returns:
            ValidationResult with verdict and feedback
        """
        context = "\n\n---\n\n".join(source_texts[:6])  # cap context window
        key_points_str = "\n".join(f"- {p}" for p in summary_result.key_points)

        prompt = VALIDATOR_USER_TEMPLATE.format(
            context=context,
            summary=summary_result.summary,
            key_points=key_points_str,
        )

        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        import json
        try:
            response = self.llm.chat(messages)
            raw = response.message.content.strip()
            parsed = json.loads(raw)

            verdict = ValidationVerdict(parsed.get("verdict", "FAIL"))
            result = ValidationResult(
                verdict=verdict,
                grounding_score=float(parsed.get("grounding_score", 0.0)),
                unsupported_claims=parsed.get("unsupported_claims", []),
                supported_claims=parsed.get("supported_claims", []),
                feedback=parsed.get("feedback", ""),
                summary_ref=summary_result.summary[:100],
                retry_count=retry_count,
            )

            logger.info(
                f"Validation: {verdict.value} | "
                f"grounding={result.grounding_score:.2f} | "
                f"unsupported_claims={len(result.unsupported_claims)}"
            )
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Validator parse error: {e}")
            return ValidationResult(
                verdict=ValidationVerdict.FAIL,
                grounding_score=0.0,
                unsupported_claims=["Validation parse error"],
                supported_claims=[],
                feedback=f"Internal validator error: {str(e)}",
                summary_ref=summary_result.summary[:100],
                retry_count=retry_count,
            )

    def should_retry(self, result: ValidationResult) -> bool:
        """Returns True if the pipeline should retry with validator feedback."""
        return (
            result.verdict in (ValidationVerdict.FAIL, ValidationVerdict.PARTIAL)
            and result.retry_count < settings.validator_retry_limit
        )
