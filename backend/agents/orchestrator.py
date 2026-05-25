"""
AgentFlow - Multi-Agent Orchestrator
Coordinates the Retriever → Summarizer → Validator pipeline with
retry-with-feedback loop.

Orchestration flow:
  1. RetrieverAgent: hybrid search → RetrievalResult
  2. SummarizerAgent: structured summary → SummaryResult
  3. ValidatorAgent: grounding check → ValidationResult
     └─ If FAIL/PARTIAL and retries remain:
        → Inject validator feedback into Summarizer prompt → Retry

This design allows transparent reasoning traces useful for debugging
and for interview demonstrations.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.agents.retriever_agent import RetrieverAgent, RetrievalResult
from backend.agents.summarizer_agent import SummarizerAgent, SummaryResult
from backend.agents.validator_agent import ValidatorAgent, ValidationResult, ValidationVerdict
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class AgentFlowResponse:
    """Final response from the multi-agent pipeline."""
    query: str
    answer: str
    key_points: list[str]
    confidence: float
    grounding_score: float
    verdict: str
    sources: list[str]
    retrieval_latency_ms: float
    total_latency_ms: float
    retries: int
    pipeline_trace: list[dict] = field(default_factory=list)


class AgentOrchestrator:
    """
    Coordinates the three-agent pipeline with retry-with-feedback.

    The retry protocol solves the hardest multi-agent coordination problem:
    when the Validator disagrees with the Summarizer, we don't just re-run —
    we inject the Validator's specific feedback into the Summarizer's next
    attempt so it can correct targeted claims rather than regenerating blindly.
    """

    def __init__(self):
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.validator = ValidatorAgent()
        self.request_count = 0

    def run(
        self,
        query: str,
        doc_type_filter: str | None = None,
        top_k: int | None = None,
        prompt_version: str = "v1.0",
    ) -> AgentFlowResponse:
        """
        Execute the full Retrieve → Summarize → Validate pipeline.

        Args:
            query: User question
            doc_type_filter: Optional document type filter
            top_k: Override retrieval count
            prompt_version: Which prompt version to use in Summarizer

        Returns:
            AgentFlowResponse with answer, metrics, and trace
        """
        self.request_count += 1
        start_total = time.perf_counter()
        trace = []

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        retrieval = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
        )
        trace.append({
            "step": "retrieval",
            "chunks_retrieved": len(retrieval.nodes),
            "latency_ms": retrieval.latency_ms,
            "doc_type_filter": doc_type_filter,
        })
        logger.info(f"[Orchestrator] Retrieval done: {len(retrieval.nodes)} chunks")

        # ── Step 2+3: Summarize → Validate (with retry) ──────────────────
        self.summarizer.prompt_version = prompt_version
        retry_count = 0
        feedback_injection: str | None = None
        summary: SummaryResult | None = None
        validation: ValidationResult | None = None

        while retry_count <= settings.validator_retry_limit:
            # Build context, optionally with validator feedback
            context_override = None
            if feedback_injection and summary:
                # Prepend feedback as an instruction to the context
                injection = (
                    f"[VALIDATOR FEEDBACK - Address these issues in your response]\n"
                    f"{feedback_injection}\n\n"
                )
                context_override = [injection] + retrieval.texts

            summary = self.summarizer.summarize(
                retrieval_result=retrieval,
                context_override=context_override,
            )
            trace.append({
                "step": f"summarize_attempt_{retry_count + 1}",
                "confidence": summary.confidence,
                "key_points_count": len(summary.key_points),
                "coverage_gaps": summary.coverage_gaps,
            })

            validation = self.validator.validate(
                summary_result=summary,
                source_texts=retrieval.texts,
                retry_count=retry_count,
            )
            trace.append({
                "step": f"validate_attempt_{retry_count + 1}",
                "verdict": validation.verdict.value,
                "grounding_score": validation.grounding_score,
                "unsupported_claims": validation.unsupported_claims,
            })

            if not self.validator.should_retry(validation):
                logger.info(
                    f"[Orchestrator] Pipeline complete after {retry_count} retries. "
                    f"Verdict: {validation.verdict.value}"
                )
                break

            feedback_injection = validation.feedback
            retry_count += 1
            logger.warning(
                f"[Orchestrator] Retry {retry_count}/{settings.validator_retry_limit}: "
                f"{validation.feedback[:80]}"
            )

        total_ms = (time.perf_counter() - start_total) * 1000

        return AgentFlowResponse(
            query=query,
            answer=summary.summary,
            key_points=summary.key_points,
            confidence=summary.confidence,
            grounding_score=validation.grounding_score,
            verdict=validation.verdict.value,
            sources=summary.sources_used,
            retrieval_latency_ms=retrieval.latency_ms,
            total_latency_ms=round(total_ms, 2),
            retries=retry_count,
            pipeline_trace=trace,
        )

    def get_health(self) -> dict:
        return {
            "status": "healthy",
            "total_requests": self.request_count,
            "retriever_stats": self.retriever.get_stats(),
        }
