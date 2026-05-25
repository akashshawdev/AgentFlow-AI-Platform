"""
AgentFlow - Summarizer Agent
Transforms retrieved document chunks into coherent, structured summaries.

Design rationale:
  - Uses map-reduce for large context windows (>6 chunks) to avoid token limits
  - Structured output enforces a consistent JSON contract that the Validator
    agent can verify programmatically
  - System prompt is versioned via the PromptRegistry (see core/prompt_registry.py)
"""
import logging
from dataclasses import dataclass
from typing import Optional

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from backend.core.config import get_settings
from backend.agents.retriever_agent import RetrievalResult

logger = logging.getLogger(__name__)
settings = get_settings()


SUMMARIZER_SYSTEM_PROMPT = """You are a precise technical summarizer. Given retrieved document chunks,
produce a structured summary in valid JSON. Be faithful to the source material.
Do NOT add information not present in the chunks. If unsure, say so explicitly."""

SUMMARIZER_USER_TEMPLATE = """Retrieved context chunks:
{context}

User query: {query}

Respond ONLY with a JSON object matching this schema:
{{
  "summary": "<concise answer to the query based on the chunks>",
  "key_points": ["<point 1>", "<point 2>", "..."],
  "confidence": <float 0.0-1.0>,
  "sources_used": ["<source filename>", "..."],
  "coverage_gaps": "<what the retrieved context does NOT cover, or 'none'>"
}}"""


@dataclass
class SummaryResult:
    query: str
    summary: str
    key_points: list[str]
    confidence: float
    sources_used: list[str]
    coverage_gaps: str
    prompt_version: str
    raw_response: str
    input_chunks: int


class SummarizerAgent:
    """
    Generates structured summaries from retrieval results.

    Uses map-reduce strategy for large chunk sets:
      - ≤6 chunks: single-pass summarization
      - >6 chunks: summarize in groups of 3, then reduce
    """

    def __init__(self, llm: LLM | None = None, prompt_version: str = "v1.0"):
        self.llm = llm or OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.05,
        )
        self.prompt_version = prompt_version

    def summarize(
        self,
        retrieval_result: RetrievalResult,
        context_override: list[str] | None = None,
    ) -> SummaryResult:
        """
        Summarize retrieved chunks into a structured response.

        Args:
            retrieval_result: Output from RetrieverAgent
            context_override: Optional manual context (for testing)

        Returns:
            SummaryResult with structured fields
        """
        texts = context_override or retrieval_result.texts
        sources = retrieval_result.sources

        if len(texts) > 6:
            context = self._map_reduce(texts, retrieval_result.query)
        else:
            context = self._format_context(texts, sources)

        prompt = SUMMARIZER_USER_TEMPLATE.format(
            context=context,
            query=retrieval_result.query,
        )

        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        import json
        try:
            response = self.llm.chat(messages)
            raw = response.message.content.strip()
            parsed = json.loads(raw)
            return SummaryResult(
                query=retrieval_result.query,
                summary=parsed.get("summary", ""),
                key_points=parsed.get("key_points", []),
                confidence=float(parsed.get("confidence", 0.5)),
                sources_used=parsed.get("sources_used", sources[:3]),
                coverage_gaps=parsed.get("coverage_gaps", "none"),
                prompt_version=self.prompt_version,
                raw_response=raw,
                input_chunks=len(texts),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Summarizer JSON parse failed: {e}. Using fallback.")
            return SummaryResult(
                query=retrieval_result.query,
                summary=raw if "raw" in dir() else "Summarization failed.",
                key_points=[],
                confidence=0.2,
                sources_used=sources[:3],
                coverage_gaps="Parse error - raw response returned",
                prompt_version=self.prompt_version,
                raw_response=raw if "raw" in dir() else "",
                input_chunks=len(texts),
            )

    def _format_context(self, texts: list[str], sources: list[str]) -> str:
        parts = []
        for i, (text, source) in enumerate(zip(texts, sources)):
            parts.append(f"[Chunk {i+1} | Source: {source}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def _map_reduce(self, texts: list[str], query: str) -> str:
        """Map-reduce for large context: summarize groups then combine."""
        group_size = 3
        intermediate = []

        for i in range(0, len(texts), group_size):
            group = texts[i: i + group_size]
            mini_context = "\n\n".join(group)
            mini_prompt = f"Summarize these chunks relevant to: '{query}'\n\n{mini_context}"
            resp = self.llm.complete(mini_prompt)
            intermediate.append(resp.text)

        combined = "\n\n".join(
            [f"[Group {i+1}]\n{s}" for i, s in enumerate(intermediate)]
        )
        logger.debug(f"Map-reduce: {len(texts)} chunks → {len(intermediate)} groups")
        return combined
