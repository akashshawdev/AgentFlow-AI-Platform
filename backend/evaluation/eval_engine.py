"""
AgentFlow - Evaluation Engine
Computes RAGAS metrics and SelfCheckGPT hallucination scores.

Metrics tracked:
  - faithfulness: Are claims in the answer supported by retrieved context?
  - answer_relevancy: Does the answer address the question?
  - context_precision: Are the retrieved chunks actually useful?
  - context_recall: Does retrieval cover the expected answer?
  - hallucination_score: Variance across stochastic samples (SelfCheckGPT proxy)
  - response_consistency: Similarity of answers across 3 temperature variations
"""
import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

from llama_index.llms.openai import OpenAI

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EvalSample:
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class EvalMetrics:
    query: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    hallucination_score: float  # 0=none, 1=severe
    response_consistency: float  # 0=inconsistent, 1=perfectly consistent
    overall_score: float
    details: dict = field(default_factory=dict)


class EvaluationEngine:
    """
    Runs quantitative evaluation on query-answer-context triples.

    SelfCheckGPT implementation:
      Sample N responses with temperature > 0, measure pairwise semantic
      similarity. Low similarity = high hallucination risk.
    """

    def __init__(self):
        self.llm = OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
        self.stochastic_llm = OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.8,
        )

    def evaluate_sample(self, sample: EvalSample) -> EvalMetrics:
        """Run all metrics on a single query-answer-context triple."""
        faithfulness = self._score_faithfulness(sample)
        relevancy = self._score_answer_relevancy(sample)
        precision = self._score_context_precision(sample)
        hallucination = self._score_hallucination(sample)
        consistency = self._score_response_consistency(sample)

        overall = (
            faithfulness * 0.30
            + relevancy * 0.25
            + precision * 0.20
            + (1 - hallucination) * 0.15
            + consistency * 0.10
        )

        return EvalMetrics(
            query=sample.query,
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(relevancy, 3),
            context_precision=round(precision, 3),
            hallucination_score=round(hallucination, 3),
            response_consistency=round(consistency, 3),
            overall_score=round(overall, 3),
            details={
                "weights": {
                    "faithfulness": 0.30,
                    "relevancy": 0.25,
                    "precision": 0.20,
                    "hallucination_penalty": 0.15,
                    "consistency": 0.10,
                }
            },
        )

    def evaluate_batch(self, samples: list[EvalSample]) -> list[EvalMetrics]:
        """Evaluate a batch of samples."""
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}: '{sample.query[:50]}'")
            try:
                results.append(self.evaluate_sample(sample))
            except Exception as e:
                logger.error(f"Eval failed for sample {i}: {e}")
        return results

    def aggregate(self, metrics: list[EvalMetrics]) -> dict:
        """Compute aggregate statistics across a batch."""
        if not metrics:
            return {}

        fields = [
            "faithfulness", "answer_relevancy", "context_precision",
            "hallucination_score", "response_consistency", "overall_score"
        ]
        return {
            f: {
                "mean": round(statistics.mean(getattr(m, f) for m in metrics), 3),
                "median": round(statistics.median(getattr(m, f) for m in metrics), 3),
                "stdev": round(statistics.stdev(getattr(m, f) for m in metrics) if len(metrics) > 1 else 0.0, 3),
                "min": round(min(getattr(m, f) for m in metrics), 3),
                "max": round(max(getattr(m, f) for m in metrics), 3),
            }
            for f in fields
        }

    # ── Private scoring methods ────────────────────────────────────────────

    def _score_faithfulness(self, sample: EvalSample) -> float:
        """Check if each sentence in the answer is grounded in context."""
        context = "\n\n".join(sample.contexts[:4])
        prompt = f"""Given the context and answer below, score how faithfully the answer
is grounded in the context. Score 0.0 (not grounded) to 1.0 (fully grounded).
Respond with ONLY a float.

Context: {context}

Answer: {sample.answer}

Faithfulness score:"""
        try:
            resp = self.llm.complete(prompt)
            return max(0.0, min(1.0, float(resp.text.strip())))
        except Exception:
            return 0.5

    def _score_answer_relevancy(self, sample: EvalSample) -> float:
        """Check how relevant the answer is to the query."""
        prompt = f"""Rate how relevant this answer is to the query.
Score 0.0 (irrelevant) to 1.0 (perfectly relevant). Respond with ONLY a float.

Query: {sample.query}
Answer: {sample.answer}

Relevancy score:"""
        try:
            resp = self.llm.complete(prompt)
            return max(0.0, min(1.0, float(resp.text.strip())))
        except Exception:
            return 0.5

    def _score_context_precision(self, sample: EvalSample) -> float:
        """Check what fraction of retrieved chunks actually help answer the query."""
        useful = 0
        for ctx in sample.contexts:
            prompt = f"""Does this context chunk help answer the query?
Query: {sample.query}
Chunk: {ctx[:300]}
Answer yes or no:"""
            try:
                resp = self.llm.complete(prompt)
                if "yes" in resp.text.lower():
                    useful += 1
            except Exception:
                useful += 0

        return useful / len(sample.contexts) if sample.contexts else 0.0

    def _score_hallucination(self, sample: EvalSample) -> float:
        """
        SelfCheckGPT proxy: sample N stochastic responses and measure
        semantic variance. High variance → high hallucination risk.
        """
        samples = []
        for _ in range(settings.selfcheck_num_samples):
            try:
                prompt = f"Answer this question concisely: {sample.query}"
                resp = self.stochastic_llm.complete(prompt)
                samples.append(resp.text.strip())
            except Exception:
                pass

        if len(samples) < 2:
            return 0.5

        # Measure pairwise overlap using simple word-level Jaccard similarity
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        similarities = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                similarities.append(jaccard(samples[i], samples[j]))

        mean_sim = statistics.mean(similarities)
        # Low similarity = high hallucination score
        return round(1.0 - mean_sim, 3)

    def _score_response_consistency(self, sample: EvalSample) -> float:
        """Run the same query at 3 temperatures and measure answer similarity."""
        responses = []
        for temp in [0.0, 0.3, 0.6]:
            llm = OpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                temperature=temp,
            )
            try:
                ctx = "\n".join(sample.contexts[:2])
                resp = llm.complete(
                    f"Given: {ctx}\n\nAnswer: {sample.query}"
                )
                responses.append(resp.text.strip())
            except Exception:
                pass

        if len(responses) < 2:
            return 0.5

        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append(jaccard(responses[i], responses[j]))

        return round(statistics.mean(pairs), 3) if pairs else 0.5
