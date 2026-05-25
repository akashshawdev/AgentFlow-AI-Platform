"""
AgentFlow - Test Suite
Unit and integration tests for agents, evaluation engine, and prompt registry.

Run: pytest tests/ -v --cov=backend --cov-report=term-missing
"""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_retrieval_result():
    from backend.agents.retriever_agent import RetrievalResult
    mock_node = MagicMock()
    mock_node.node.text = "HNSW ef=128 reduces P95 latency by 38% on 50k document corpus."
    mock_node.node.metadata = {"source": "vector_search_tuning.md"}
    mock_node.score = 0.91

    mock_node2 = MagicMock()
    mock_node2.node.text = "Pre-filtering by doc_type reduces candidate pool from 50k to 8k."
    mock_node2.node.metadata = {"source": "vector_search_tuning.md"}
    mock_node2.score = 0.87

    return RetrievalResult(
        query="How do I reduce retrieval latency?",
        nodes=[mock_node, mock_node2],
        latency_ms=192.0,
        doc_type_filter="api_reference",
        top_k=8,
    )


@pytest.fixture
def sample_summary_result():
    from backend.agents.summarizer_agent import SummaryResult
    return SummaryResult(
        query="How do I reduce retrieval latency?",
        summary="Increase HNSW ef to 128 and apply doc_type pre-filtering.",
        key_points=[
            "Set HNSW ef=128 for better recall",
            "Apply doc_type filter before ANN scoring",
            "Pre-filtering reduces candidates from 50k to 8k",
        ],
        confidence=0.92,
        sources_used=["vector_search_tuning.md"],
        coverage_gaps="none",
        prompt_version="v1.1",
        raw_response='{"summary": "...", "key_points": [], "confidence": 0.92}',
        input_chunks=2,
    )


# ── RetrieverAgent tests ──────────────────────────────────────────────────────

class TestRetrieverAgent:

    def test_retrieval_result_texts(self, sample_retrieval_result):
        texts = sample_retrieval_result.texts
        assert len(texts) == 2
        assert "HNSW ef=128" in texts[0]

    def test_retrieval_result_scores(self, sample_retrieval_result):
        scores = sample_retrieval_result.scores
        assert all(0 <= s <= 1 for s in scores)
        assert scores[0] > scores[1]  # First result should score higher

    def test_retrieval_result_sources(self, sample_retrieval_result):
        sources = sample_retrieval_result.sources
        assert all(s == "vector_search_tuning.md" for s in sources)

    def test_cache_hit_rate_empty(self):
        from backend.agents.retriever_agent import RetrieverAgent
        with patch("backend.agents.retriever_agent.get_index"):
            agent = RetrieverAgent()
            assert agent._cache_hit_rate() == 0.0

    def test_get_stats_structure(self):
        from backend.agents.retriever_agent import RetrieverAgent
        with patch("backend.agents.retriever_agent.get_index"):
            agent = RetrieverAgent()
            stats = agent.get_stats()
            assert "total_queries" in stats
            assert "cache_hits" in stats
            assert "cache_hit_rate" in stats
            assert "cached_queries" in stats


# ── SummarizerAgent tests ─────────────────────────────────────────────────────

class TestSummarizerAgent:

    def test_summarize_returns_summary_result(self, sample_retrieval_result):
        from backend.agents.summarizer_agent import SummarizerAgent, SummaryResult

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "summary": "Use HNSW ef=128 for 38% latency reduction.",
            "key_points": ["Tune ef parameter", "Apply pre-filtering"],
            "confidence": 0.92,
            "sources_used": ["vector_search_tuning.md"],
            "coverage_gaps": "none"
        })
        mock_llm.chat.return_value = mock_response

        agent = SummarizerAgent(llm=mock_llm)
        result = agent.summarize(sample_retrieval_result)

        assert isinstance(result, SummaryResult)
        assert result.confidence == 0.92
        assert len(result.key_points) == 2
        assert result.coverage_gaps == "none"

    def test_summarize_handles_json_parse_error(self, sample_retrieval_result):
        from backend.agents.summarizer_agent import SummarizerAgent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "This is not valid JSON"
        mock_llm.chat.return_value = mock_response

        agent = SummarizerAgent(llm=mock_llm)
        result = agent.summarize(sample_retrieval_result)

        # Should fall back gracefully
        assert result.confidence == 0.2
        assert "Parse error" in result.coverage_gaps

    def test_map_reduce_called_for_large_context(self, sample_retrieval_result):
        from backend.agents.summarizer_agent import SummarizerAgent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "summary": "Large context summary.",
            "key_points": ["Point 1"],
            "confidence": 0.8,
            "sources_used": [],
            "coverage_gaps": "none"
        })
        mock_response.text = "group summary"
        mock_llm.chat.return_value = mock_response
        mock_llm.complete.return_value = mock_response

        # Override texts to have >6 items
        large_retrieval = MagicMock()
        large_retrieval.query = "test query"
        large_retrieval.texts = [f"Chunk {i}" for i in range(8)]
        large_retrieval.sources = ["doc.md"] * 8

        agent = SummarizerAgent(llm=mock_llm)
        result = agent.summarize(large_retrieval)
        assert result is not None

    def test_format_context_interleaves_sources(self):
        from backend.agents.summarizer_agent import SummarizerAgent
        mock_llm = MagicMock()
        agent = SummarizerAgent(llm=mock_llm)

        texts = ["chunk one", "chunk two"]
        sources = ["doc_a.md", "doc_b.md"]
        ctx = agent._format_context(texts, sources)

        assert "Chunk 1" in ctx
        assert "doc_a.md" in ctx
        assert "---" in ctx


# ── ValidatorAgent tests ──────────────────────────────────────────────────────

class TestValidatorAgent:

    def test_validate_pass_verdict(self, sample_summary_result):
        from backend.agents.validator_agent import ValidatorAgent, ValidationVerdict

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "verdict": "PASS",
            "unsupported_claims": [],
            "supported_claims": ["HNSW ef=128 reduces latency"],
            "grounding_score": 0.95,
            "feedback": "ok"
        })
        mock_llm.chat.return_value = mock_response

        agent = ValidatorAgent()
        agent.llm = mock_llm

        result = agent.validate(sample_summary_result, ["HNSW ef=128 reduces P95 by 38%."])

        assert result.verdict == ValidationVerdict.PASS
        assert result.grounding_score == 0.95
        assert len(result.unsupported_claims) == 0

    def test_validate_fail_verdict(self, sample_summary_result):
        from backend.agents.validator_agent import ValidatorAgent, ValidationVerdict

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "verdict": "FAIL",
            "unsupported_claims": ["Rate limits apply per workspace"],
            "supported_claims": [],
            "grounding_score": 0.3,
            "feedback": "Claim about workspace-level limits not supported"
        })
        mock_llm.chat.return_value = mock_response

        agent = ValidatorAgent()
        agent.llm = mock_llm

        result = agent.validate(sample_summary_result, ["Some unrelated context."])

        assert result.verdict == ValidationVerdict.FAIL
        assert len(result.unsupported_claims) == 1

    def test_should_retry_on_fail(self, sample_summary_result):
        from backend.agents.validator_agent import ValidatorAgent, ValidationResult, ValidationVerdict

        agent = ValidatorAgent()
        fail_result = ValidationResult(
            verdict=ValidationVerdict.FAIL,
            grounding_score=0.3,
            unsupported_claims=["unsupported claim"],
            supported_claims=[],
            feedback="Fix this",
            summary_ref="summary...",
            retry_count=0,
        )
        assert agent.should_retry(fail_result) is True

    def test_should_not_retry_when_limit_reached(self, sample_summary_result):
        from backend.agents.validator_agent import ValidatorAgent, ValidationResult, ValidationVerdict
        from backend.core.config import get_settings

        agent = ValidatorAgent()
        settings = get_settings()
        fail_result = ValidationResult(
            verdict=ValidationVerdict.FAIL,
            grounding_score=0.3,
            unsupported_claims=["unsupported claim"],
            supported_claims=[],
            feedback="Fix this",
            summary_ref="summary...",
            retry_count=settings.validator_retry_limit,
        )
        assert agent.should_retry(fail_result) is False

    def test_validator_handles_parse_error(self, sample_summary_result):
        from backend.agents.validator_agent import ValidatorAgent, ValidationVerdict

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "not valid json"
        mock_llm.chat.return_value = mock_response

        agent = ValidatorAgent()
        agent.llm = mock_llm

        result = agent.validate(sample_summary_result, ["context"])
        assert result.verdict == ValidationVerdict.FAIL
        assert result.grounding_score == 0.0


# ── EvaluationEngine tests ────────────────────────────────────────────────────

class TestEvaluationEngine:

    def test_aggregate_returns_all_metrics(self):
        from backend.evaluation.eval_engine import EvaluationEngine, EvalMetrics

        metrics = [
            EvalMetrics(
                query=f"q{i}", faithfulness=0.8+i*0.01,
                answer_relevancy=0.85, context_precision=0.75,
                hallucination_score=0.1, response_consistency=0.82,
                overall_score=0.8,
            )
            for i in range(5)
        ]
        engine = EvaluationEngine.__new__(EvaluationEngine)
        agg = engine.aggregate(metrics)

        assert "faithfulness" in agg
        assert "hallucination_score" in agg
        assert "mean" in agg["faithfulness"]
        assert "stdev" in agg["faithfulness"]
        assert agg["faithfulness"]["mean"] == pytest.approx(0.82, abs=0.01)

    def test_aggregate_empty_returns_empty(self):
        from backend.evaluation.eval_engine import EvaluationEngine
        engine = EvaluationEngine.__new__(EvaluationEngine)
        assert engine.aggregate([]) == {}

    def test_jaccard_similarity(self):
        """Test the internal Jaccard used for hallucination scoring."""
        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        assert jaccard("hello world", "hello world") == 1.0
        assert jaccard("hello world", "goodbye universe") == 0.0
        assert 0 < jaccard("the quick brown fox", "the quick red fox") < 1.0


# ── PromptRegistry tests ──────────────────────────────────────────────────────

class TestPromptRegistry:

    @pytest.fixture
    def registry(self, tmp_path):
        from backend.core.prompt_registry import PromptRegistry
        db_url = f"sqlite:///{tmp_path}/test.db"
        return PromptRegistry(db_url=db_url)

    def test_register_and_get(self, registry):
        from backend.core.prompt_registry import PromptSpec
        spec = PromptSpec(
            name="test_prompt",
            version="v1.0",
            template="Answer: {query}",
            system_prompt="You are helpful.",
            description="Test prompt",
        )
        registry.register(spec)
        retrieved = registry.get("test_prompt", "v1.0")
        assert retrieved is not None
        assert retrieved.template == "Answer: {query}"

    def test_get_nonexistent_returns_none(self, registry):
        result = registry.get("nonexistent", "v99.0")
        assert result is None

    def test_list_versions(self, registry):
        from backend.core.prompt_registry import PromptSpec
        for v in ["v1.0", "v1.1", "v2.0"]:
            registry.register(PromptSpec(
                name="multi_ver", version=v,
                template=f"Template {v}", traffic_split=0.33,
            ))
        versions = registry.list_versions("multi_ver")
        assert len(versions) == 3
        assert all(v["name"] == "multi_ver" for v in versions)

    def test_select_version_returns_valid_version(self, registry):
        from backend.core.prompt_registry import PromptSpec
        registry.register(PromptSpec("ab_test", "ctrl", "Template A", traffic_split=0.5))
        registry.register(PromptSpec("ab_test", "var", "Template B", traffic_split=0.5))

        selected = registry.select_version("ab_test")
        assert selected in ["ctrl", "var"]

    def test_log_run_and_experiment_summary(self, registry):
        from backend.core.prompt_registry import PromptSpec
        registry.register(PromptSpec("eval_prompt", "v1.0", "Template"))

        registry.log_run(
            run_id="run-001",
            prompt_name="eval_prompt",
            prompt_version="v1.0",
            query="test query",
            metrics={"faithfulness": 0.85, "hallucination_score": 0.08, "overall_score": 0.82},
        )

        summary = registry.get_experiment_summary("eval_prompt")
        assert len(summary) >= 1
        assert summary[0]["runs"] >= 1

    def test_update_existing_version(self, registry):
        from backend.core.prompt_registry import PromptSpec
        registry.register(PromptSpec("upd", "v1.0", "Old template"))
        registry.register(PromptSpec("upd", "v1.0", "New template"))

        retrieved = registry.get("upd", "v1.0")
        assert retrieved.template == "New template"


# ── Ingestion pipeline tests ──────────────────────────────────────────────────

class TestIngestionPipeline:

    def test_detect_doc_type(self):
        from backend.pipelines.ingestion import detect_doc_type
        from pathlib import Path

        assert detect_doc_type(Path("docs/guide.md")) == "tutorial"
        assert detect_doc_type(Path("api/endpoints.json")) == "api_reference"
        assert detect_doc_type(Path("ops/runbook.txt")) == "runbook"
        assert detect_doc_type(Path("spec/schema.pdf")) == "specification"
        assert detect_doc_type(Path("unknown.xyz")) == "general"

    def test_doc_hash_deterministic(self):
        from backend.pipelines.ingestion import doc_hash
        h1 = doc_hash("same text content", "source.md")
        h2 = doc_hash("same text content", "source.md")
        assert h1 == h2

    def test_doc_hash_differs_by_source(self):
        from backend.pipelines.ingestion import doc_hash
        h1 = doc_hash("same text", "source_a.md")
        h2 = doc_hash("same text", "source_b.md")
        assert h1 != h2

    def test_batched_splits_correctly(self):
        from backend.pipelines.ingestion import IngestionPipeline
        with patch.object(IngestionPipeline, '__init__', lambda s, **k: None):
            pipeline = IngestionPipeline.__new__(IngestionPipeline)
            items = list(range(10))
            batches = list(pipeline._batched(items, 3))
            assert len(batches) == 4
            assert batches[0] == [0, 1, 2]
            assert batches[-1] == [9]
