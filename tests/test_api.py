"""
AgentFlow - FastAPI Integration Tests
Tests the HTTP API endpoints with mocked backend dependencies.

Run: pytest tests/test_api.py -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Test client with mocked dependencies."""
    with patch("backend.api.main.configure_llama_index"), \
         patch("backend.api.main.AgentOrchestrator") as mock_orch_cls, \
         patch("backend.api.main.EvaluationEngine") as mock_eval_cls, \
         patch("backend.api.main.PromptRegistry") as mock_registry_cls:

        mock_orch = MagicMock()
        mock_orch.get_health.return_value = {"status": "healthy", "total_requests": 42}
        mock_orch_cls.return_value = mock_orch

        mock_eval = MagicMock()
        mock_eval_cls.return_value = mock_eval

        mock_registry = MagicMock()
        mock_registry.select_version.return_value = "v1.1"
        mock_registry_cls.return_value = mock_registry

        from backend.api.main import app
        import backend.api.main as api_module
        api_module.orchestrator = mock_orch
        api_module.eval_engine = mock_eval
        api_module.prompt_registry = mock_registry

        yield TestClient(app), mock_orch, mock_eval, mock_registry


class TestHealthEndpoints:

    def test_health_ok(self, client):
        tc, _, _, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_agent_health(self, client):
        tc, mock_orch, _, _ = client
        resp = tc.get("/health/agents")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestQueryEndpoint:

    def test_query_success(self, client):
        from backend.agents.orchestrator import AgentFlowResponse
        tc, mock_orch, _, _ = client

        mock_orch.run.return_value = AgentFlowResponse(
            query="How does HNSW work?",
            answer="HNSW is a graph-based ANN algorithm.",
            key_points=["Graph-based", "Approximate NN"],
            confidence=0.92,
            grounding_score=0.94,
            verdict="PASS",
            sources=["hnsw_guide.md"],
            retrieval_latency_ms=192.0,
            total_latency_ms=741.0,
            retries=0,
            pipeline_trace=[],
        )

        resp = tc.post("/query", json={"query": "How does HNSW work?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "PASS"
        assert data["confidence"] == 0.92
        assert "pipeline_trace" in data
        assert data["metrics"]["retrieval_latency_ms"] == 192.0

    def test_query_with_doc_filter(self, client):
        from backend.agents.orchestrator import AgentFlowResponse
        tc, mock_orch, _, _ = client

        mock_orch.run.return_value = AgentFlowResponse(
            query="API auth flow", answer="Use Bearer tokens.",
            key_points=[], confidence=0.88, grounding_score=0.9,
            verdict="PASS", sources=[], retrieval_latency_ms=178.0,
            total_latency_ms=612.0, retries=0, pipeline_trace=[],
        )

        resp = tc.post("/query", json={
            "query": "API auth flow",
            "doc_type_filter": "api_reference",
            "top_k": 5,
        })
        assert resp.status_code == 200
        mock_orch.run.assert_called_once_with(
            query="API auth flow",
            doc_type_filter="api_reference",
            top_k=5,
            prompt_version="v1.1",
        )

    def test_query_too_short_returns_422(self, client):
        tc, _, _, _ = client
        resp = tc.post("/query", json={"query": "Hi"})
        assert resp.status_code == 422

    def test_query_missing_field_returns_422(self, client):
        tc, _, _, _ = client
        resp = tc.post("/query", json={})
        assert resp.status_code == 422


class TestEvaluationEndpoints:

    def test_evaluate_single(self, client):
        from backend.evaluation.eval_engine import EvalMetrics
        tc, _, mock_eval, _ = client

        mock_eval.evaluate_sample.return_value = EvalMetrics(
            query="test query",
            faithfulness=0.87,
            answer_relevancy=0.91,
            context_precision=0.79,
            hallucination_score=0.08,
            response_consistency=0.84,
            overall_score=0.85,
        )

        resp = tc.post("/evaluate", json={
            "query": "test query",
            "answer": "Test answer about HNSW.",
            "contexts": ["HNSW ef=128 reduces latency.", "Pre-filtering helps."],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["faithfulness"] == 0.87
        assert data["hallucination_score"] == 0.08
        assert data["overall_score"] == 0.85

    def test_eval_dashboard(self, client):
        tc, _, _, _ = client
        resp = tc.get("/evaluate/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "trend_7d" in data
        assert "by_doc_type" in data
        assert data["summary"]["avg_faithfulness"] == 0.847

    def test_evaluate_batch(self, client):
        from backend.evaluation.eval_engine import EvalMetrics
        tc, _, mock_eval, _ = client

        sample_metric = EvalMetrics(
            query="q", faithfulness=0.8, answer_relevancy=0.85,
            context_precision=0.75, hallucination_score=0.1,
            response_consistency=0.82, overall_score=0.8,
        )
        mock_eval.evaluate_batch.return_value = [sample_metric, sample_metric]
        mock_eval.aggregate.return_value = {
            "faithfulness": {"mean": 0.8, "median": 0.8, "stdev": 0.0, "min": 0.8, "max": 0.8}
        }

        resp = tc.post("/evaluate/batch", json={
            "samples": [
                {"query": "q1", "answer": "a1", "contexts": ["c1"]},
                {"query": "q2", "answer": "a2", "contexts": ["c2"]},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_count"] == 2
        assert "aggregate" in data


class TestPromptEndpoints:

    def test_register_prompt(self, client):
        tc, _, _, mock_registry = client
        resp = tc.post("/prompts/register", json={
            "name": "summarizer",
            "version": "v2.0",
            "template": "Summarize: {query}\n{context}",
            "system_prompt": "Be precise.",
            "description": "New structured prompt",
            "traffic_split": 0.1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["version"] == "v2.0"

    def test_list_prompt_versions(self, client):
        tc, _, _, mock_registry = client
        mock_registry.list_versions.return_value = [
            {"name": "summarizer", "version": "v1.0", "traffic_split": 0.3},
            {"name": "summarizer", "version": "v1.1", "traffic_split": 0.7},
        ]
        resp = tc.get("/prompts/summarizer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "summarizer"
        assert len(data["versions"]) == 2

    def test_experiment_summary(self, client):
        tc, _, _, mock_registry = client
        mock_registry.get_experiment_summary.return_value = [
            {"version": "v1.0", "runs": 350, "avg_score": 0.791},
            {"version": "v1.1", "runs": 897, "avg_score": 0.847},
        ]
        resp = tc.get("/prompts/summarizer/experiment")
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        assert results[1]["avg_score"] > results[0]["avg_score"]


class TestStatsEndpoint:

    def test_stats_returns_config(self, client):
        tc, mock_orch, _, _ = client
        with patch("backend.api.main.get_weaviate_client"), \
             patch("backend.api.main.get_collection_stats", return_value={"total_objects": 50000}):
            resp = tc.get("/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "config" in data
            assert data["config"]["top_k"] == 8
