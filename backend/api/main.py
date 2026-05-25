"""
AgentFlow - FastAPI Application
Main entry point with all routes for query, agents, evaluation, and prompts.
"""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.llama_setup import configure_llama_index
from backend.core.prompt_registry import PromptRegistry
from backend.agents.orchestrator import AgentOrchestrator
from backend.evaluation.eval_engine import EvaluationEngine, EvalSample

logger = logging.getLogger(__name__)
settings = get_settings()

# Singletons (initialised at startup)
orchestrator: AgentOrchestrator | None = None
eval_engine: EvaluationEngine | None = None
prompt_registry: PromptRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    global orchestrator, eval_engine, prompt_registry
    logger.info("AgentFlow starting up...")
    configure_llama_index()
    orchestrator = AgentOrchestrator()
    eval_engine = EvaluationEngine()
    prompt_registry = PromptRegistry()
    logger.info("AgentFlow ready.")
    yield
    logger.info("AgentFlow shutting down.")


app = FastAPI(
    title="AgentFlow AI Platform",
    description="Multi-agent retrieval and LLM workflow experimentation platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    doc_type_filter: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    prompt_version: Optional[str] = Field(default=None)


class EvalRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


class BatchEvalRequest(BaseModel):
    samples: list[EvalRequest]


class PromptRegisterRequest(BaseModel):
    name: str
    version: str
    template: str
    system_prompt: str = ""
    description: str = ""
    traffic_split: float = Field(default=1.0, ge=0.0, le=1.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "environment": settings.environment}


@app.get("/health/agents")
async def agent_health():
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialised")
    return orchestrator.get_health()


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint. Runs the full Retrieve → Summarize → Validate pipeline.
    Returns structured answer with full pipeline trace.
    """
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")

    # Use A/B-selected prompt version if not specified
    version = req.prompt_version
    if not version and prompt_registry:
        version = prompt_registry.select_version("summarizer")

    try:
        result = orchestrator.run(
            query=req.query,
            doc_type_filter=req.doc_type_filter,
            top_k=req.top_k,
            prompt_version=version or "v1.0",
        )
        return {
            "query": result.query,
            "answer": result.answer,
            "key_points": result.key_points,
            "confidence": result.confidence,
            "grounding_score": result.grounding_score,
            "verdict": result.verdict,
            "sources": result.sources,
            "prompt_version_used": version,
            "metrics": {
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "retries": result.retries,
            },
            "pipeline_trace": result.pipeline_trace,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline error: {str(e)}")


@app.post("/evaluate")
async def evaluate_single(req: EvalRequest):
    """Evaluate a single query-answer-context triple."""
    if not eval_engine:
        raise HTTPException(503, "Eval engine not ready")
    sample = EvalSample(
        query=req.query,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    metrics = eval_engine.evaluate_sample(sample)
    return {
        "query": metrics.query,
        "faithfulness": metrics.faithfulness,
        "answer_relevancy": metrics.answer_relevancy,
        "context_precision": metrics.context_precision,
        "hallucination_score": metrics.hallucination_score,
        "response_consistency": metrics.response_consistency,
        "overall_score": metrics.overall_score,
    }


@app.post("/evaluate/batch")
async def evaluate_batch(req: BatchEvalRequest):
    """Evaluate a batch of samples and return aggregate stats."""
    if not eval_engine:
        raise HTTPException(503, "Eval engine not ready")
    samples = [
        EvalSample(
            query=s.query, answer=s.answer,
            contexts=s.contexts, ground_truth=s.ground_truth
        )
        for s in req.samples
    ]
    results = eval_engine.evaluate_batch(samples)
    aggregate = eval_engine.aggregate(results)
    return {
        "sample_count": len(results),
        "aggregate": aggregate,
        "per_sample": [
            {
                "query": m.query[:80],
                "faithfulness": m.faithfulness,
                "hallucination_score": m.hallucination_score,
                "overall_score": m.overall_score,
            }
            for m in results
        ],
    }


@app.get("/evaluate/dashboard")
async def eval_dashboard():
    """Returns mock dashboard metrics for demonstration."""
    return {
        "summary": {
            "total_queries_evaluated": 1247,
            "avg_faithfulness": 0.847,
            "avg_hallucination_rate": 0.083,
            "avg_context_precision": 0.791,
            "avg_overall_score": 0.812,
            "retrieval_p95_latency_ms": 192,
        },
        "trend_7d": [
            {"date": "2024-11-01", "faithfulness": 0.81, "hallucination": 0.11},
            {"date": "2024-11-02", "faithfulness": 0.83, "hallucination": 0.09},
            {"date": "2024-11-03", "faithfulness": 0.84, "hallucination": 0.09},
            {"date": "2024-11-04", "faithfulness": 0.85, "hallucination": 0.08},
            {"date": "2024-11-05", "faithfulness": 0.86, "hallucination": 0.08},
            {"date": "2024-11-06", "faithfulness": 0.85, "hallucination": 0.08},
            {"date": "2024-11-07", "faithfulness": 0.85, "hallucination": 0.083},
        ],
        "by_doc_type": {
            "api_reference": {"faithfulness": 0.91, "precision": 0.88},
            "tutorial": {"faithfulness": 0.84, "precision": 0.79},
            "runbook": {"faithfulness": 0.82, "precision": 0.76},
            "specification": {"faithfulness": 0.87, "precision": 0.83},
        },
    }


@app.post("/prompts/register")
async def register_prompt(req: PromptRegisterRequest):
    """Register a new prompt version."""
    if not prompt_registry:
        raise HTTPException(503, "Prompt registry not ready")
    from backend.core.prompt_registry import PromptSpec
    spec = PromptSpec(
        name=req.name, version=req.version, template=req.template,
        system_prompt=req.system_prompt, description=req.description,
        traffic_split=req.traffic_split,
    )
    prompt_registry.register(spec)
    return {"status": "registered", "name": req.name, "version": req.version}


@app.get("/prompts/{name}")
async def list_prompt_versions(name: str):
    if not prompt_registry:
        raise HTTPException(503, "Prompt registry not ready")
    return {"name": name, "versions": prompt_registry.list_versions(name)}


@app.get("/prompts/{name}/experiment")
async def experiment_summary(name: str):
    if not prompt_registry:
        raise HTTPException(503, "Prompt registry not ready")
    return {"name": name, "results": prompt_registry.get_experiment_summary(name)}


@app.get("/stats")
async def platform_stats():
    """High-level platform statistics."""
    from backend.core.weaviate_client import get_weaviate_client, get_collection_stats
    try:
        client = get_weaviate_client()
        weaviate_stats = get_collection_stats(client)
    except Exception as e:
        weaviate_stats = {"error": str(e)}

    return {
        "platform": "AgentFlow v1.0",
        "weaviate": weaviate_stats,
        "agents": orchestrator.get_health() if orchestrator else {},
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "top_k": settings.top_k_retrieval,
            "hybrid_alpha": settings.hybrid_alpha,
            "hnsw_ef": settings.hnsw_ef,
        },
    }
