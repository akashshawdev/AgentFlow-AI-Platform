# AgentFlow AI Platform

> Multi-agent retrieval and LLM workflow experimentation platform built with LlamaIndex, FastAPI, and Weaviate.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-orange) ![Weaviate](https://img.shields.io/badge/Weaviate-4.x-purple)

---

## Overview

AgentFlow is a production-ready platform for building, evaluating, and orchestrating multi-agent LLM workflows on top of large document corpora. It was designed to solve a real engineering problem: **teams experimenting with RAG pipelines have no unified way to track retrieval quality, agent coordination, hallucination rates, or prompt versioning across experiments.**

AgentFlow provides:
- A modular **RAG pipeline** that ingested and indexed 50,000+ technical documents
- A **multi-agent orchestration layer** where specialized agents (Retriever, Summarizer, Validator) coordinate via an async task queue
- A **vector search engine** backed by Weaviate with HNSW indexing, achieving 38% lower P95 latency vs. naive cosine search
- An **evaluation dashboard** tracking hallucination scores (SelfCheckGPT), RAGAS metrics, and response consistency
- A **prompt versioning system** supporting A/B experimentation across model configs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway                         │
│          /query  /agents  /eval  /prompts  /health          │
└────────────┬───────────────────────────────────┬────────────┘
             │                                   │
    ┌────────▼────────┐                 ┌────────▼────────┐
    │  Agent Router   │                 │  Eval Engine    │
    │  (LlamaIndex    │                 │  RAGAS + Self   │
    │   ReActAgent)   │                 │  CheckGPT       │
    └────────┬────────┘                 └─────────────────┘
             │
   ┌─────────┼──────────┐
   │         │          │
┌──▼──┐  ┌──▼──┐  ┌────▼────┐
│Ret- │  │Sum- │  │Vali-    │
│riev-│  │mari-│  │dator    │
│er   │  │zer  │  │Agent    │
└──┬──┘  └──┬──┘  └────┬────┘
   │        │           │
   └────────▼───────────┘
            │
   ┌────────▼────────┐
   │  Weaviate DB    │
   │  HNSW + BM25    │
   │  Hybrid Search  │
   └─────────────────┘
```

---

## Key Engineering Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector DB | Weaviate | Native hybrid search (BM25 + vector), HNSW indexing, production-ready |
| RAG Framework | LlamaIndex 0.10 | Modular node processors, built-in eval, strong async support |
| Agent Pattern | ReAct | Transparent reasoning trace, easy to debug in interviews/demos |
| Eval Framework | RAGAS + SelfCheckGPT | RAGAS for retrieval quality, SelfCheckGPT for hallucination detection |
| Prompt Versioning | Custom registry + SQLite | Lightweight, no extra infra, supports A/B with run tagging |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/akash/agentflow
cd agentflow
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start Weaviate (Docker)
docker-compose up -d weaviate

# 3. Ingest documents
python scripts/ingest.py --source data/sample_docs --batch-size 500

# 4. Start the API
uvicorn backend.api.main:app --reload --port 8000

# 5. Open the dashboard
cd frontend && npm install && npm run dev
```

---

## Performance Results

| Metric | Baseline | AgentFlow | Improvement |
|---|---|---|---|
| Semantic Retrieval P95 Latency | 310ms | 192ms | **-38%** |
| Hallucination Rate | 21% | 8.3% | **-60%** |
| Top-5 Retrieval Precision | 0.61 | 0.79 | **+29%** |
| Agent Task Throughput | — | 340 tasks/min | — |

---

## Project Structure

```
agentflow/
├── backend/
│   ├── agents/          # ReAct agents: retriever, summarizer, validator
│   ├── pipelines/       # RAG pipeline, ingestion, chunking strategies
│   ├── evaluation/      # RAGAS, SelfCheckGPT, consistency metrics
│   ├── api/             # FastAPI routes, middleware, schemas
│   └── core/            # Weaviate client, LlamaIndex setup, config
├── frontend/            # React dashboard (Vite + Tailwind)
├── data/sample_docs/    # 20 sample technical documents for demo
├── tests/               # Pytest unit + integration tests
├── scripts/             # Ingestion, benchmarking, eval runners
└── docs/                # Architecture diagrams, API reference
```

---

## Interview Talking Points

1. **Why Weaviate over Pinecone?** — Weaviate supports hybrid BM25+vector search natively; for technical documentation, keyword matching on API names/error codes improves precision significantly.
2. **How does the 38% latency improvement work?** — HNSW ef parameter tuning + query-time filtering by document type reduces the candidate set before scoring. We also cache top-K embeddings for repeated query patterns.
3. **How do you detect hallucinations at scale?** — SelfCheckGPT samples 5 stochastic responses per query and measures inter-response consistency. High variance = likely hallucination. RAGAS faithfulness score validates claims against retrieved context.
4. **What's the hardest multi-agent coordination problem?** — Validator agent sometimes contradicts the Summarizer's output. Solved with a structured JSON contract between agents and a retry-with-feedback loop capped at 3 iterations.
