# AgentFlow Vector Search Configuration

## HNSW Index Parameters

AgentFlow uses Weaviate's HNSW (Hierarchical Navigable Small World) algorithm
for approximate nearest neighbor search. The following parameters are tunable
per collection.

### Key Parameters

| Parameter | Default | AgentFlow Setting | Impact |
|---|---|---|---|
| `ef` | 64 | 128 | Query recall vs. latency tradeoff |
| `ef_construction` | 128 | 256 | Index build quality (one-time cost) |
| `max_connections` | 32 | 64 | Graph density, affects memory |

### Why ef=128?

The `ef` parameter controls how many candidates are explored during the query
phase of HNSW graph traversal. Increasing from 64 to 128:

- Increases recall from ~0.93 to ~0.97 at top-8 retrieval
- Adds ~15ms to raw ANN search time
- Net effect: fewer irrelevant results require fewer LLM verification calls,
  reducing end-to-end latency by 38% despite the higher ef cost

Benchmarked on 50,000 technical documents with 500 evaluation queries.

## Hybrid Search

AgentFlow uses Weaviate's native hybrid search combining:
- **BM25** (keyword): Precision for exact API names, error codes, version strings
- **Vector** (semantic): Recall for paraphrased queries and conceptual search

The `alpha` parameter controls the blend:
- `alpha=0.0`: BM25 only (keyword-dominant)
- `alpha=1.0`: Vector only (semantic-dominant)
- `alpha=0.75`: AgentFlow default — 75% semantic, 25% keyword

## Pre-filtering Strategy

Filtering by `doc_type` BEFORE vector scoring is the primary latency optimization.
Weaviate applies filters at the HNSW graph traversal level, not post-hoc.

```python
retriever = VectorIndexRetriever(
    index=index,
    filters=MetadataFilters(filters=[
        MetadataFilter(key="doc_type", value="api_reference")
    ]),
    vector_store_query_mode="hybrid",
    alpha=0.75,
)
```

This reduces the effective candidate pool from ~50,000 to ~8,000 for
`api_reference` queries, explaining most of the latency improvement.
