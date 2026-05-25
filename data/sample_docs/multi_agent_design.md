# Multi-Agent Coordination Design

## Agent Roles

### RetrieverAgent
Responsible for hybrid semantic + keyword search against the Weaviate vector store.
Uses HNSW-tuned index with optional doc_type pre-filtering.

**Input**: Natural language query string
**Output**: List of NodeWithScore (text chunks with relevance scores)

### SummarizerAgent
Transforms retrieved chunks into a structured JSON summary.
Uses map-reduce for large context windows to avoid token limits.

**Input**: RetrievalResult + optional validator feedback
**Output**: SummaryResult (summary, key_points, confidence, sources)

### ValidatorAgent
Cross-checks each claim in the summary against source context.
Uses temperature=0.0 for deterministic validation.

**Input**: SummaryResult + source texts
**Output**: ValidationResult (PASS/PARTIAL/FAIL + feedback)

## Retry-with-Feedback Protocol

```
Summarizer → Output → Validator
                          ↓
                     PASS? ──→ Return to user
                          ↓
                     FAIL/PARTIAL?
                          ↓
              Inject Validator.feedback into Summarizer prompt
                          ↓
                     Retry (max 3 attempts)
```

### Why structured JSON between agents?

Free-text handoffs between agents create cascading parse failures.
The structured contract ensures:
1. Validator can programmatically check each `key_point` against context
2. Orchestrator can inject targeted feedback rather than full regeneration
3. Downstream consumers (API, dashboard) get consistent schema

## Throughput

The three-agent pipeline processes 340 queries/minute at P95 under:
- 8 retrieved chunks per query
- Average 1.2 validation retries
- gpt-4o-mini as the LLM for all agents

## Failure Modes

| Failure | Detection | Recovery |
|---|---|---|
| Retriever returns 0 chunks | Empty node list | Return "no results" response |
| Summarizer JSON parse error | json.JSONDecodeError | Return raw text with low confidence |
| Validator parse error | Exception catch | Return PARTIAL verdict, no retry |
| Max retries exceeded | retry_count >= limit | Return PARTIAL result with caveat |
