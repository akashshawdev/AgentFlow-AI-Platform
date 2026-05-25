# Webhook Configuration Runbook

## Overview
AgentFlow can emit events to your webhook endpoint for pipeline completions,
evaluation results, and ingestion job statuses.

## Setup

### Registering a Webhook

```bash
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-service.com/agentflow-events",
  "events": ["query.completed", "eval.completed", "ingest.finished"],
  "secret": "your-signing-secret"
}
```

### Signature Verification

Each webhook payload is signed with HMAC-SHA256 using your secret:

```python
import hmac
import hashlib

def verify_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

The signature is passed in the `X-AgentFlow-Signature` header.

## Retry Policy

Failed webhook deliveries are retried with exponential backoff:
- Attempt 1: immediate
- Attempt 2: 30 seconds
- Attempt 3: 5 minutes
- Attempt 4: 30 minutes
- Attempt 5: 2 hours (final)

A delivery is considered failed if the endpoint returns a non-2xx status
or does not respond within 10 seconds.

## Event Payloads

### query.completed

```json
{
  "event": "query.completed",
  "timestamp": "2024-11-01T10:00:00Z",
  "data": {
    "query_id": "q_abc123",
    "query": "How does HNSW work?",
    "verdict": "PASS",
    "grounding_score": 0.94,
    "latency_ms": 741
  }
}
```

### ingest.finished

```json
{
  "event": "ingest.finished",
  "timestamp": "2024-11-01T10:05:00Z",
  "data": {
    "job_id": "ingest_xyz",
    "documents_loaded": 1200,
    "chunks_upserted": 4800,
    "elapsed_seconds": 187.3,
    "errors": 0
  }
}
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| 401 on delivery | Wrong secret | Regenerate webhook secret |
| Timeouts | Endpoint too slow | Respond 200 immediately, process async |
| Missing events | Event type not subscribed | Update subscription events list |
