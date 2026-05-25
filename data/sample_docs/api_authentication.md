# AgentFlow API Reference - Authentication

## Overview

AgentFlow uses API key authentication for all endpoints. Keys are issued per workspace
and can be scoped to specific permissions (read, write, admin).

## Authentication Header

All requests must include the following header:

```
Authorization: Bearer <your_api_key>
```

## API Key Management

### Creating an API Key

```bash
POST /auth/keys
Content-Type: application/json

{
  "name": "production-service",
  "scopes": ["query:read", "eval:write"],
  "expires_at": "2025-12-31T00:00:00Z"
}
```

Response:
```json
{
  "key_id": "key_abc123",
  "secret": "agf_sk_...",
  "created_at": "2024-11-01T10:00:00Z"
}
```

**Important**: The secret is shown only once. Store it securely.

### Key Rotation

To rotate a key without downtime:
1. Create a new key with the same scopes
2. Update your service configuration to use the new key
3. Revoke the old key after verifying the new key works

```bash
DELETE /auth/keys/{key_id}
```

## Rate Limits

| Scope | Requests/minute | Burst |
|---|---|---|
| Free tier | 20 | 50 |
| Pro tier | 200 | 500 |
| Enterprise | Custom | Custom |

Rate limit headers are included in every response:
```
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 195
X-RateLimit-Reset: 1699999999
```

Error code `429 Too Many Requests` is returned when the limit is exceeded.

## OAuth2 Support

AgentFlow supports OAuth2 with the following grant types:
- `client_credentials` — for server-to-server authentication
- `authorization_code` — for user-delegated access

Token endpoint: `POST /auth/oauth/token`
