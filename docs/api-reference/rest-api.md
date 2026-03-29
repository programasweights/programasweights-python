# REST API Reference

The ProgramAsWeights HTTP API is versioned under a single base path. All paths below are relative to that base unless stated otherwise.

## Base URL

```
https://programasweights.com/api/v1
```

Use HTTPS in production. The SDK and CLI default to this host unless overridden.

## Authentication

- **Session cookie:** Set by the GitHub OAuth callback after `GET /auth/github` completes the browser flow.
- **API key:** Send `X-API-Key` with a valid key for programmatic access.

Some routes require authentication; unauthenticated requests receive an appropriate HTTP error.

## Endpoints

### `POST /compile`

Compile a specification.

**Request body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `spec` | string | Natural-language specification. |
| `compiler` | string | Compiler identifier. |

**Response (JSON):** includes `job_id`, `status`, `program_id`, `pseudo_program`, and `timings` as applicable to the job state.

### `POST /infer`

Run inference for a compiled program (server-side execution when offered by the deployment).

**Request body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `program_id` | string | Program identifier. |
| `input` | string | Input text. |
| `max_tokens` | integer | Maximum tokens to generate. |
| `temperature` | number | Sampling temperature. |

**Response (JSON):** includes `output`, `tokens_generated`, and `latency_ms`.

### `GET /programs`

List or search programs.

**Query parameters:**

| Parameter | Description |
|-----------|-------------|
| `q` | Search query. |
| `sort` | Sort order. |
| `tag` | Filter by tag. |
| `page` | Page index. |
| `per_page` | Page size. |

**Response (JSON):** `programs` (array), `total`, `page`, `per_page`.

### `GET /programs/{id_or_slug}`

Program detail including aliases, specification, and pseudo-program where exposed by policy.

### `GET /programs/{id}/download`

Redirects to the Hugging Face CDN for the `.paw` artifact download.

### `GET /programs/resolve/{slug}`

Resolve a slug or alias to a canonical `program_id`.

### `POST /programs/{id}/alias`

Create an alias for the program. **Authentication required.**

**Request body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `slug` | string | Alias slug to register. |

### `POST /programs/{id}/vote`

Submit a vote for a program. **Authentication required.**

**Request body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `vote` | integer | `1` (up) or `-1` (down). |

### `GET /programs/{id}/cases` and `POST /programs/{id}/cases`

Community feedback and case submission for a program. Method-specific payloads follow the server schema for each verb.

### `GET /auth/github`

Start GitHub OAuth; redirects the client through the provider flow.

### `GET /auth/me`

Return the current authenticated user, or an error if not authenticated.

### `GET /models/compilers`

List available compiler models and identifiers for use with compile requests.

### `GET /health`

Liveness or readiness style health check for the API service.

## Errors

Error responses use a consistent JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `error` | string | Short error code or type. |
| `message` | string | Human-readable message. |
| `details` | optional | Additional structured detail. |
| `request_id` | string | Correlation id for support and logs. |

## Rate limiting

Successful responses may include:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests per window. |
| `X-RateLimit-Remaining` | Remaining requests in the current window. |

Clients should backoff when receiving `429 Too Many Requests` and respect `Retry-After` when present.

## Related

- [Python SDK Reference](python-sdk.md)
- [CLI Reference](cli.md)
