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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spec` | string | Yes | Natural-language specification. |
| `compiler` | string | No | Compiler identifier (default: `paw-4b-qwen3-0.6b`). |
| `slug` | string | No | URL-safe handle (e.g. `message-classifier`). Creates a `username/slug` alias. Requires auth. |
| `public` | boolean | No | List on public hub (default: true). |
| `name` | string | No | Display title. Auto-generated if omitted. |
| `tags` | string[] | No | Tags for discovery. |

**Response:**
- **202 Accepted** on success — JSON with `job_id`, `status`, `program_id`, `slug` (if created), `pseudo_program`, and `timings`
- **400** — unknown compiler
- **422** — spec too short, too long, or token limit exceeded
- **500** — compilation failed (server error)

### `POST /infer`

Run inference for a compiled program (server-side execution).

**Request body (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `program_id` | string | Program identifier — hash ID or slug (e.g. `da03/message-classifier`). |
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

Program detail. Accepts hash ID or slug. Private programs return 404 to non-owners.

**Response** includes: `id`, `name`, `spec`, `interpreter`, `public`, `aliases`, `user_slug`, `author`, `tags`, `upvotes`, `downloads`, `created_at`.

### `PATCH /programs/{id_or_slug}`

Update program metadata. **Authentication required.**

Owner-only fields: `public`, `name`, `tags`. Any authenticated user can set their own `slug`.

| Field | Type | Description |
|-------|------|-------------|
| `public` | boolean | Visibility (owner only). |
| `name` | string | Display title, max 100 chars (owner only). |
| `tags` | string[] | Tags, max 5 items (owner only). |
| `slug` | string | URL-safe handle, 2-50 chars, `^[a-z0-9][a-z0-9-]*[a-z0-9]$`. Creates/replaces your alias for this program. |

### `GET /programs/{id_or_slug}/download`

Downloads the `.paw` artifact. Returns one of:
- **302** redirect to HuggingFace CDN (program already uploaded)
- **200** file from server (freshly compiled, not yet on HF)
- **202 Accepted** with `Retry-After` header (program assets still generating, retry after the indicated seconds)
- **404** if program not found or private and not owned by the requester

### `GET /programs/resolve/{slug}`

Resolve a slug (e.g. `da03/my-classifier` or bare `email-triage`) to a canonical `program_id`.

### `POST /programs/{id}/alias`

Create or update your alias for a program. **Authentication required.** If you already have an alias for this program, it is replaced (atomic update).

| Field | Type | Description |
|-------|------|-------------|
| `slug` | string | Slug name, 2-50 chars, lowercase alphanumeric and hyphens. |

### `POST /programs/{id_or_slug}/vote`

Submit a vote. **Authentication required.**

| Field | Type | Description |
|-------|------|-------------|
| `vote` | integer | `1` (upvote) or `-1` (downvote). Same value again removes the vote. |

### `GET /programs/{id}/cases` and `POST /programs/{id}/cases`

Community case submission for a program.

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

Hosted compile quotas:

- Anonymous: **20 compiles/hour**, **1 concurrent compile**
- Authenticated: **60 compiles/hour**, **2 concurrent compiles**

Hosted server-side endpoints may also enforce additional operational concurrency or safety limits. Clients should back off when receiving `429 Too Many Requests` and respect `Retry-After` when present.

## Related

- [Python SDK Reference](python-sdk.md)
- [CLI Reference](cli.md)
