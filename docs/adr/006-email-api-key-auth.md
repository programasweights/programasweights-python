# ADR-006: Email-to-API-key auth for frictionless onboarding

## Status: Accepted (2026-03-21)

## Context

We need auth for rate limiting tiers and data attribution. GitHub OAuth exists but adds friction (redirect flow, GitHub account required). We want maximum adoption with minimum barriers.

## Decision

Add email-based API key flow:

1. User enters email (website or `paw.login("user@example.com")`)
2. Server sends 6-digit code via transactional email
3. User enters code → server returns API key
4. Key stored locally in `~/.cache/programasweights/config.json`
5. API key included as `X-API-Key` header in all requests

Keep GitHub OAuth as an alternative for higher trust tier.

## Security

- API keys stored as SHA-256 hashes in PostgreSQL (never raw)
- Verification codes stored in Redis with 10-minute TTL
- Keys can be rotated via `POST /api/v1/auth/rotate-key`
- Rate limits enforced per-key, not per-IP, for authenticated users

## Rate Limit Tiers

| Tier | Compiles/hour | Inferences/hour |
|------|--------------|-----------------|
| Anonymous (IP) | 3 | 30 |
| Email-verified | 30 | 300 |
| GitHub OAuth | 50 | 500 |

## Consequences

- Near-zero friction: just an email, no password, no OAuth redirect
- Works from SDK without a browser
- Email doubles as contact for notifications (model updates, etc.)
- Need a transactional email service (SendGrid/Mailgun/Resend)
