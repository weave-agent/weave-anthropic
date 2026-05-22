# CLAUDE.md

## Provider Runtime Configuration

The Anthropic provider resolves HTTP and retry behavior during provider initialization through shared Weave SDK helpers:

- `providerhttp.ForProvider(cfg, "anthropic")`
- `providerretry.ForProvider(cfg, "anthropic")`

Production Anthropic traffic must use the resolved HTTP client via `option.WithHTTPClient`. Store the resolved retry policy on provider state and use it in the stream retry loop instead of package-global retry defaults.

Retry debug logs must use safe metadata fields only, such as attempt numbers, retry delay, max retries, and error type. Do not log API keys, prompts, request bodies, or response bodies.
