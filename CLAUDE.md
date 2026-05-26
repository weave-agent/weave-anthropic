# CLAUDE.md

## Provider Runtime Configuration

The Anthropic provider resolves HTTP and retry behavior during provider initialization through shared Weave SDK helpers:

- `providerhttp.ForProvider(cfg, "anthropic")`
- `providerretry.ForProvider(cfg, "anthropic")`

Production Anthropic traffic must use the resolved HTTP client via `option.WithHTTPClient`. Store the resolved retry policy on provider state and use it in the stream retry loop instead of package-global retry defaults.

Retry debug logs must use safe metadata fields only, such as attempt numbers, retry delay, max retries, and error type. Do not log API keys, prompts, request bodies, or response bodies.

## Token Counting

The provider implements `sdk.TokenCounter` with Anthropic's messages count-tokens endpoint. Keep `CountTokens` request construction aligned with streaming for system prompt, converted messages, tools, model override, thinking level resolution, and cache-control markers.

Count requests must use `/v1/messages/count_tokens` only and must not include stream or max-token generation settings. They use the provider retry policy, but must not emit provider usage events.

Token-count errors must be wrapped with provider context but must not include API keys, prompts, tool names, request bodies, or response bodies.
