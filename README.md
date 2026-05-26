# weave-anthropic

Anthropic provider extension for [weave](https://github.com/weave-agent/weave) — an event-driven coding agent framework.

## Fork & Customize

1. Fork this repo
2. Edit the extension implementation
3. Install your fork: `weave install github.com/<you>/weave-anthropic --name anthropic`

The `--name anthropic` ensures your fork shadows the official extension.

## Install

```bash
weave install github.com/weave-agent/weave-anthropic --name anthropic
```

## Configuration

The Anthropic provider reads `ANTHROPIC_API_KEY` for auth and supports optional `ANTHROPIC_MODEL` and `ANTHROPIC_MAX_TOKENS` overrides.

Anthropic also supports shared Weave provider HTTP and retry settings. Defaults can be configured under `providers.defaults`; Anthropic-specific overrides go under `providers.anthropic`.

```json
{
  "providers": {
    "anthropic": {
      "model": "claude-sonnet-4-6",
      "max_tokens": 16384,
      "http": {
        "dial_timeout": "10s",
        "tls_handshake_timeout": "10s",
        "response_header_timeout": "60s",
        "idle_conn_timeout": "90s"
      },
      "retry": {
        "max_retries": 5,
        "base_delay": "1s",
        "max_delay": "30s",
        "multiplier": 2,
        "jitter": "full"
      }
    }
  }
}
```

Duration values use Go duration strings such as `250ms`, `2s`, or `1m`. Retry jitter accepts `full` or `none`.

## Capabilities

The Anthropic provider supports exact preflight input token counting through Anthropic's messages count-tokens API. Counts use the same converted system prompt, messages, tools, model selection, and thinking settings as streaming requests, allowing Weave to make compaction decisions before generation starts.

Preflight counting calls `/v1/messages/count_tokens` before generation and returns an error if Anthropic rejects the count request; it does not stream or emit usage events. Anthropic documents count-token responses as estimates, so downstream code should tolerate small differences from final streamed usage even though this provider reports the preflight count as provider-exact.

## Development

```bash
git clone git@github.com:weave-agent/weave-anthropic.git
cd weave-anthropic

# Add temporary replace for local SDK (don't commit this)
echo 'replace github.com/weave-agent/weave => /path/to/local/weave' >> go.mod

go test ./...
```

## License

Same as the main weave project.
