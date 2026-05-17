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
