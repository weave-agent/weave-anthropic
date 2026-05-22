# Anthropic Provider Resilience

## Overview
Apply shared provider retry SDK support to Anthropic and configure provider HTTP behavior where supported by the Anthropic SDK. This replaces package-global runtime retry defaults with provider runtime config while preserving existing stream deduplication semantics.

## Context (from discovery)
- Files/components involved:
  - `anthropic.go`
  - `anthropic_test.go`
  - root SDK packages `sdk/providerhttp`, `sdk/providerretry`, and `sdk/retry`
- Related patterns found:
  - Anthropic has a manual retry loop and stream accumulator dedupe
  - Anthropic retry currently uses package-global `retryConfig`
  - Anthropic SDK client is created through `anthropic.NewClient(option.WithAPIKey(apiKey))`
- Dependencies identified:
  - `github.com/anthropics/anthropic-sdk-go`
  - shared provider retry SDK support

## Development Approach
- **Testing approach**: Regular
- Complete each task fully before moving to the next
- Make small, focused changes
- Every task that changes code includes new or updated tests
- All tests must pass before starting the next task
- Update this plan file when scope changes during implementation
- Maintain backward compatibility for existing provider config fields

## Testing Strategy
- Tests for provider retry config replacing package defaults
- Tests for existing dedupe behavior after retry changes
- Tests for invalid provider config failing initialization
- Run `go test ./...` after each task

## Progress Tracking
- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Keep plan in sync with implementation

## What Goes Where
- Implementation Steps contain automatable code/test/doc tasks
- Post-Completion contains manual verification and external coordination
- Checkboxes belong only in task sections

## Implementation Steps

### Task 1: Wire Anthropic provider runtime configuration
- [x] update provider initialization to resolve `providerretry.ForProvider(cfg, "anthropic")`
- [x] configure SDK HTTP behavior from `providerhttp` if the SDK exposes a clean custom client or timeout option
- [x] store retry config in provider runtime state
- [x] preserve existing API key, model, and max token behavior
- [x] write tests for provider init with custom retry config
- [x] write tests for invalid retry config failing provider init
- [x] run `go test ./...` - must pass before next task

### Task 2: Preserve retry and dedupe semantics
- [ ] replace runtime use of package-global retry defaults with provider runtime retry config
- [ ] keep existing stream accumulator behavior for retried partial streams
- [ ] apply configured jittered retry delay in Anthropic retry loop
- [ ] add debug logging for retry attempts with safe fields only
- [ ] update existing retry tests to use configured retry values
- [ ] write tests for no duplicate text/thinking after retry
- [ ] run `go test ./...` - must pass before next task

### Task 3: Verify acceptance criteria
- [ ] verify Anthropic retry uses provider config, not package-global runtime defaults
- [ ] verify invalid provider HTTP/retry config fails provider initialization
- [ ] verify retry debug logs exclude secrets, prompts, request bodies, and response bodies
- [ ] run `go test ./...`

## Technical Details
Anthropic HTTP client configuration depends on Anthropic SDK support. If the SDK cannot accept a custom HTTP client cleanly, record the limitation in this plan and still apply shared retry config.

## Post-Completion
Manual verification:
- Run Anthropic with default provider config
- Run Anthropic with provider-specific retry overrides
- Confirm retry debug logs do not contain secrets or prompts
