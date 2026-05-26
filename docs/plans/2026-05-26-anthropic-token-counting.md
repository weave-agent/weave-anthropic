# Add Anthropic Exact Token Counting

## Overview
- Implement exact preflight token counting for the Anthropic provider using Anthropic's messages count-tokens API.
- Ensure counts use the same converted system prompt, messages, tools, model, and thinking settings as streaming requests.
- Enable the agent to make safer compaction decisions for Claude models.

## Context (from discovery)
- Files/components involved:
  - `anthropic.go`
  - `anthropic_test.go`
  - `models.go`
- Related patterns found:
  - `buildParams` already creates `anthropic.MessageNewParams` for streaming.
  - `convertMessages` and `convertTools` centralize provider-specific request conversion.
  - Usage events already include input/output and cache creation/read tokens.
  - System prompt uses Anthropic cache control.
- Dependencies identified:
  - Requires SDK optional `TokenCounter` contract from the root repo.
  - Anthropic SDK exposes a messages count-tokens API with request shape similar to message creation.

## Development Approach
- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next.
- Make small, focused changes.
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task.
- **CRITICAL: all tests must pass before starting next task** - no exceptions.
- **CRITICAL: update this plan file when scope changes during implementation**.
- Run tests after each change.
- Maintain backward compatibility.

## Testing Strategy
- **Unit tests**: required for every task.
- **E2E tests**: not expected for provider token counting.

## Progress Tracking
- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with ➕ prefix.
- Document issues/blockers with ⚠️ prefix.
- Update plan if implementation deviates from original scope.
- Keep plan in sync with actual work done.

## What Goes Where
- **Implementation Steps** (`[ ]` checkboxes): tasks achievable within this codebase - code changes, tests, documentation updates.
- **Post-Completion** (no checkboxes): items requiring external action - manual testing, changes in consuming projects, deployment configs, third-party verifications.
- **Checkbox placement**: Checkboxes belong only in Task sections.

## Implementation Steps

### Task 1: Refactor request parameter construction for reuse
- [x] extract shared message/tool/system/thinking parameter construction if needed for both stream and count paths
- [x] ensure streaming behavior and cache control are unchanged
- [x] preserve model override behavior from `model.StreamOption`
- [x] write tests proving stream request params remain equivalent after refactor
- [x] write tests for thinking level clamping in reused path
- [x] run `go test ./...` - must pass before next task

### Task 2: Implement `sdk.TokenCounter` for Anthropic provider
- [x] add `CountTokens(ctx, req, opts...)` method on provider
- [x] call Anthropic count-tokens API using converted system prompt, messages, tools, model, and thinking settings
- [x] return `sdk.TokenCount` with exact source and high confidence
- [x] wrap provider/API errors with `anthropic:` context
- [x] write tests for successful count including system prompt and tools
- [x] write tests for API error propagation
- [x] run `go test ./...` - must pass before next task

### Task 3: Verify thinking and cache behavior
- [x] confirm count requests include thinking configuration when enabled
- [x] confirm count requests are not treated as prompt-cache writes in telemetry
- [x] preserve existing cache control in normal streaming requests
- [x] write tests for count with thinking enabled and disabled
- [x] write tests confirming normal usage events still emit cache creation/read tokens
- [x] run `go test ./...` - must pass before next task

### Task 4: Verify acceptance criteria
- [ ] verify provider satisfies `sdk.TokenCounter`
- [ ] verify stream API behavior is unchanged
- [ ] run full provider tests with `go test ./...`
- [ ] run `golangci-lint run` or repo lint command
- [ ] verify no credentials or request bodies are logged in count errors

### Task 5: Update documentation
- [ ] update README or docs to mention exact token counting support if provider capabilities are documented

## Technical Details
- Count-token calls should use the same model resolution as streaming.
- Count-token calls should not emit provider usage events; they are preflight accounting, not generation usage.
- Anthropic's count-tokens result is still documented as an estimate; mark source as provider exact/high-confidence but keep downstream code tolerant of small drift.

## Post-Completion

**Manual verification**:
- Run a Claude session near the context limit and confirm agent receives exact preflight counts.

**External system updates**:
- Agent repo should consume this through the optional SDK interface.
