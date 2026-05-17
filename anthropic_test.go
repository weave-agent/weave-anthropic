package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weave-agent/weave/sdk"
	"github.com/weave-agent/weave/sdk/model"
	"github.com/weave-agent/weave/sdk/retry"
)

func TestMain(m *testing.M) {
	retryConfig = retry.Config{
		MaxRetries: 2,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   10 * time.Millisecond,
		Multiplier: 2,
	}

	os.Exit(m.Run())
}

type sseEvent struct {
	EventType string
	Data      string
}

func writeSSE(w http.ResponseWriter, events []sseEvent) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	for _, evt := range events {
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", evt.EventType, evt.Data)

		flusher.Flush()
	}
}

func textStreamEvents(text string) []sseEvent {
	return []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}
}

func toolCallEvents(toolID, toolName, inputJSON string) []sseEvent {
	return []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: fmt.Sprintf(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":%q,"name":%q,"input":{}}}`, toolID, toolName)},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":%q}}`, inputJSON)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":50}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}
}

func textAndToolEvents(text, toolID, toolName, inputJSON string) []sseEvent {
	return []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "content_block_start", Data: fmt.Sprintf(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":%q,"name":%q,"input":{}}}`, toolID, toolName)},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":%q}}`, inputJSON)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":1}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":60}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}
}

func newTestProvider(server *httptest.Server) sdk.Provider {
	client := anthropic.NewClient(
		option.WithAPIKey("test-key"),
		option.WithBaseURL(server.URL),
	)

	return NewProviderWithClient(client, "claude-sonnet-4-6")
}

func collectEvents(t *testing.T, ch <-chan sdk.ProviderEvent) []sdk.ProviderEvent {
	t.Helper()

	var events []sdk.ProviderEvent

	for {
		select {
		case evt, ok := <-ch:
			if !ok {
				return events
			}

			events = append(events, evt)
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for events")
		}
	}
}

func TestStream_TextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, textStreamEvents("Hello, world!"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Say hello"),
		},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var textDeltas []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventTextDelta {
			textDeltas = append(textDeltas, evt.Content.(string))
		}
	}

	assert.Equal(t, []string{"Hello, world!"}, textDeltas)
	assert.NoError(t, err)
}

func TestStream_ToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, toolCallEvents("toolu_123", "bash", `{"command":"ls -la"}`))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("List files"),
		},
		Tools: []sdk.ToolDef{
			{
				Name:        "bash",
				Description: "Run a bash command",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{"type": "string"},
					},
					"required": []string{"command"},
				},
			},
		},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var toolCalls []sdk.ToolCall

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventToolCall {
			toolCalls = append(toolCalls, evt.Content.(sdk.ToolCall))
		}
	}

	require.Len(t, toolCalls, 1)
	assert.Equal(t, "toolu_123", toolCalls[0].ID)
	assert.Equal(t, "bash", toolCalls[0].Name)
	assert.Equal(t, map[string]any{"command": "ls -la"}, toolCalls[0].Arguments)
}

func TestStream_TextAndToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, textAndToolEvents("I'll list the files.", "toolu_456", "bash", `{"command":"ls"}`))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("List files"),
		},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var (
		textDeltas []string
		toolCalls  []sdk.ToolCall
	)

	for _, evt := range events {
		switch evt.Type {
		case sdk.ProviderEventTextDelta:
			textDeltas = append(textDeltas, evt.Content.(string))
		case sdk.ProviderEventToolCall:
			toolCalls = append(toolCalls, evt.Content.(sdk.ToolCall))
		}
	}

	assert.Equal(t, []string{"I'll list the files."}, textDeltas)
	require.Len(t, toolCalls, 1)
	assert.Equal(t, "bash", toolCalls[0].Name)
}

func TestStream_WithSystemPrompt(t *testing.T) {
	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("response"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		SystemPrompt: "You are a helpful assistant.",
		Messages: []sdk.Message{
			sdk.NewUserMessage("Hello"),
		},
	})
	require.NoError(t, err)
	collectEvents(t, ch)

	assert.Contains(t, receivedBody, "You are a helpful assistant.")
}

func TestStream_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"invalid model"}}`)
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Hello"),
		},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var errorMsgs []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventError {
			errorMsgs = append(errorMsgs, evt.Content.(string))
		}
	}

	assert.NotEmpty(t, errorMsgs, "expected at least one error event")
}

func TestStream_EmptyMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, textStreamEvents("Hi"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)
	assert.NotEmpty(t, events)
}

func TestStream_MultipleToolCalls(t *testing.T) {
	events := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"ls\"}"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"read","input":{}}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp/test\"}"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":1}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":80}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, events)
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Run ls then read a file"),
		},
	})
	require.NoError(t, err)

	collected := collectEvents(t, ch)

	var toolCalls []sdk.ToolCall

	for _, evt := range collected {
		if evt.Type == sdk.ProviderEventToolCall {
			toolCalls = append(toolCalls, evt.Content.(sdk.ToolCall))
		}
	}

	require.Len(t, toolCalls, 2)
	assert.Equal(t, "toolu_1", toolCalls[0].ID)
	assert.Equal(t, "bash", toolCalls[0].Name)
	assert.Equal(t, "toolu_2", toolCalls[1].ID)
	assert.Equal(t, "read", toolCalls[1].Name)
}

func TestConvertMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []sdk.Message
		wantLen  int
	}{
		{
			name:     "empty",
			messages: []sdk.Message{},
			wantLen:  0,
		},
		{
			name: "user message",
			messages: []sdk.Message{
				sdk.NewUserMessage("Hello"),
			},
			wantLen: 1,
		},
		{
			name: "user and assistant",
			messages: []sdk.Message{
				sdk.NewUserMessage("Hello"),
				sdk.NewAssistantMessage("Hi there"),
			},
			wantLen: 2,
		},
		{
			name: "tool result groups into single user message",
			messages: []sdk.Message{
				sdk.NewUserMessage("List files"),
				{Role: sdk.RoleAssistant, ToolCalls: []sdk.ToolCall{
					{ID: "toolu_1", Name: "bash", Arguments: map[string]any{"command": "ls"}},
					{ID: "toolu_2", Name: "read", Arguments: map[string]any{"path": "/tmp"}},
				}},
				sdk.NewToolResultMessage("toolu_1", "bash", "file1.txt\nfile2.txt", false),
				sdk.NewToolResultMessage("toolu_2", "read", "content", false),
			},
			wantLen: 3, // user + assistant + grouped tool results
		},
		{
			name: "tool result with error",
			messages: []sdk.Message{
				sdk.NewToolResultMessage("toolu_err", "bash", "command not found", true),
			},
			wantLen: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := convertMessages(tt.messages)
			assert.Len(t, params, tt.wantLen)
		})
	}
}

func TestConvertTools(t *testing.T) {
	tools := []sdk.ToolDef{
		{
			Name:        "bash",
			Description: "Run a command",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "The command",
					},
				},
				"required": []string{"command"},
			},
		},
	}

	result := convertTools(tools)
	assert.Len(t, result, 1)
	assert.NotNil(t, result[0].OfTool)
	assert.Equal(t, "bash", result[0].OfTool.Name)
}

func TestConvertTools_NilParameters(t *testing.T) {
	tools := []sdk.ToolDef{
		{
			Name:        "bash",
			Description: "Run a command",
		},
	}

	result := convertTools(tools)
	assert.Len(t, result, 1)
	assert.NotNil(t, result[0].OfTool)
}

func TestStream_SplitToolInputJSON(t *testing.T) {
	events := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_split","name":"bash","input":{}}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"com"}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"mand\":\"ls\"}"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, events)
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Run ls"),
		},
	})
	require.NoError(t, err)

	collected := collectEvents(t, ch)

	var toolCalls []sdk.ToolCall

	for _, evt := range collected {
		if evt.Type == sdk.ProviderEventToolCall {
			toolCalls = append(toolCalls, evt.Content.(sdk.ToolCall))
		}
	}

	require.Len(t, toolCalls, 1)
	assert.Equal(t, "bash", toolCalls[0].Name)
	assert.Equal(t, map[string]any{"command": "ls"}, toolCalls[0].Arguments)
}

func TestStream_RetryOn429(t *testing.T) {
	attemptCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}`)

			return
		}

		writeSSE(w, textStreamEvents("Hello after retry!"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Say hello")},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var textDeltas []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventTextDelta {
			textDeltas = append(textDeltas, evt.Content.(string))
		}
	}

	assert.Equal(t, []string{"Hello after retry!"}, textDeltas)
	assert.Equal(t, 3, attemptCount, "expected 3 attempts (2 failures + 1 success)")
}

func TestRegister(t *testing.T) {
	assert.True(t, sdk.ProviderRegistered("anthropic"))
}

func TestStream_WithThinkingLevel(t *testing.T) {
	model.ResetModelRegistry()
	defer model.ResetModelRegistry()

	RegisterModels()

	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("thinking response"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("think")},
	}, model.WithThinkingLevel(model.ThinkingHigh))
	require.NoError(t, err)
	collectEvents(t, ch)

	assert.Contains(t, receivedBody, `"thinking"`)
	assert.Contains(t, receivedBody, `"adaptive"`)
	assert.Contains(t, receivedBody, `"output_config"`)
	assert.Contains(t, receivedBody, `"effort"`)
	assert.Contains(t, receivedBody, `"high"`)
}

func TestStream_ThinkingOff_NoThinkingParam(t *testing.T) {
	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("no thinking"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("hello")},
	}, model.WithThinkingLevel(model.ThinkingOff))
	require.NoError(t, err)
	collectEvents(t, ch)

	assert.NotContains(t, receivedBody, `"thinking"`)
}

func TestStream_WithModelOverride(t *testing.T) {
	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("response"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("hello")},
	}, model.WithModel("claude-opus-4-7"))
	require.NoError(t, err)
	collectEvents(t, ch)

	assert.Contains(t, receivedBody, "claude-opus-4-7")
}

func TestStream_ThinkingContentEmitted(t *testing.T) {
	events := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":1}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, events)
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("think")},
	}, model.WithThinkingLevel(model.ThinkingMedium))
	require.NoError(t, err)

	evts := collectEvents(t, ch)

	var (
		thinkingDeltas []string
		textDeltas     []string
	)

	for _, evt := range evts {
		switch evt.Type {
		case sdk.ProviderEventThinking:
			thinkingDeltas = append(thinkingDeltas, evt.Content.(string))
		case sdk.ProviderEventTextDelta:
			textDeltas = append(textDeltas, evt.Content.(string))
		}
	}

	assert.Equal(t, []string{"let me think"}, thinkingDeltas)
	assert.Equal(t, []string{"answer"}, textDeltas)
}

func TestStream_UsageEventEmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, textStreamEvents("Hello, world!"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Say hello"),
		},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var usageEvents []sdk.ProviderUsage

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventUsage {
			usageEvents = append(usageEvents, evt.Content.(sdk.ProviderUsage))
		}
	}

	require.Len(t, usageEvents, 1)
	assert.Equal(t, 10, usageEvents[0].InputTokens)
	assert.Equal(t, 5, usageEvents[0].OutputTokens)
	assert.Equal(t, 0, usageEvents[0].CacheCreationTokens)
	assert.Equal(t, 0, usageEvents[0].CacheReadTokens)
}

func TestStream_UsageEventWithCacheTokens(t *testing.T) {
	events := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":1,"cache_creation_input_tokens":50,"cache_read_input_tokens":200}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cached response"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":10}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, events)
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{
			sdk.NewUserMessage("Say hello"),
		},
	})
	require.NoError(t, err)

	collected := collectEvents(t, ch)

	var usageEvents []sdk.ProviderUsage

	for _, evt := range collected {
		if evt.Type == sdk.ProviderEventUsage {
			usageEvents = append(usageEvents, evt.Content.(sdk.ProviderUsage))
		}
	}

	require.Len(t, usageEvents, 1)
	assert.Equal(t, 100, usageEvents[0].InputTokens)
	assert.Equal(t, 10, usageEvents[0].OutputTokens)
	assert.Equal(t, 50, usageEvents[0].CacheCreationTokens)
	assert.Equal(t, 200, usageEvents[0].CacheReadTokens)
}

func TestAnthropic_CacheControlMarkers(t *testing.T) {
	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("response"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		SystemPrompt: "You are a helpful assistant.",
		Messages: []sdk.Message{
			sdk.NewUserMessage("Hello"),
			sdk.NewAssistantMessage("Hi there"),
			sdk.NewUserMessage("What's the weather?"),
		},
	})
	require.NoError(t, err)
	collectEvents(t, ch)

	var body map[string]any

	err = json.Unmarshal([]byte(receivedBody), &body)
	require.NoError(t, err)

	// System prompt should have cache_control
	system, ok := body["system"].([]any)
	require.True(t, ok)
	require.Len(t, system, 1)

	sysBlock := system[0].(map[string]any)
	cacheControl, ok := sysBlock["cache_control"].(map[string]any)
	require.True(t, ok, "system prompt should have cache_control")
	assert.Equal(t, "ephemeral", cacheControl["type"])

	// Last user message should have cache_control
	messages, ok := body["messages"].([]any)
	require.True(t, ok)

	var lastUserMsg map[string]any

	for _, m := range messages {
		msg := m.(map[string]any)
		if msg["role"] == "user" {
			lastUserMsg = msg
		}
	}

	require.NotNil(t, lastUserMsg)

	content := lastUserMsg["content"].([]any)
	require.GreaterOrEqual(t, len(content), 1)

	firstBlock := content[0].(map[string]any)
	cacheControl, ok = firstBlock["cache_control"].(map[string]any)
	require.True(t, ok, "last user message should have cache_control")
	assert.Equal(t, "ephemeral", cacheControl["type"])
}

func TestAnthropic_CacheControlOnCompactionSummary(t *testing.T) {
	var receivedBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		receivedBody = string(buf)

		writeSSE(w, textStreamEvents("response"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		SystemPrompt: "You are a helpful assistant.",
		Messages: []sdk.Message{
			sdk.NewUserMessage("Hello"),
			sdk.NewAssistantMessage("[Compaction Summary]\nThis is a summary."),
			sdk.NewUserMessage("What's next?"),
		},
	})
	require.NoError(t, err)
	collectEvents(t, ch)

	var body map[string]any

	err = json.Unmarshal([]byte(receivedBody), &body)
	require.NoError(t, err)

	messages, ok := body["messages"].([]any)
	require.True(t, ok)

	// Find compaction summary message
	var compactionMsg map[string]any

	for _, m := range messages {
		msg := m.(map[string]any)
		if msg["role"] != "assistant" {
			continue
		}

		contentBlocks, isArr := msg["content"].([]any)
		if !isArr {
			continue
		}

		for _, c := range contentBlocks {
			block, isMap := c.(map[string]any)
			if !isMap {
				continue
			}

			text, isStr := block["text"].(string)
			if isStr && strings.HasPrefix(text, "[Compaction Summary]") {
				compactionMsg = msg

				break
			}
		}
	}

	require.NotNil(t, compactionMsg, "compaction summary message not found")

	content := compactionMsg["content"].([]any)
	firstBlock := content[0].(map[string]any)
	cacheControl, ok := firstBlock["cache_control"].(map[string]any)
	require.True(t, ok, "compaction summary should have cache_control")
	assert.Equal(t, "ephemeral", cacheControl["type"])
}
