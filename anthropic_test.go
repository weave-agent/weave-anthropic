package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
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

func fastRetryConfig() retry.Config {
	return retry.Config{
		MaxRetries: 2,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   10 * time.Millisecond,
		Multiplier: 2,
		Jitter:     retry.JitterNone,
	}
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
		option.WithMaxRetries(0),
	)

	p := NewProviderWithClient(client, "claude-sonnet-4-6").(*provider)
	p.retryConfig = fastRetryConfig()

	return p
}

type providerConfigStub struct {
	providers map[string]map[string]any
	sdk.NoopConfig
}

func (s *providerConfigStub) ExtensionConfig(scope, name string, target any) error {
	if scope != "providers" {
		return fmt.Errorf("unknown scope %q", scope)
	}

	section, ok := s.providers[name]
	if !ok {
		return nil
	}

	data, err := json.Marshal(section)
	if err != nil {
		return fmt.Errorf("marshal stub config: %w", err)
	}

	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("unmarshal stub config: %w", err)
	}

	return nil
}

type failOnConfigLookup struct {
	sdk.NoopConfig
	t *testing.T
}

func (s failOnConfigLookup) ExtensionConfig(scope, name string, target any) error {
	s.t.Helper()
	s.t.Fatalf("ExtensionConfig(%q, %q) should not be called", scope, name)

	return nil
}

func TestNewProvider_MissingAPIKey(t *testing.T) {
	got, err := newProvider(failOnConfigLookup{t: t}, AnthropicConfig{
		Model:     defaultModel,
		MaxTokens: defaultMaxTokens,
	}, AuthConfig{})
	require.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "API key required")
}

func TestNewProvider_CustomRetryConfig(t *testing.T) {
	cfg := &providerConfigStub{
		providers: map[string]map[string]any{
			"anthropic": {
				"retry": map[string]any{
					"max_retries": 3,
					"base_delay":  "250ms",
					"max_delay":   "2s",
					"multiplier":  1.5,
					"jitter":      "none",
				},
			},
		},
	}

	got, err := newProvider(cfg, AnthropicConfig{
		Model:     "claude-opus-4-1",
		MaxTokens: 2048,
	}, AuthConfig{APIKey: "test-key"})
	require.NoError(t, err)

	p, ok := got.(*provider)
	require.True(t, ok)

	assert.Equal(t, "claude-opus-4-1", p.model)
	assert.Equal(t, 2048, p.maxTokens)
	assert.Equal(t, 3, p.retryConfig.MaxRetries)
	assert.Equal(t, 250*time.Millisecond, p.retryConfig.BaseDelay)
	assert.Equal(t, 2*time.Second, p.retryConfig.MaxDelay)
	assert.InDelta(t, 1.5, p.retryConfig.Multiplier, 0.0001)
	assert.Equal(t, retry.JitterNone, p.retryConfig.Jitter)
}

func TestNewProvider_AppliesCustomHTTPConfig(t *testing.T) {
	cfg := &providerConfigStub{
		providers: map[string]map[string]any{
			"anthropic": {
				"http": map[string]any{
					"tls_handshake_timeout":   "123ms",
					"response_header_timeout": "456ms",
					"idle_conn_timeout":       "789ms",
				},
			},
		},
	}

	var capturedAPIKey string
	var capturedHTTPClient *http.Client

	oldFactory := newAnthropicClient
	newAnthropicClient = func(apiKey string, httpClient *http.Client) anthropic.Client {
		capturedAPIKey = apiKey
		capturedHTTPClient = httpClient

		return anthropic.Client{}
	}
	defer func() {
		newAnthropicClient = oldFactory
	}()

	got, err := newProvider(cfg, AnthropicConfig{
		Model:     defaultModel,
		MaxTokens: defaultMaxTokens,
	}, AuthConfig{APIKey: "test-key"})
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "test-key", capturedAPIKey)
	require.NotNil(t, capturedHTTPClient)

	transport, ok := capturedHTTPClient.Transport.(*http.Transport)
	require.True(t, ok)
	assert.Equal(t, 123*time.Millisecond, transport.TLSHandshakeTimeout)
	assert.Equal(t, 456*time.Millisecond, transport.ResponseHeaderTimeout)
	assert.Equal(t, 789*time.Millisecond, transport.IdleConnTimeout)
}

func TestNewProvider_InvalidRetryConfig(t *testing.T) {
	cfg := &providerConfigStub{
		providers: map[string]map[string]any{
			"anthropic": {
				"retry": map[string]any{
					"base_delay": "not-a-duration",
				},
			},
		},
	}

	got, err := newProvider(cfg, AnthropicConfig{
		Model:     defaultModel,
		MaxTokens: defaultMaxTokens,
	}, AuthConfig{APIKey: "test-key"})
	require.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "provider anthropic")
	assert.Contains(t, err.Error(), "invalid base_delay")
}

func TestNewProvider_InvalidHTTPConfig(t *testing.T) {
	cfg := &providerConfigStub{
		providers: map[string]map[string]any{
			"anthropic": {
				"http": map[string]any{
					"dial_timeout": "not-a-duration",
				},
			},
		},
	}

	got, err := newProvider(cfg, AnthropicConfig{
		Model:     defaultModel,
		MaxTokens: defaultMaxTokens,
	}, AuthConfig{APIKey: "test-key"})
	require.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "provider anthropic")
	assert.Contains(t, err.Error(), "invalid dial_timeout")
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

func TestStream_ConfiguredZeroRetryDisablesSDKRetries(t *testing.T) {
	attemptCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}`)
	}))
	defer server.Close()

	p := newTestProvider(server).(*provider)
	p.retryConfig = retry.Config{
		MaxRetries: 0,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   1 * time.Millisecond,
		Multiplier: 1,
		Jitter:     retry.JitterNone,
	}

	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Say hello")},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var errorMsgs []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventError {
			errorMsgs = append(errorMsgs, evt.Content.(string))
		}
	}

	require.NotEmpty(t, errorMsgs)
	assert.Contains(t, errorMsgs[len(errorMsgs)-1], "max retries exceeded (0)")
	assert.Equal(t, 1, attemptCount)
}

func TestStream_UsesConfiguredRetryConfig(t *testing.T) {
	attemptCount := 0
	partialEvents := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		writeSSEAndClose(w, partialEvents)
	}))
	defer server.Close()

	p := newTestProvider(server).(*provider)
	p.retryConfig = retry.Config{
		MaxRetries: 0,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   1 * time.Millisecond,
		Multiplier: 1,
		Jitter:     retry.JitterNone,
	}

	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Say hello")},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var errorMsgs []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventError {
			errorMsgs = append(errorMsgs, evt.Content.(string))
		}
	}

	require.NotEmpty(t, errorMsgs)
	assert.Contains(t, errorMsgs[len(errorMsgs)-1], "max retries exceeded (0)")
	assert.Equal(t, 1, attemptCount)
}

func TestStream_UsesConfiguredRetryDelay(t *testing.T) {
	var logs bytes.Buffer

	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(oldLogger)

	attemptCount := 0
	partialEvents := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount <= 2 {
			writeSSEAndClose(w, partialEvents)
			return
		}

		writeSSE(w, textStreamEvents("ok"))
	}))
	defer server.Close()

	p := newTestProvider(server).(*provider)
	p.retryConfig = retry.Config{
		MaxRetries: 2,
		BaseDelay:  7 * time.Millisecond,
		MaxDelay:   10 * time.Millisecond,
		Multiplier: 2,
		Jitter:     retry.JitterNone,
	}

	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Say hello")},
	})
	require.NoError(t, err)
	collectEvents(t, ch)

	got := logs.String()
	assert.Contains(t, got, `"delay":"7ms"`)
	assert.Contains(t, got, `"delay":"10ms"`)
	assert.Equal(t, 3, attemptCount)
}

func thinkingAndTextStreamEvents(thinking, text string) []sseEvent {
	return []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":%q}}`, thinking)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: fmt.Sprintf(`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":%q}}`, text)},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":1}`},
		{EventType: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`},
		{EventType: "message_stop", Data: `{"type":"message_stop"}`},
	}
}

func writeSSEAndClose(w http.ResponseWriter, events []sseEvent) {
	flusher := w.(http.Flusher)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	for _, evt := range events {
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", evt.EventType, evt.Data)
		flusher.Flush()
	}

	conn, _, err := w.(http.Hijacker).Hijack()
	if err == nil {
		_ = conn.Close()
	}
}

func TestStream_DeduplicatesTextAndThinkingAfterRetry(t *testing.T) {
	attemptCount := 0

	firstAttemptEvents := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}`},
		{EventType: "content_block_stop", Data: `{"type":"content_block_stop","index":0}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hel"}}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount == 1 {
			writeSSEAndClose(w, firstAttemptEvents)
			return
		}

		writeSSE(w, thinkingAndTextStreamEvents("plan next", "Hello"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Think then answer")},
	}, model.WithThinkingLevel(model.ThinkingMedium))
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var thinkingDeltas, textDeltas, errorMsgs []string

	for _, evt := range events {
		switch evt.Type {
		case sdk.ProviderEventThinking:
			thinkingDeltas = append(thinkingDeltas, evt.Content.(string))
		case sdk.ProviderEventTextDelta:
			textDeltas = append(textDeltas, evt.Content.(string))
		case sdk.ProviderEventError:
			errorMsgs = append(errorMsgs, evt.Content.(string))
		}
	}

	assert.Empty(t, errorMsgs)
	assert.Equal(t, []string{"plan", " next"}, thinkingDeltas)
	assert.Equal(t, []string{"Hel", "lo"}, textDeltas)
	assert.Equal(t, 2, attemptCount)
}

func TestStream_ErrorsWhenSuccessfulRetryIsShorterThanEmittedText(t *testing.T) {
	attemptCount := 0

	firstAttemptEvents := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount == 1 {
			writeSSEAndClose(w, firstAttemptEvents)
			return
		}

		writeSSE(w, textStreamEvents("Hello"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		Messages: []sdk.Message{sdk.NewUserMessage("Say hello")},
	})
	require.NoError(t, err)

	events := collectEvents(t, ch)

	var errorMsgs []string

	for _, evt := range events {
		if evt.Type == sdk.ProviderEventError {
			errorMsgs = append(errorMsgs, evt.Content.(string))
		}
	}

	require.NotEmpty(t, errorMsgs)
	assert.Contains(t, errorMsgs[len(errorMsgs)-1], "stream diverged after retry")
	assert.Equal(t, 2, attemptCount)
}

func TestStream_RetryDebugLogUsesSafeFields(t *testing.T) {
	var logs bytes.Buffer

	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(oldLogger)

	attemptCount := 0
	partialEvents := []sseEvent{
		{EventType: "message_start", Data: `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`},
		{EventType: "content_block_start", Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{EventType: "content_block_delta", Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"secret response body"}}`},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount == 1 {
			writeSSEAndClose(w, partialEvents)
			return
		}

		writeSSE(w, textStreamEvents("ok"))
	}))
	defer server.Close()

	p := newTestProvider(server)
	ch, err := p.Stream(context.Background(), sdk.ProviderRequest{
		SystemPrompt: "secret system prompt",
		Messages:     []sdk.Message{sdk.NewUserMessage("secret user prompt")},
	})
	require.NoError(t, err)
	collectEvents(t, ch)

	got := logs.String()
	assert.Contains(t, got, "anthropic stream retry")
	assert.Contains(t, got, `"attempt":1`)
	assert.Contains(t, got, `"max_retries":2`)
	assert.NotContains(t, got, "test-key")
	assert.NotContains(t, got, "secret system prompt")
	assert.NotContains(t, got, "secret user prompt")
	assert.NotContains(t, got, "secret response body")
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
