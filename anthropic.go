package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/weave-agent/weave/sdk"
	"github.com/weave-agent/weave/sdk/model"
	"github.com/weave-agent/weave/sdk/providerhttp"
	"github.com/weave-agent/weave/sdk/providerretry"
	"github.com/weave-agent/weave/sdk/retry"
)

const (
	providerName     = "anthropic"
	defaultModel     = "claude-sonnet-4-6"
	defaultMaxTokens = 16384
)

// AnthropicConfig holds per-provider configuration for the Anthropic provider.
type AnthropicConfig struct {
	Model     string `json:"model" default:"claude-sonnet-4-6" env:"ANTHROPIC_MODEL" description:"Model name"`
	MaxTokens int    `json:"max_tokens" default:"16384" env:"ANTHROPIC_MAX_TOKENS" validate:"gt=0" description:"Maximum tokens"`
}

// AuthConfig holds authentication credentials for the Anthropic provider.
type AuthConfig struct {
	APIKey string `json:"api_key" env:"ANTHROPIC_API_KEY" description:"API key"`
}

type provider struct {
	client      anthropic.Client
	model       string
	maxTokens   int
	retryConfig retry.Config
}

var _ sdk.TokenCounter = (*provider)(nil)

//nolint:gochecknoinits // Provider registration is intentionally package side-effect driven.
func init() {
	sdk.RegisterProvider[AnthropicConfig, AuthConfig](providerName, newProvider)
}

var newAnthropicClient = func(apiKey string, httpClient *http.Client) anthropic.Client {
	return anthropic.NewClient(
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(0),
	)
}

func newProvider(cfg sdk.Config, ac AnthropicConfig, a AuthConfig) (sdk.Provider, error) {
	if a.APIKey == "" {
		return nil, errors.New("anthropic: API key required (set ANTHROPIC_API_KEY)")
	}

	httpClient, _, err := providerhttp.ForProvider(cfg, providerName)
	if err != nil {
		return nil, fmt.Errorf("anthropic: configure http: %w", err)
	}

	retryCfg, _, err := providerretry.ForProvider(cfg, providerName)
	if err != nil {
		return nil, fmt.Errorf("anthropic: configure retry: %w", err)
	}

	return &provider{
		client:      newAnthropicClient(a.APIKey, httpClient),
		model:       ac.Model,
		maxTokens:   ac.MaxTokens,
		retryConfig: retryCfg,
	}, nil
}

// NewProviderWithClient creates a provider with a pre-configured client (for testing).
func NewProviderWithClient(client anthropic.Client, modelName string) sdk.Provider {
	if modelName == "" {
		modelName = defaultModel
	}

	return &provider{
		client:      client,
		model:       modelName,
		maxTokens:   defaultMaxTokens,
		retryConfig: retry.DefaultConfig(),
	}
}

func (p *provider) Stream(ctx context.Context, req sdk.ProviderRequest, opts ...model.StreamOption) (<-chan sdk.ProviderEvent, error) {
	ch := make(chan sdk.ProviderEvent, 64)

	so := model.NewStreamOptions(opts...)

	modelName := so.Model
	if modelName == "" {
		modelName = p.model
	}

	maxTokens := so.MaxTokens
	if maxTokens == 0 {
		maxTokens = p.maxTokens
	}

	params := p.buildParams(req, modelName, maxTokens, so.ThinkingLevel)

	send := func(evt sdk.ProviderEvent) bool {
		select {
		case ch <- evt:
			return true
		case <-ctx.Done():
			return false
		}
	}

	go func() {
		defer close(ch)

		acc := &streamAccumulator{
			seenToolCalls: make(map[string]bool),
		}

		cfg := p.retryConfig

		var lastErr error

		success := false

		for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
			if attempt > 0 {
				delay := retry.JitteredDelay(retry.CalculateDelay(cfg, attempt-1), cfg.Jitter)
				sdk.Logger("anthropic").Debug("anthropic stream retry",
					"attempt", attempt,
					"next_attempt", attempt+1,
					"max_retries", cfg.MaxRetries,
					"delay", delay.String(),
					"error_type", fmt.Sprintf("%T", lastErr),
				)

				timer := time.NewTimer(delay)

				select {
				case <-timer.C:
				case <-ctx.Done():
					timer.Stop()
					send(sdk.ProviderEvent{
						Type:    sdk.ProviderEventError,
						Content: ctx.Err().Error(),
					})

					return
				}
			}

			stream := p.client.Messages.NewStreaming(ctx, params)

			var message anthropic.Message

			var curText, curThinking strings.Builder

			for stream.Next() {
				event := stream.Current()
				_ = message.Accumulate(event)

				e, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent)
				if !ok {
					continue
				}

				if e.Delta.Text != "" {
					curText.WriteString(e.Delta.Text)

					if !acc.emitTextIfNew(curText.String(), send) {
						return
					}
				}

				if e.Delta.Thinking != "" {
					curThinking.WriteString(e.Delta.Thinking)

					if !acc.emitThinkingIfNew(curThinking.String(), send) {
						return
					}
				}
			}

			if err := stream.Err(); err != nil {
				if !isRetriableError(err) {
					send(sdk.ProviderEvent{
						Type:    sdk.ProviderEventError,
						Content: err.Error(),
					})

					return
				}

				lastErr = err

				continue
			}

			if !acc.validateCompletedTotals(curText.String(), curThinking.String(), send) {
				return
			}

			success = true

			emitContentBlocksWithAccumulator(message.Content, acc, send)
			emitUsageEvent(message, send)

			break
		}

		if !success && lastErr != nil {
			send(sdk.ProviderEvent{
				Type:    sdk.ProviderEventError,
				Content: fmt.Sprintf("max retries exceeded (%d): %v", cfg.MaxRetries, lastErr),
			})
		}
	}()

	return ch, nil
}

func (p *provider) CountTokens(ctx context.Context, req sdk.ProviderRequest, opts ...model.StreamOption) (sdk.TokenCount, error) {
	so := model.NewStreamOptions(opts...)

	modelName := so.Model
	if modelName == "" {
		modelName = p.model
	}

	params := p.buildCountTokensParams(req, modelName, so.ThinkingLevel)

	count, err := p.countTokensWithRetry(ctx, params)
	if err != nil {
		return sdk.TokenCount{}, fmt.Errorf("anthropic: count tokens: %w", err)
	}

	return sdk.TokenCount{
		InputTokens: int(count.InputTokens),
		Source:      sdk.TokenCountSourceExact,
		Confidence:  1.0,
	}, nil
}

func (p *provider) countTokensWithRetry(ctx context.Context, params anthropic.MessageCountTokensParams) (*anthropic.MessageTokensCount, error) {
	cfg := p.retryConfig

	var lastErr error

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := retry.JitteredDelay(retry.CalculateDelay(cfg, attempt-1), cfg.Jitter)
			sdk.Logger("anthropic").Debug("anthropic count tokens retry",
				"attempt", attempt,
				"next_attempt", attempt+1,
				"max_retries", cfg.MaxRetries,
				"delay", delay.String(),
				"error_type", fmt.Sprintf("%T", lastErr),
			)

			timer := time.NewTimer(delay)

			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()

				return nil, fmt.Errorf("retry delay canceled: %w", ctx.Err())
			}
		}

		count, err := p.client.Messages.CountTokens(ctx, params)
		if err == nil {
			return count, nil
		}

		if !isRetriableError(err) {
			return nil, fmt.Errorf("request failed: %w", safeCountTokensError(err))
		}

		lastErr = err
	}

	return nil, fmt.Errorf("max retries exceeded (%d): %w", cfg.MaxRetries, safeCountTokensError(lastErr))
}

func safeCountTokensError(err error) error {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		return err
	}

	errorType := countTokensAPIErrorType(apiErr)
	if errorType == "" {
		errorType = "unknown"
	}

	requestID := apiErr.RequestID
	if requestID == "" {
		requestID = "unknown"
	}

	return fmt.Errorf("anthropic API error: status_code=%d status=%q request_id=%q error_type=%q",
		apiErr.StatusCode,
		http.StatusText(apiErr.StatusCode),
		requestID,
		errorType,
	)
}

func countTokensAPIErrorType(apiErr *anthropic.Error) string {
	var body struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}

	if err := json.Unmarshal([]byte(apiErr.RawJSON()), &body); err != nil {
		return ""
	}

	return body.Error.Type
}

// streamAccumulator tracks content emitted across retry attempts to
// deduplicate when a retried stream re-emits previously-seen content.
type streamAccumulator struct {
	text             strings.Builder
	thinking         strings.Builder
	seenToolCalls    map[string]bool
	signedThinking   []sdk.SignedThinking
	redactedThinking []sdk.RedactedThinking
}

//nolint:dupl // text and thinking deduplication follow the same pattern intentionally
func (a *streamAccumulator) emitTextIfNew(curTotal string, send func(sdk.ProviderEvent) bool) bool {
	existing := a.text.String()

	if len(curTotal) <= len(existing) {
		if existing[:len(curTotal)] == curTotal {
			return true
		}
	}

	if strings.HasPrefix(curTotal, existing) {
		toEmit := curTotal[len(existing):]
		if toEmit != "" {
			a.text.WriteString(toEmit)
			return send(sdk.ProviderEvent{Type: sdk.ProviderEventTextDelta, Content: toEmit})
		}

		return true
	}

	// Divergence — shouldn't happen for deterministic streams.
	// Emit an error and stop processing so downstream state isn't corrupted.
	send(sdk.ProviderEvent{
		Type:    sdk.ProviderEventError,
		Content: errors.New("anthropic: stream diverged after retry"),
	})

	return false
}

//nolint:dupl // text and thinking deduplication follow the same pattern intentionally
func (a *streamAccumulator) emitThinkingIfNew(curTotal string, send func(sdk.ProviderEvent) bool) bool {
	existing := a.thinking.String()

	if len(curTotal) <= len(existing) {
		if existing[:len(curTotal)] == curTotal {
			return true
		}
	}

	if strings.HasPrefix(curTotal, existing) {
		toEmit := curTotal[len(existing):]
		if toEmit != "" {
			a.thinking.WriteString(toEmit)
			return send(sdk.ProviderEvent{Type: sdk.ProviderEventThinking, Content: toEmit})
		}

		return true
	}

	// Divergence — shouldn't happen for deterministic streams.
	// Emit an error and stop processing so downstream state isn't corrupted.
	send(sdk.ProviderEvent{
		Type:    sdk.ProviderEventError,
		Content: errors.New("anthropic: stream diverged after retry"),
	})

	return false
}

func (a *streamAccumulator) validateCompletedTotals(curText, curThinking string, send func(sdk.ProviderEvent) bool) bool {
	if !completedTotalCoversEmitted(a.text.String(), curText) || !completedTotalCoversEmitted(a.thinking.String(), curThinking) {
		send(sdk.ProviderEvent{
			Type:    sdk.ProviderEventError,
			Content: "anthropic: stream diverged after retry",
		})

		return false
	}

	return true
}

func completedTotalCoversEmitted(emitted, completed string) bool {
	return len(emitted) <= len(completed) && strings.HasPrefix(completed, emitted)
}

func (a *streamAccumulator) emitThinkingDone(st sdk.SignedThinking, send func(sdk.ProviderEvent) bool) bool {
	for _, existing := range a.signedThinking {
		if existing.Signature == st.Signature {
			return true
		}
	}

	a.signedThinking = append(a.signedThinking, st)

	return send(sdk.ProviderEvent{Type: sdk.ProviderEventThinkingDone, Content: st})
}

func (a *streamAccumulator) emitRedactedThinkingDone(rt sdk.RedactedThinking, send func(sdk.ProviderEvent) bool) bool {
	for _, existing := range a.redactedThinking {
		if existing.Data == rt.Data {
			return true
		}
	}

	a.redactedThinking = append(a.redactedThinking, rt)

	return send(sdk.ProviderEvent{Type: sdk.ProviderEventRedactedThinkingDone, Content: rt})
}

func (a *streamAccumulator) emitToolCall(tc sdk.ToolCall, send func(sdk.ProviderEvent) bool) bool {
	if a.seenToolCalls[tc.ID] {
		return true
	}

	a.seenToolCalls[tc.ID] = true

	return send(sdk.ProviderEvent{Type: sdk.ProviderEventToolCall, Content: tc})
}

func emitUsageEvent(message anthropic.Message, send func(sdk.ProviderEvent) bool) {
	if message.Usage.InputTokens > 0 || message.Usage.OutputTokens > 0 {
		send(sdk.ProviderEvent{
			Type: sdk.ProviderEventUsage,
			Content: sdk.ProviderUsage{
				InputTokens:         int(message.Usage.InputTokens),
				OutputTokens:        int(message.Usage.OutputTokens),
				CacheCreationTokens: int(message.Usage.CacheCreationInputTokens),
				CacheReadTokens:     int(message.Usage.CacheReadInputTokens),
			},
		})
	}
}

func emitContentBlocksWithAccumulator(blocks []anthropic.ContentBlockUnion, acc *streamAccumulator, send func(sdk.ProviderEvent) bool) {
	for _, block := range blocks {
		switch b := block.AsAny().(type) {
		case anthropic.ThinkingBlock:
			if !acc.emitThinkingDone(sdk.SignedThinking{Signature: b.Signature, Thinking: b.Thinking}, send) {
				return
			}
		case anthropic.RedactedThinkingBlock:
			if !acc.emitRedactedThinkingDone(sdk.RedactedThinking{Data: b.Data}, send) {
				return
			}
		case anthropic.ToolUseBlock:
			args, ok := parseToolArgs(b.Name, b.JSON.Input.Raw(), send)
			if !ok {
				return
			}

			if !acc.emitToolCall(sdk.ToolCall{ID: b.ID, Name: b.Name, Arguments: args}, send) {
				return
			}
		}
	}
}

func isRetriableError(err error) bool {
	if err == nil {
		return false
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Timeout()
	}

	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return urlErr.Timeout()
	}

	msgLower := strings.ToLower(err.Error())

	if strings.Contains(msgLower, "429") || strings.Contains(msgLower, "rate limit") || strings.Contains(msgLower, "too many requests") {
		return true
	}

	if strings.Contains(msgLower, "500") || strings.Contains(msgLower, "502") || strings.Contains(msgLower, "503") || strings.Contains(msgLower, "504") {
		return true
	}

	if strings.Contains(msgLower, "timeout") || strings.Contains(msgLower, "deadline exceeded") || strings.Contains(msgLower, "eof") {
		return true
	}

	if strings.Contains(msgLower, "connection") && (strings.Contains(msgLower, "reset") || strings.Contains(msgLower, "refused") || strings.Contains(msgLower, "closed")) {
		return true
	}

	return false
}

func (p *provider) buildParams(req sdk.ProviderRequest, mdl string, maxTokens int, thinkingLevel model.ThinkingLevel) anthropic.MessageNewParams {
	params := anthropic.MessageNewParams{
		Model:     mdl,
		MaxTokens: int64(maxTokens),
		Messages:  convertMessages(req.Messages),
	}

	if req.SystemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Text:         req.SystemPrompt,
				CacheControl: anthropic.NewCacheControlEphemeralParam(),
			},
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
	}

	thinkingLevel = resolveThinkingLevel(mdl, thinkingLevel)

	if thinkingLevel != model.ThinkingOff {
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}

		effortMap := map[model.ThinkingLevel]anthropic.OutputConfigEffort{
			model.ThinkingMinimal: anthropic.OutputConfigEffortLow,
			model.ThinkingLow:     anthropic.OutputConfigEffortLow,
			model.ThinkingMedium:  anthropic.OutputConfigEffortMedium,
			model.ThinkingHigh:    anthropic.OutputConfigEffortHigh,
			model.ThinkingXHigh:   anthropic.OutputConfigEffortXhigh,
		}

		if effort, ok := effortMap[thinkingLevel]; ok {
			params.OutputConfig = anthropic.OutputConfigParam{Effort: effort}
		}
	}

	return params
}

func (p *provider) buildCountTokensParams(req sdk.ProviderRequest, mdl string, thinkingLevel model.ThinkingLevel) anthropic.MessageCountTokensParams {
	params := anthropic.MessageCountTokensParams{
		Model:    mdl,
		Messages: convertMessages(req.Messages),
	}

	if req.SystemPrompt != "" {
		params.System = anthropic.MessageCountTokensParamsSystemUnion{
			OfTextBlockArray: []anthropic.TextBlockParam{
				{
					Text:         req.SystemPrompt,
					CacheControl: anthropic.NewCacheControlEphemeralParam(),
				},
			},
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertCountTokensTools(req.Tools)
	}

	thinkingLevel = resolveThinkingLevel(mdl, thinkingLevel)

	if thinkingLevel != model.ThinkingOff {
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}

		effortMap := map[model.ThinkingLevel]anthropic.OutputConfigEffort{
			model.ThinkingMinimal: anthropic.OutputConfigEffortLow,
			model.ThinkingLow:     anthropic.OutputConfigEffortLow,
			model.ThinkingMedium:  anthropic.OutputConfigEffortMedium,
			model.ThinkingHigh:    anthropic.OutputConfigEffortHigh,
			model.ThinkingXHigh:   anthropic.OutputConfigEffortXhigh,
		}

		if effort, ok := effortMap[thinkingLevel]; ok {
			params.OutputConfig = anthropic.OutputConfigParam{Effort: effort}
		}
	}

	return params
}

func resolveThinkingLevel(mdl string, level model.ThinkingLevel) model.ThinkingLevel {
	if level == model.ThinkingOff {
		return model.ThinkingOff
	}

	m, ok := model.GetModelForProvider(mdl, "anthropic")
	if !ok {
		return level
	}

	if !m.Reasoning {
		return model.ThinkingOff
	}

	if level == model.ThinkingXHigh && !m.SupportsXHigh {
		return model.ThinkingHigh
	}

	return level
}

func parseToolArgs(toolName, raw string, send func(sdk.ProviderEvent) bool) (map[string]any, bool) {
	if raw == "" {
		return make(map[string]any), true
	}

	var args map[string]any

	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		send(sdk.ProviderEvent{
			Type:    sdk.ProviderEventError,
			Content: fmt.Sprintf("anthropic: parse tool call arguments for %s: %v", toolName, err),
		})

		return nil, false
	}

	return args, true
}

func convertMessages(msgs []sdk.Message) []anthropic.MessageParam {
	var (
		params                    []anthropic.MessageParam
		pendingToolResults        []anthropic.ContentBlockParamUnion
		lastUserParamIdx          = -1
		compactionSummaryParamIdx = -1
	)

	flush := func() {
		if len(pendingToolResults) > 0 {
			params = append(params, anthropic.NewUserMessage(pendingToolResults...))
			lastUserParamIdx = len(params) - 1
			pendingToolResults = nil
		}
	}

	for _, msg := range msgs {
		switch msg.Role {
		case sdk.RoleUser:
			flush()

			params = append(params, anthropic.NewUserMessage(
				anthropic.NewTextBlock(fmt.Sprint(msg.Content)),
			))
			lastUserParamIdx = len(params) - 1
		case sdk.RoleAssistant:
			flush()

			var blocks []anthropic.ContentBlockParamUnion

			for _, st := range msg.Thinking {
				blocks = append(blocks, anthropic.NewThinkingBlock(st.Signature, st.Thinking))
			}

			for _, rt := range msg.RedactedThinking {
				blocks = append(blocks, anthropic.NewRedactedThinkingBlock(rt.Data))
			}

			if text, ok := msg.Content.(string); ok && text != "" {
				if strings.HasPrefix(text, "[Compaction Summary]\n") {
					compactionSummaryParamIdx = len(params)
				}

				blocks = append(blocks, anthropic.NewTextBlock(text))
			}

			for _, tc := range msg.ToolCalls {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    tc.ID,
						Name:  tc.Name,
						Input: tc.Arguments,
					},
				})
			}

			if len(blocks) > 0 {
				params = append(params, anthropic.NewAssistantMessage(blocks...))
			}
		case sdk.RoleToolResult:
			content := fmt.Sprint(msg.Content)
			pendingToolResults = append(pendingToolResults,
				anthropic.NewToolResultBlock(msg.ToolCallID, content, msg.IsError))
		}
	}

	flush()

	cacheControl := anthropic.NewCacheControlEphemeralParam()

	if lastUserParamIdx >= 0 && lastUserParamIdx < len(params) {
		applyCacheControl(&params[lastUserParamIdx], cacheControl)
	}

	if compactionSummaryParamIdx >= 0 && compactionSummaryParamIdx < len(params) {
		applyCacheControl(&params[compactionSummaryParamIdx], cacheControl)
	}

	return params
}

func applyCacheControl(msg *anthropic.MessageParam, cacheControl anthropic.CacheControlEphemeralParam) {
	// Apply cache control to the LAST eligible block to maximize caching.
	for i := range slices.Backward(msg.Content) {
		switch {
		case msg.Content[i].OfText != nil:
			msg.Content[i].OfText.CacheControl = cacheControl
			return
		case msg.Content[i].OfToolResult != nil:
			msg.Content[i].OfToolResult.CacheControl = cacheControl
			return
		}
	}
}

func convertTools(tools []sdk.ToolDef) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, len(tools))

	for i, t := range tools {
		var (
			properties map[string]any
			required   []string
		)

		if params, ok := t.Parameters.(map[string]any); ok {
			if p, ok := params["properties"].(map[string]any); ok {
				properties = p
			}

			switch r := params["required"].(type) {
			case []string:
				required = r
			case []any:
				for _, v := range r {
					if s, ok := v.(string); ok {
						required = append(required, s)
					}
				}
			}
		}

		result[i] = anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: properties,
					Required:   required,
				},
			},
		}
	}

	return result
}

func convertCountTokensTools(tools []sdk.ToolDef) []anthropic.MessageCountTokensToolUnionParam {
	messageTools := convertTools(tools)
	result := make([]anthropic.MessageCountTokensToolUnionParam, len(messageTools))

	for i, tool := range messageTools {
		result[i] = anthropic.MessageCountTokensToolUnionParam{OfTool: tool.OfTool}
	}

	return result
}
