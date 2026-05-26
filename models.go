package anthropic

import (
	"github.com/weave-agent/weave/sdk/model"
)

//nolint:gochecknoinits // Model registration is intentionally package side-effect driven.
func init() {
	RegisterModels()
}

// RegisterModels registers Anthropic model definitions in the global model registry.
func RegisterModels() {
	model.RegisterModel(model.ModelDef{
		ID: "claude-opus-4-7", Provider: providerName,
		DisplayName: "Claude Opus 4.7", Reasoning: true, SupportsXHigh: true,
		ContextWindow: 1000000, MaxTokens: 128000,
	})
	model.RegisterModel(model.ModelDef{
		ID: "claude-sonnet-4-6", Provider: providerName,
		DisplayName: "Claude Sonnet 4.6", Reasoning: true, SupportsXHigh: false,
		ContextWindow: 1000000, MaxTokens: 64000, Default: true,
	})
	model.RegisterModel(model.ModelDef{
		ID: "claude-opus-4-5", Provider: providerName,
		DisplayName: "Claude Opus 4.5", Reasoning: true, SupportsXHigh: false,
		ContextWindow: 200000, MaxTokens: 64000,
	})
	model.RegisterModel(model.ModelDef{
		ID: "claude-sonnet-4-5", Provider: providerName,
		DisplayName: "Claude Sonnet 4.5", Reasoning: true, SupportsXHigh: false,
		ContextWindow: 200000, MaxTokens: 64000,
	})
	model.RegisterModel(model.ModelDef{
		ID: "claude-haiku-4-5", Provider: providerName,
		DisplayName: "Claude Haiku 4.5", Reasoning: true, SupportsXHigh: false,
		ContextWindow: 200000, MaxTokens: 64000,
	})
}
