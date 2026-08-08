package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

// proximaClaudeWebSearchRequest is the exact payload proxima-intelligence sends
// for a Claude web-search call (see src/proxima_intelligence/llm/constants.py).
const proximaClaudeWebSearchRequest = `{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"precio de laptop en falabella"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`

func TestClaudeWebSearchToolSurvivesTranslation(t *testing.T) {
	out := ConvertOpenAIRequestToClaude("claude-sonnet-4-5", []byte(proximaClaudeWebSearchRequest), false)

	tools := gjson.GetBytes(out, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1. Output: %s", len(tools), out)
	}
	if got := tools[0].Get("type").String(); got != "web_search_20250305" {
		t.Fatalf("tools.0.type = %q, want web_search_20250305. Output: %s", got, out)
	}
	if got := tools[0].Get("name").String(); got != "web_search" {
		t.Fatalf("tools.0.name = %q, want web_search. Output: %s", got, out)
	}
}

func TestClaudeWebSearchRequestsBeta(t *testing.T) {
	out := ConvertOpenAIRequestToClaude("claude-sonnet-4-5", []byte(proximaClaudeWebSearchRequest), false)

	betas := gjson.GetBytes(out, "betas").Array()
	var found bool
	for _, beta := range betas {
		if beta.String() == "web-search-2025-03-05" {
			found = true
		}
	}
	if !found {
		t.Fatalf("betas = %v, want it to contain web-search-2025-03-05. Output: %s", betas, out)
	}
}

func TestClaudeWebSearchCoexistsWithFunctionTools(t *testing.T) {
	in := `{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}},{"type":"web_search_20250305","name":"web_search"}]}`
	out := ConvertOpenAIRequestToClaude("claude-sonnet-4-5", []byte(in), false)

	tools := gjson.GetBytes(out, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2. Output: %s", len(tools), out)
	}
	var serverTool, functionTool bool
	for _, tool := range tools {
		if tool.Get("type").String() == "web_search_20250305" {
			serverTool = true
		}
		if tool.Get("name").String() == "lookup" && tool.Get("input_schema").Exists() {
			functionTool = true
		}
	}
	if !serverTool {
		t.Fatalf("server tool should be preserved. Output: %s", out)
	}
	if !functionTool {
		t.Fatalf("function tool should still translate. Output: %s", out)
	}
}

func TestClaudeNoBetaWithoutWebSearch(t *testing.T) {
	in := `{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]}`
	out := ConvertOpenAIRequestToClaude("claude-sonnet-4-5", []byte(in), false)

	for _, beta := range gjson.GetBytes(out, "betas").Array() {
		if beta.String() == "web-search-2025-03-05" {
			t.Fatalf("web-search beta requested without a web search tool. Output: %s", out)
		}
	}
}
