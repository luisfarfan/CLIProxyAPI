package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

// proximaWebSearchRequest is the exact payload proxima-intelligence sends for a
// Gemini web-search call (see src/proxima_intelligence/llm/constants.py).
const proximaWebSearchRequest = `{"model":"gemini-3-flash","messages":[{"role":"user","content":"precio de laptop en falabella"}],"tools":[{"type":"web_search"}]}`

func TestOpenAIWebSearchToolBecomesGoogleSearch(t *testing.T) {
	out := ConvertOpenAIRequestToGemini("gemini-3-flash", []byte(proximaWebSearchRequest), false)

	tools := gjson.GetBytes(out, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools is not an array: %s", out)
	}
	var found int
	for _, tool := range tools.Array() {
		if tool.Get("googleSearch").Exists() {
			found++
		}
	}
	if found != 1 {
		t.Fatalf("googleSearch tools = %d, want 1. Output: %s", found, out)
	}
}

func TestGoogleSearchPassthroughStillWorks(t *testing.T) {
	in := `{"model":"gemini-3-flash","messages":[{"role":"user","content":"hi"}],"tools":[{"google_search":{}}]}`
	out := ConvertOpenAIRequestToGemini("gemini-3-flash", []byte(in), false)

	if got := gjson.GetBytes(out, "tools.#(googleSearch)#").Array(); len(got) != 1 {
		t.Fatalf("googleSearch tools = %d, want 1. Output: %s", len(got), out)
	}
}

func TestWebSearchToolIsNotDuplicated(t *testing.T) {
	in := `{"model":"gemini-3-flash","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search"},{"type":"web_search"}]}`
	out := ConvertOpenAIRequestToGemini("gemini-3-flash", []byte(in), false)

	var found int
	for _, tool := range gjson.GetBytes(out, "tools").Array() {
		if tool.Get("googleSearch").Exists() {
			found++
		}
	}
	if found != 1 {
		t.Fatalf("googleSearch tools = %d, want 1 (deduplicated). Output: %s", found, out)
	}
}

func TestWebSearchCoexistsWithFunctionTools(t *testing.T) {
	in := `{"model":"gemini-3-flash","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}},{"type":"web_search"}]}`
	out := ConvertOpenAIRequestToGemini("gemini-3-flash", []byte(in), false)

	var googleSearch, declarations int
	for _, tool := range gjson.GetBytes(out, "tools").Array() {
		if tool.Get("googleSearch").Exists() {
			googleSearch++
		}
		declarations += len(tool.Get("functionDeclarations").Array())
	}
	if googleSearch != 1 || declarations != 1 {
		t.Fatalf("googleSearch=%d functionDeclarations=%d, want 1 and 1. Output: %s", googleSearch, declarations, out)
	}
}

func TestNoWebSearchToolLeavesToolsAlone(t *testing.T) {
	in := `{"model":"gemini-3-flash","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]}`
	out := ConvertOpenAIRequestToGemini("gemini-3-flash", []byte(in), false)

	for _, tool := range gjson.GetBytes(out, "tools").Array() {
		if tool.Get("googleSearch").Exists() {
			t.Fatalf("unexpected googleSearch tool injected. Output: %s", out)
		}
	}
}
