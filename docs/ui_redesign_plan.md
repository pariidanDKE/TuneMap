# UI Redesign Plan — Apple Music Knowledge Graph

## Overview

A comprehensive redesign of `app.py` to expose the full capability surface of the app:
Agent mode, LLM provider choice, and the lyrics pipeline (moods, topics, places, language, vocabulary metrics).

---

## Sidebar Sections

### 1. Mode *(existing)*
Radio: `Standard` | `Agent`

- **Standard** — single query → text answer + neovis.js graph
- **Agent** — LLM runs multiple `music_kg_query` tool calls autonomously, then synthesises a comprehensive answer; optionally calls `render_graph`

### 2. LLM Provider *(new)*
Radio: `vLLM (local)` | `OpenAI` | `Claude`

| Provider | Default model | Notes |
|---|---|---|
| vLLM | reads `VLLM_MODEL` env var | no API key needed; `enable_thinking=False` passed per-request |
| OpenAI | `gpt-4o-mini` | dropdown: gpt-4o-mini, gpt-4o, o3-mini |
| Claude | `claude-haiku-4-5-20251001` | dropdown: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 |

UI elements:
- Model selectbox (options change per provider)
- API key `text_input(type="password")` — hidden for vLLM
- `@st.cache_resource` keyed on `(provider, model, api_key)` so switching rebuilds the engine

Agent loop note: all three providers use `openai.OpenAI` client with provider-specific `base_url` / `api_key`.
Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1`) supports tool calling, so the agent loop is unchanged.

### 3. What can I ask? *(new)*
Collapsible `st.expander` sections. Each example is a `st.button`; clicking it writes the text into `st.session_state` and pre-fills the query input.

| Category | Example queries |
|---|---|
| Plays & favourites | Most played tracks · Most skipped · All loved tracks |
| Artists | Kanye's genres · Top artists by play count · Who features most? |
| Lyrics — mood | Dark melancholic songs about death · Uplifting songs about love |
| Lyrics — topics | Songs about money · Albums with both love and death |
| Lyrics — language | Languages in my library · English + Russian songs sharing a topic |
| Vocabulary | Tracks with the most varied vocabulary · Most repetitive songs |
| Places | Artists who mention New York · All cities in my library |
| Eras | Tracks per era · Top artists per decade |
| Graph traversal | Artist bridges · Cross-genre topic connections |

---

## Main Area — Welcome State

When no query has been submitted yet (history is empty), show a **capability overview card** instead of blank space:

```
┌─────────────────────────────────────────────────────────────┐
│  🎵 2,956 tracks · 1,150 artists · 1,532 albums             │
│                                                             │
│  📊 Graph          📝 Lyrics            🤖 Agent            │
│  genres · eras     moods · topics       multi-step          │
│  playlists         places · language    synthesis           │
│  collaborations    vocabulary metrics   graph rendering      │
└─────────────────────────────────────────────────────────────┘
```

Stats pulled live from Neo4j on first load (cached).

---

## `query_engine.py` Changes

### New function: `build_llm(provider, model, api_key)`

```python
def build_llm(provider, model=None, api_key=None) -> LLM:
    if provider == "vllm":
        return OpenAILike(
            model=model or VLLM_MODEL,
            api_base=VLLM_BASE_URL,
            api_key="not-needed",
            additional_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            ...
        )
    elif provider == "openai":
        return OpenAILike(model=model or "gpt-4o-mini", api_base="https://api.openai.com/v1", api_key=api_key, ...)
    elif provider == "claude":
        from llama_index.llms.anthropic import Anthropic   # pip install llama-index-llms-anthropic
        return Anthropic(model=model or "claude-haiku-4-5-20251001", api_key=api_key, ...)
```

### Updated signatures

```python
def build_engine(provider="vllm", model=None, api_key=None): ...
def build_agent(engine=None, provider="vllm", model=None, api_key=None, max_steps=8): ...
```

`build_agent` constructs its raw `openai.OpenAI` client as:

```python
base_urls = {
    "vllm":   VLLM_BASE_URL,
    "openai": "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com/v1",
}
client = OpenAI(base_url=base_urls[provider], api_key=api_key or "not-needed")
```

---

## Capabilities Currently Hidden from Users

| Capability | Where it lives | Currently visible? |
|---|---|---|
| Mood queries | `HAS_MOOD` edges, 13 moods | No |
| Topic queries | `HAS_TOPIC` edges, 50+ topics | No |
| Place mentions | `MENTIONS_PLACE` edges | No |
| Language filter | `t.language` ISO codes | No |
| Vocabulary richness | `type_token_ratio`, `repetition_rate` | No |
| Agent multi-step | Agent mode | Partially (caption only) |
| Self-healing Cypher | `_wrap_with_self_healing` | No |
| Graph visualisation | neovis.js panel | Only after query |

All of the above will be discoverable via the "What can I ask?" example query buttons.
