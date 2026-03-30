# Onboarding & Pipeline UX Plan — Library.xml Drag & Drop

## Overview

A guided setup flow embedded in the app that takes a user from a blank Neo4j instance
to a fully queryable Knowledge Graph, with live progress and an optional lyrics enrichment step.

---

## Pipeline stages (what actually runs)

```
Library.xml  (drag & drop)
     │
     ▼
[1] parse_library.py        ~seconds     → Data/library.json, Data/overview.json
     │
     ▼
[2] ingest_graph.py         ~2–5 min     → Neo4j KG (tracks, artists, albums, genres, eras, playlists)
     │
     ▼ (optional — user chooses)
[3] lyrics_pipeline.py      ~2–4 hours   → Data/track_themes.json
     │                                     (LRCLIB fetch + 3 LLM calls per track: moods, topics, places)
     ▼
[4] ingest_lyrics.py        ~1–2 min     → enriches Neo4j with Mood/Topic/Place nodes
     │
     ▼
App fully operational (all query dimensions available)
```

---

## UX States

### State 0 — Neo4j is empty (first run detection)

On app startup, check `MATCH (n) RETURN count(n) LIMIT 1`.
If the graph is empty → show the setup UI instead of the query interface.
If the graph has data → show the query interface (setup accessible from sidebar).

---

### State 1 — Upload

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   🎵 Apple Music Knowledge Graph                                │
│                                                                 │
│   To get started, export your Apple Music library:             │
│   File → Library → Export Library…  (saves Library.xml)        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │          Drag & drop Library.xml here                   │  │
│   │                  — or —                                 │  │
│   │              Browse files                               │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Implemented with `st.file_uploader("", type=["xml"], label_visibility="collapsed")`.
File stays in memory — no disk write needed until parsing begins.

---

### State 2 — Preview (file selected, not yet ingested)

Parse the XML in memory just enough to count tracks and playlists (fast, no Neo4j).

```
✅ Library.xml loaded

   Tracks      2,956
   Artists     ~1,150 (estimated)
   Playlists   8
   File size   18.4 MB

   [ Build Knowledge Graph ]
```

---

### State 3 — KG ingestion running

Live progress via a background thread + `st.fragment(run_every=1)` polling loop.

```
┌─ Building your Knowledge Graph ──────────────────────────────┐
│                                                              │
│  ✅  Parse Library.xml          2,956 tracks · 8 playlists  │
│  ⏳  Ingest into Neo4j          [████████░░] 78%  (2,305 / 2,956) │
│  ○   Lyrics enrichment          (next step)                 │
│                                                              │
│  Currently: "Pure Cocaine" — Metro Boomin                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### State 4 — KG ready, lyrics decision

```
┌─ Knowledge Graph ready ──────────────────────────────────────┐
│                                                              │
│  ✅  2,956 tracks  ·  1,150 artists  ·  1,532 albums         │
│                                                              │
│  Optional: Lyrics Enrichment                                 │
│  Adds moods, topics, places, language and vocabulary         │
│  metrics to ~65% of your tracks via LRCLIB + local LLM.     │
│                                                              │
│  ⚠️  Requires local vLLM running.  Est. time: 2–4 hours.    │
│                                                              │
│  [ Start Lyrics Enrichment ]   [ Skip — use basic graph ]   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

The "Skip" path makes the app immediately usable for all graph-only queries.
The lyrics step can be triggered later from the sidebar → "Settings → Enrich lyrics".

---

### State 5 — Lyrics pipeline running

This is the long-running step. The UI must:
- Show per-track progress
- Allow the user to start querying while enrichment runs in the background
- Handle interruption gracefully (pipeline is already resumable on re-run)

```
┌─ Lyrics Enrichment ──────────────────────────────────────────┐
│                                                              │
│  Overall     [█████░░░░░] 47%  (1,387 / 2,956 tracks)       │
│  Found       901 tracks with lyrics                          │
│  Skipped     486 tracks (no lyrics on LRCLIB)               │
│                                                              │
│  Currently: "Spinnin" — Metro Boomin                        │
│                                                              │
│  Estimated time remaining: ~1h 45m                          │
│                                                              │
│  [ Continue to app — enrichment runs in background ]        │
│                                          [ Cancel ]         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

A persistent status badge appears in the sidebar while enrichment runs in the background:

```
sidebar:
  ⏳ Lyrics enrichment  47%  (1,387 / 2,956)
```

---

### State 6 — Done

```
✅  Lyrics enrichment complete

   Tracks with lyrics    1,921 / 2,956  (65%)
   Moods tagged          1,847 tracks
   Topics tagged         1,891 tracks
   Places tagged         743 tracks

   [ Start exploring → ]
```

---

## Technical Implementation

### Where it lives in app.py

```python
def is_graph_empty() -> bool:
    # cached, checked once per session
    result = graph_store.structured_query("MATCH (n) RETURN count(n) AS c LIMIT 1")
    return result[0]["c"] == 0

if is_graph_empty():
    show_onboarding()
else:
    show_main_app()
```

A sidebar link "⚙️ Re-ingest library" lets the user redo setup even when the graph has data.

---

### Background threading

Streamlit is single-threaded per session. Long pipeline steps run in a `threading.Thread`,
with progress written to `st.session_state` via a shared dict.

```python
# Shared progress dict (written by thread, read by Streamlit)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = {
        "stage":         "idle",   # idle | parsing | ingesting | lyrics | done | error
        "progress":      0,
        "total":         0,
        "current_track": "",
        "lyrics_found":  0,
        "error":         None,
        "thread":        None,
    }

def run_pipeline(xml_bytes: bytes, run_lyrics: bool):
    p = st.session_state.pipeline

    # Stage 1: parse
    p["stage"] = "parsing"
    library = parse_library_from_bytes(xml_bytes)            # refactored to accept bytes

    # Stage 2: ingest graph
    p["stage"] = "ingesting"
    p["total"] = len(library["tracks"])
    ingest_graph(library, on_progress=lambda i, name: p.update(progress=i, current_track=name))

    if not run_lyrics:
        p["stage"] = "done"
        return

    # Stage 3: lyrics pipeline (already async — run in thread via asyncio.run)
    p["stage"] = "lyrics"
    asyncio.run(run_lyrics_pipeline(
        on_progress=lambda i, found, name: p.update(progress=i, lyrics_found=found, current_track=name)
    ))

    # Stage 4: ingest lyrics results
    ingest_lyrics()
    p["stage"] = "done"

# Launch
thread = threading.Thread(target=run_pipeline, args=(xml_bytes, run_lyrics), daemon=True)
st.session_state.pipeline["thread"] = thread
thread.start()
```

---

### Progress polling with st.fragment

`st.fragment(run_every=1)` rerenders just the progress block every second
without triggering a full script rerun:

```python
@st.fragment(run_every=1)
def pipeline_progress():
    p = st.session_state.pipeline
    if p["stage"] == "idle":
        return
    if p["stage"] == "done":
        st.success("Done!")
        st.rerun()   # full rerun to switch to main app

    # render progress bars, current track, ETA ...
    st.progress(p["progress"] / max(p["total"], 1))
    st.caption(f"Currently: {p['current_track']}")
```

---

### Refactoring needed in data_processing scripts

The pipeline scripts currently read from / write to disk paths and are run as `__main__`.
They need small refactors to be importable and accept callbacks:

| Script | Refactor needed |
|---|---|
| `parse_library.py` | Add `parse_from_bytes(xml_bytes) -> dict` function |
| `ingest_graph.py` | Add `on_progress: Callable[[int, str], None]` param to main ingest function |
| `lyrics_pipeline.py` | Add `on_progress` callback to async pipeline; ensure OUTPUT_PATH is configurable |
| `ingest_lyrics.py` | No major change — reads from file, can be called as-is |

---

### ETA estimation (lyrics step)

Track the rolling average time per track over the last 50 tracks:

```python
from collections import deque
import time

times = deque(maxlen=50)
last_t = time.time()

def on_track_done(i, found, name):
    now = time.time()
    times.append(now - last_t)
    avg = sum(times) / len(times)
    remaining = (total - i) * avg
    # format remaining as "Xh Ym" and store in session state
```

---

## Sidebar integration (while enrichment runs in background)

```python
# In sidebar, always rendered
p = st.session_state.get("pipeline", {})
if p.get("stage") == "lyrics":
    pct = int(100 * p["progress"] / max(p["total"], 1))
    st.sidebar.progress(pct / 100)
    st.sidebar.caption(f"⏳ Lyrics enrichment  {pct}%  ({p['progress']} / {p['total']})")
```

---

## Lyrics step — LLM provider integration

### Two sub-steps with different setup burdens

**Sub-step A: Fetch lyrics via LRCLIB**
Zero setup. Free public API, no authentication required. Unchanged regardless of provider choice.

**Sub-step B: LLM classification (moods / topics / places)**
Currently hardcoded to local vLLM (`LLM_URL`, `LLM_MODEL` at top of `lyrics_pipeline.py`).
Should be refactored to respect the same provider choice as the query engine.

### Setup story per provider

| Provider | Extra setup for lyrics | Est. time (2,956 tracks) | Est. cost |
|---|---|---|---|
| vLLM (local) | GPU + model already running | 2–4 hours | $0 |
| OpenAI | API key already entered in UI | 30–60 min | ~$1–2 |
| Claude | API key already entered in UI | 30–60 min | ~$2–4 |

With OpenAI or Claude the user has already done the setup (entered their key for the query engine).
The lyrics pipeline reuses it — no extra configuration needed.

Cost basis: ~5,700 small classification calls (3 per track with lyrics).
At `gpt-4o-mini` rates (~$0.15/1M input, ~$0.60/1M output), ~500 tokens in + ~100 tokens out per call
≈ **$1–2 for a full 3,000-track library.**

### Resulting onboarding flow (with API provider)

```
1. Drag & drop Library.xml                   ← no setup
2. [ Build Knowledge Graph ]                 ← runs locally, ~3 min
3. LLM provider already chosen in sidebar    ← done, key already entered
   [ Enrich with Lyrics (~$1, ~45 min) ]
   [ Skip — use basic graph ]
```

No GPU required if the user is on OpenAI or Claude.

### What changes in `lyrics_pipeline.py`

The raw `aiohttp` POST calls to `LLM_URL` get replaced with calls through the shared
`build_llm(provider, model, api_key)` factory. The LRCLIB fetch logic is untouched.
The `on_progress` callback and resumability logic are also untouched.

Show a cost estimate and provider label in the UI before the user confirms:

```
Lyrics Enrichment

  Lyrics source    LRCLIB (free, no setup)
  LLM provider     OpenAI · gpt-4o-mini
  Est. cost        ~$1–2
  Est. time        ~45 min
  Resumable        Yes — safe to cancel and continue later

  [ Start ]   [ Skip ]
```

---

## Edge cases

| Scenario | Handling |
|---|---|
| User uploads wrong file (not a valid Library.xml) | Catch XML parse error, show clear message |
| Neo4j not running when ingestion starts | Catch connection error before launching thread, show instructions |
| vLLM not running when lyrics step starts | Check `/health` endpoint before starting, offer to skip |
| User closes browser mid-enrichment | Thread is daemon=True — dies with the Streamlit process. Pipeline is resumable because `lyrics_pipeline.py` already skips processed tracks. Re-upload not needed — just re-trigger from sidebar. |
| Library already ingested (re-upload) | Warn: "Graph already has data — re-ingesting will clear and rebuild." Require confirmation. |
| Lyrics pipeline LLM provider | Currently hardcoded to local vLLM. Future: respect the LLM provider setting from the sidebar. |
