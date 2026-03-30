# Apple Music Knowledge Graph

A personal portfolio project that transforms an Apple Music library into a queryable Knowledge Graph, enabling natural language exploration of listening history through a Graph RAG pipeline.

---

## Overview

Most music analytics tools treat a library as a flat table — tracks with attributes. This project models it as a **graph**, where the relationships between tracks, artists, albums, genres, and eras are first-class citizens. On top of that graph sits a natural language interface: ask a question in plain English, get back a structured answer and an interactive visualisation of the exact subgraph that produced it.

---

## Architecture

```
Library.xml (Apple Music export)
        ↓
parse_library.py        → library.json + overview.json
        ↓
ingest_graph.py         → Neo4j Knowledge Graph
        ↓
┌──────────────────────────────────────────────┐
│              Streamlit App                   │
│                                              │
│  User question (natural language)            │
│         ↓                                    │
│  LlamaIndex PropertyGraphQueryEngine         │
│  + Neo4jPropertyGraphStore                   │
│  + custom schema prompt                      │
│         ↓                                    │
│  vLLM (Qwen3.5-9B-AWQ)                       │
│  generates Cypher  ← thinks via <think> tags │
│         ↓                                    │
│  Neo4j executes query                        │
│         ↓                    ↓               │
│  vLLM generates answer    neovis.js          │
│  from results             renders subgraph   │
│         ↓                    ↓               │
│  Text answer          Interactive graph      │
└──────────────────────────────────────────────┘
```

---

## Knowledge Graph Schema

```
(:Track)-[:BY]----------->(:Artist)       primary artist
(:Track)-[:FEATURES]------>(:Artist)       featured artists
(:Track)-[:ON]------------>(:Album)        album tracks only
(:Track)-[:IS_SINGLE]----->(:Single)       standalone singles
(:Track)-[:IN_GENRE]------>(:Genre)
(:Track)-[:IN_ERA]-------->(:Era)
(:Track)-[:IN_PLAYLIST]--->(:Playlist)
(:Album)-[:BY]------------>(:Artist)
(:Artist)-[:IN_GENRE]----->(:Genre)        derived, weighted by track count
```

**Track properties:** `name`, `year`, `release_date`, `duration_ms`, `play_count`, `skip_count`, `loved`, `explicit`, `date_added`, `track_number`

---

## Library Stats

| | |
|---|---|
| Tracks | 2,956 |
| Unique artists | 1,150 |
| Unique albums | 1,532 |
| Genres | ~50 (after normalisation) |
| Total playtime | 175 hours |
| Explicit tracks | 64% |
| Playlists | 8 |

---

## What's Been Built

### 1. Data Pipeline (`data_processing/`)

**`parse_library.py`**
Parses the Apple Music `Library.xml` (plist format) into structured JSON.
- Outputs `library.json` (full track + playlist data) and `overview.json` (aggregated stats)
- Handles all 51 available Apple Music fields

**`visualize_library.py`**
Generates a self-contained interactive HTML dashboard from the parsed data.
- 9 Plotly charts: genre breakdown, top artists, release year distribution, play count histogram, most played/skipped tracks, library growth over time, track flags
- Dark theme, no server required — open directly in browser

**`ingest_graph.py`**
Ingests `library.json` into a local Neo4j instance.
- Normalises genre variants (Hip-Hop, Rap, Old School Rap → Hip-Hop/Rap)
- Strips album edition suffixes (Deluxe Edition, Remaster, etc.)
- Parses featured artists from collaboration strings (`&`, `,`, `featuring`, `x`, `with`)
- Uses `Album Artist` as canonical primary artist identity
- Derives Era nodes from release year (Pre-90s, 90s, 2000s, 2010s, 2020s)
- Post-ingestion step creates weighted `Artist-[:IN_GENRE]->Genre` edges

**`visualize_graph.py`**
Pulls a subgraph from Neo4j and renders it as an interactive HTML graph using pyvis.
- Currently visualises top 10 artists with their tracks, albums, and genres
- Colour-coded node types, hover tooltips, force-directed layout

**`data_processing/test_neovis.html`**
Standalone proof-of-concept for neovis.js graph visualisation.
- Connects directly to Neo4j from the browser via bolt
- Renders any Cypher query result as an interactive graph automatically
- Colour-coded by node type (Artist, Track, Album, Genre)
- Will be embedded in the Streamlit app to visualise query results

### 2. Graph RAG Pipeline (`query_engine.py`)

**`query_engine.py`**
LlamaIndex `TextToCypherRetriever` wired to Neo4j and a local vLLM endpoint.
- Uses `Neo4jPropertyGraphStore` + APOC for live schema introspection
- Custom schema prompt injected so the model understands the full KG structure
- `TextToCypherRetriever` handles: schema → Cypher generation → Neo4j execution → natural language answer
- `enable_thinking: False` passed via `extra_body` to skip Qwen3's `<think>` tokens for Cypher (deterministic, fast)
- `summarize_response=True` — second LLM call converts raw query results into a readable answer

**vLLM serve command:**
```bash
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b-awq \
  --quantization awq_marlin \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --port 8000
```
- `--reasoning-parser qwen3` — vLLM strips `<think>` tokens automatically; `enable_thinking: False` passed per-request for Cypher generation
- OpenAI-compatible endpoint at `http://localhost:8000/v1`

**Example queries and answers:**

| Question | Answer |
|---|---|
| What genres does Kanye West span? | Hip-Hop/Rap and Pop |
| What are my top 10 most played tracks? | ЮМОРИСТ (93), Spinnin (88), Pure Cocaine (81) … |
| Which artists feature most on my tracks? | Hamilton cast, Future, Young Thug … |
| How many tracks per era? | 2010s: 1,299 · 2020s: 1,151 · 2000s: 273 · 90s: 117 · Pre-90s: 116 |

---

## What's Next

### Streamlit App
A web interface with two panels per query:
- **Left** — natural language answer from LlamaIndex
- **Right** — neovis.js subgraph of the exact nodes and edges involved in the answer

### Docker
Once the full pipeline is working locally, the entire stack (Neo4j + ingestion + Streamlit app) will be containerised so anyone can run it against their own `Library.xml` with a single command.

---

## Why Graph over SQL?

A flat SQL table handles aggregation queries fine. The graph earns its keep for:

- **Path finding** — "What connects artist X and artist Y in my library?" requires recursive joins in SQL, one line in Cypher
- **Variable-depth traversal** — "Find artists within 2 hops of my taste profile" is natural in Cypher, exponentially messy in SQL
- **Subgraph visualisation** — query results are already a graph structure, making the pyvis visualisation a natural output
- **Extensibility** — adding external data (artist influences, collaborations, audio features from Spotify API) fits the graph model without schema migrations

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data parsing | Python (`plistlib`) |
| Dashboard | Plotly |
| Graph database | Neo4j |
| Graph visualisation | neovis.js (browser, query-driven) |
| Graph visualisation (static) | pyvis |
| RAG pipeline | LlamaIndex (`PropertyGraphQueryEngine`) |
| LLM inference | vLLM (local, OpenAI-compatible) |
| Model | Qwen3.5-9B-AWQ (AWQ quantised via awq_marlin) |
| App framework | Streamlit |
| LLM client | `llama_index.llms.openai_like.OpenAILike` |

---

## Running Locally

### 1. Parse the library
```bash
python data_processing/parse_library.py
```

### 2. Generate the dashboard
```bash
python data_processing/visualize_library.py
# open Data/dashboard.html in browser
```

### 3. Start Neo4j, then ingest the graph
```bash
python data_processing/ingest_graph.py
# explore at http://localhost:7474
```

### 4. Visualise a subgraph
```bash
python data_processing/visualize_graph.py
# open Data/graph.html in browser
```

### 5. (Coming) Run the app
```bash
streamlit run app.py
```
