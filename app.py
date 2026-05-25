"""
app.py

Streamlit app — Apple Music Knowledge Graph explorer.

Left panel:  natural language answer (LlamaIndex + vLLM)
Right panel: neovis.js interactive subgraph of the query result

Run:
    streamlit run app.py
"""

import asyncio
import concurrent.futures
import json
import os
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL",  "http://localhost:8000/v1")
VLLM_MODEL     = os.getenv("VLLM_MODEL",     "qwen3.5-9b-awq")
LLM_URL        = VLLM_BASE_URL.rstrip("/v1").rstrip("/") + "/v1/chat/completions"

DATA_DIR       = Path("Data")
LIBRARY_XML    = DATA_DIR / "Library.xml"
LIBRARY_JSON   = DATA_DIR / "library.json"
THEMES_JSON    = DATA_DIR / "track_themes.json"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TuneMap",
    page_icon="♪",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d0d; }
  [data-testid="stHeader"]           { background: transparent; }
  section[data-testid="stSidebar"]   { background: #111; }
  h1, h2, h3, p, label              { color: #e0e0e0 !important; }
  .stTextInput input                 { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; }
  .stButton button                   { background: #A8303F; color: #fff; font-weight: 600; border: none; }
  .stButton button:hover             { background: #8a2533; }
  .answer-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #e0e0e0;
    font-size: 0.95rem;
    line-height: 1.6;
    min-height: 200px;
  }
  .cypher-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #888;
    font-family: monospace;
    font-size: 0.82rem;
    margin-top: 0.8rem;
    white-space: pre-wrap;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = {
        "stage":         "idle",  # idle | parsing | ingesting | done | lyrics | ingesting_lyrics | complete | error
        "progress":      0,
        "total":         0,
        "current_track": "",
        "lyrics_found":  0,
        "error":         None,
        "thread":        None,
        "library":       None,
        "overview":      None,
    }

# ── Neo4j graph store (shared) ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_graph_store():
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )

@st.cache_data(ttl=30, show_spinner=False)
def is_graph_empty() -> bool:
    try:
        result = get_graph_store().structured_query("MATCH (n) RETURN count(n) AS c LIMIT 1")
        return result[0]["c"] == 0
    except Exception:
        return True

if "graph_empty" not in st.session_state:
    st.session_state.graph_empty = is_graph_empty()

# ── Query engine / agent (cached) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Neo4j and vLLM ...")
def get_engine():
    from query_engine import build_engine
    return build_engine()

@st.cache_resource(show_spinner="Building agent ...")
def get_agent():
    from query_engine import build_agent
    return build_agent(engine=get_engine())

# ── Background pipeline threads ───────────────────────────────────────────────

def _run_ingest_pipeline(p: dict, xml_bytes: bytes):
    try:
        p["stage"] = "parsing"
        from data_processing.parse_library import parse_from_bytes
        library, overview = parse_from_bytes(xml_bytes)
        p["library"]  = library
        p["overview"] = overview
        p["total"]    = len(library["tracks"])

        # Save to disk so it survives container restarts
        DATA_DIR.mkdir(exist_ok=True)
        with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
            json.dump(library, f, ensure_ascii=False)

        p["stage"]    = "ingesting"
        p["progress"] = 0

        from data_processing.ingest_graph import ingest

        def on_graph_progress(current, total, name):
            p["progress"]      = current
            p["total"]         = total
            p["current_track"] = name

        ingest(library, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, on_progress=on_graph_progress)

        p["stage"]    = "done"
        p["progress"] = p["total"]

    except Exception as e:
        p["stage"] = "error"
        p["error"] = str(e)


def _run_lyrics_pipeline(p: dict, library: dict):
    try:
        p["stage"]        = "lyrics"
        p["progress"]     = 0
        p["lyrics_found"] = 0

        from data_processing.lyrics_pipeline import run as run_lyrics

        def on_lyrics_progress(current, total, found, name):
            p["progress"]      = current
            p["total"]         = total
            p["lyrics_found"]  = found
            p["current_track"] = name

        asyncio.run(run_lyrics(
            library,
            THEMES_JSON,
            llm_url=LLM_URL,
            llm_model=VLLM_MODEL,
            on_progress=on_lyrics_progress,
        ))

        p["stage"]    = "ingesting_lyrics"
        p["progress"] = 0

        with open(THEMES_JSON, encoding="utf-8") as f:
            themes = json.load(f)

        from data_processing.ingest_lyrics import ingest as ingest_lyrics

        def on_ingest_progress(current, total):
            p["progress"] = current
            p["total"]    = total

        ingest_lyrics(library, themes, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, on_progress=on_ingest_progress)

        p["stage"] = "complete"

    except Exception as e:
        p["stage"] = "error"
        p["error"] = str(e)


def _launch_thread(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    st.session_state.pipeline["thread"] = t
    t.start()

# ── neovis.js HTML template ───────────────────────────────────────────────────

def neovis_html(cypher: str) -> str:
    cypher_escaped = cypher.replace("\\", "\\\\").replace("`", "\\`")
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://unpkg.com/neovis.js@2.0.2/dist/neovis.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0d0d0d; }}
    #viz {{ width:100%; height:580px; }}
    #legend {{
      position:fixed; bottom:16px; right:16px;
      background:#1a1a1a; border:1px solid #333; border-radius:10px;
      padding:12px 16px; z-index:999;
    }}
    .legend-title {{ color:#888; font-size:10px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; font-family:sans-serif; }}
    .legend-item  {{ display:flex; align-items:center; gap:7px; margin-bottom:5px; }}
    .dot          {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
    .legend-label {{ color:#e0e0e0; font-size:12px; font-family:sans-serif; }}
  </style>
</head>
<body>
<div id="viz"></div>
<div id="legend">
  <div class="legend-title">Node types</div>
  <div class="legend-item"><div class="dot" style="background:#A8303F"></div><span class="legend-label">Artist</span></div>
  <div class="legend-item"><div class="dot" style="background:#4ECDC4"></div><span class="legend-label">Track</span></div>
  <div class="legend-item"><div class="dot" style="background:#FFE66D"></div><span class="legend-label">Album</span></div>
  <div class="legend-item"><div class="dot" style="background:#FF6B6B"></div><span class="legend-label">Genre</span></div>
  <div class="legend-item"><div class="dot" style="background:#A29BFE"></div><span class="legend-label">Era</span></div>
  <div class="legend-item"><div class="dot" style="background:#FD79A8"></div><span class="legend-label">Playlist</span></div>
  <div class="legend-item"><div class="dot" style="background:#FDCB6E"></div><span class="legend-label">Mood</span></div>
  <div class="legend-item"><div class="dot" style="background:#E17055"></div><span class="legend-label">Topic</span></div>
  <div class="legend-item"><div class="dot" style="background:#00CEC9"></div><span class="legend-label">Place</span></div>
  <div class="legend-item"><div class="dot" style="background:#81ECEC"></div><span class="legend-label">Single</span></div>
</div>
<script>
  const config = {{
    containerId: "viz",
    neo4j: {{
      serverUrl:      "{NEO4J_URI}",
      serverUser:     "{NEO4J_USER}",
      serverPassword: "{NEO4J_PASSWORD}",
    }},
    visConfig: {{
      nodes: {{
        shape: "dot",
        font: {{ color: "#e0e0e0", size: 12, face: "Inter, Helvetica, sans-serif", background: "none", strokeWidth: 0 }},
        borderWidth: 0,
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.4 }} }},
        smooth: {{ type: "continuous" }},
      }},
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{ gravitationalConstant: -50, springLength: 100, damping: 0.5 }},
        stabilization: {{ iterations: 200 }},
        minVelocity: 1,
      }},
    }},
    labels: {{
      Artist: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#A8303F", border: "#A8303F" }}, size: 30 }}
        }}
      }},
      Track: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#4ECDC4", border: "#4ECDC4" }}, size: 8 }}
        }}
      }},
      Album: {{
        label: "title",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#FFE66D", border: "#FFE66D" }}, size: 20 }}
        }}
      }},
      Genre: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#FF6B6B", border: "#FF6B6B" }}, size: 25 }}
        }}
      }},
      Era: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#A29BFE", border: "#A29BFE" }}, size: 22 }}
        }}
      }},
      Playlist: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#FD79A8", border: "#FD79A8" }}, size: 22 }}
        }}
      }},
      Mood: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#FDCB6E", border: "#FDCB6E" }}, size: 18 }}
        }}
      }},
      Topic: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#E17055", border: "#E17055" }}, size: 18 }}
        }}
      }},
      Place: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#00CEC9", border: "#00CEC9" }}, size: 16 }}
        }}
      }},
      Single: {{
        label: "name",
        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
          static: {{ color: {{ background: "#81ECEC", border: "#81ECEC" }}, size: 14 }}
        }}
      }},
    }},
    relationships: {{
      BY:            {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#555555", width: 1 }} }} }},
      FEATURES:      {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#FF6B6B", dashes: true, width: 1.5 }} }} }},
      ON:            {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#444444", width: 1 }} }} }},
      IN_GENRE:      {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#333333", width: 1 }} }} }},
      IN_ERA:        {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#444466", width: 1 }} }} }},
      IN_PLAYLIST:   {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#553355", width: 1 }} }} }},
      IS_SINGLE:     {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#445544", width: 1 }} }} }},
      HAS_MOOD:      {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#FDCB6E", width: 1 }} }} }},
      HAS_TOPIC:     {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#E17055", width: 1 }} }} }},
      MENTIONS_PLACE: {{ [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{ static: {{ color: "#00CEC9", width: 1 }} }} }},
    }},
    initialCypher: `{cypher_escaped}`,
  }};
  const viz = new NeoVis.default(config);
  viz.render();
</script>
</body>
</html>"""


def make_graph_cypher(query_cypher: str) -> str:
    upper = query_cypher.upper()
    if "MATCH" not in upper:
        return query_cypher
    wrapped = f"""
CALL {{
  {query_cypher}
}}
RETURN *
"""
    return wrapped

# ── Onboarding UI ─────────────────────────────────────────────────────────────

@st.fragment(run_every=1)
def _pipeline_progress():
    """Polls pipeline state and re-renders progress. Triggers full rerun on completion."""
    p = st.session_state.pipeline
    stage = p["stage"]

    if stage == "parsing":
        st.info("Parsing Library.xml...")

    elif stage == "ingesting":
        total = max(p["total"], 1)
        st.progress(p["progress"] / total)
        st.caption(f"{p['progress']} / {p['total']} tracks ingested")
        if p["current_track"]:
            st.caption(f"Currently: {p['current_track']}")

    elif stage == "lyrics":
        total = max(p["total"], 1)
        st.progress(p["progress"] / total)
        st.caption(
            f"{p['progress']} / {p['total']} tracks processed  ·  "
            f"{p['lyrics_found']} with lyrics"
        )
        if p["current_track"]:
            st.caption(f"Currently: {p['current_track']}")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Continue to app (enrichment runs in background)"):
                p["stage"] = "complete"
                st.rerun()
        with col2:
            if st.button("Cancel"):
                p["stage"] = "complete"
                st.rerun()

    elif stage == "ingesting_lyrics":
        total = max(p["total"], 1)
        st.progress(p["progress"] / total)
        st.caption(f"Ingesting lyrics into graph... {p['progress']} / {p['total']}")

    elif stage in ("done", "complete"):
        st.rerun()

    elif stage == "error":
        st.error(f"Pipeline error: {p['error']}")


def show_onboarding():
    st.title("♪ TuneMap")
    p = st.session_state.pipeline
    stage = p["stage"]

    if stage == "idle":
        st.markdown("### Get started")
        st.markdown(
            "Export your Apple Music library: **File → Library → Export Library…**  "
            "then drop the `Library.xml` file below."
        )

        xml_bytes = None

        if LIBRARY_XML.exists():
            st.info(f"Found `Data/Library.xml` on disk.")
            if st.button("Use existing Library.xml"):
                xml_bytes = LIBRARY_XML.read_bytes()
        else:
            uploaded = st.file_uploader("Upload Library.xml", type=["xml"], label_visibility="collapsed")
            if uploaded is not None:
                xml_bytes = uploaded.read()
                DATA_DIR.mkdir(exist_ok=True)
                LIBRARY_XML.write_bytes(xml_bytes)

        if xml_bytes is not None:
            st.session_state.pipeline["stage"] = "parsing"
            _launch_thread(_run_ingest_pipeline, st.session_state.pipeline, xml_bytes)
            st.rerun()

    elif stage in ("parsing", "ingesting"):
        st.markdown("### Building your Knowledge Graph")
        _pipeline_progress()

    elif stage == "done":
        overview = p.get("overview") or {}
        totals   = overview.get("totals", {})

        st.success("Knowledge Graph ready!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tracks",  totals.get("tracks",          "—"))
        col2.metric("Artists", totals.get("unique_artists",  "—"))
        col3.metric("Albums",  totals.get("unique_albums",   "—"))

        st.markdown("---")
        st.markdown("#### Optional: Lyrics Enrichment")
        st.markdown(
            "Adds moods, topics, places, language and vocabulary metrics "
            "to a subset of your tracks available via LRCLIB + local vLLM.  \n"
            "⚠️ Requires local vLLM running. This may take hours depending on library size, model, and hardware."
        )

        col_lyrics, col_skip = st.columns([1, 1])
        with col_lyrics:
            if st.button("Start lyrics enrichment"):
                library = p.get("library")
                if library is None and LIBRARY_JSON.exists():
                    with open(LIBRARY_JSON, encoding="utf-8") as f:
                        library = json.load(f)
                if library:
                    st.session_state.pipeline["stage"] = "lyrics"
                    _launch_thread(_run_lyrics_pipeline, st.session_state.pipeline, library)
                    st.rerun()
        with col_skip:
            if st.button("Skip — start exploring →"):
                p["stage"] = "complete"
                st.rerun()

    elif stage in ("lyrics", "ingesting_lyrics"):
        st.markdown("### Lyrics Enrichment")
        _pipeline_progress()

    elif stage == "complete":
        # Clear caches and mark graph as populated
        get_graph_store.clear()
        is_graph_empty.clear()
        st.session_state.graph_empty = False
        st.rerun()

    elif stage == "error":
        st.error(f"Something went wrong: {p['error']}")
        if st.button("Try again"):
            st.session_state.pipeline["stage"] = "idle"
            st.rerun()

# ── Main app UI ───────────────────────────────────────────────────────────────

def show_main_app():
    p = st.session_state.get("pipeline", {})

    with st.sidebar:
        mode = st.radio(
            "Mode",
            ["Simple", "Agent"],
            index=1,
            help="Simple: one direct query, fast answer, no graph.\nAgent: runs multiple queries autonomously and synthesises a richer answer with an interactive graph.",
        )
        st.divider()

        # Lyrics enrichment progress badge
        if p.get("stage") == "lyrics":
            total = max(p.get("total", 1), 1)
            pct   = int(100 * p.get("progress", 0) / total)
            st.progress(pct / 100)
            st.caption(f"Lyrics enrichment {pct}% ({p.get('progress', 0)} / {total})")
            st.divider()

        # Re-ingest option
        with st.expander("Re-ingest library"):
            st.caption("This will wipe the current graph and rebuild from a new Library.xml.")
            if st.button("Reset & re-ingest"):
                st.session_state.pipeline = {
                    "stage": "idle", "progress": 0, "total": 0,
                    "current_track": "", "lyrics_found": 0,
                    "error": None, "thread": None,
                    "library": None, "overview": None,
                }
                get_graph_store.clear()
                get_engine.clear()
                get_agent.clear()
                st.rerun()

        st.divider()

        # ── Example queries ───────────────────────────────────────────────────
        st.markdown("### What can I ask?")

        EXAMPLE_QUERIES = {
            "Quick lookups": [
                "What are my top 10 most played tracks?",
                "Who are my top 10 artists by play count?",
                "What genres are in my library?",
                "How many tracks do I have per era?",
                "Which tracks have I marked as loved?",
                "What are my most skipped tracks?",
            ],
            "Deep dives (Agent)": [
                "Summarise my music taste",
                "What does my Hip-Hop listening say about me?",
                "Which artists do I keep coming back to and why?",
                "What eras and genres define my library?",
                "Find me artists who are connected through shared collaborators",
                "Which albums contain songs about both love and death?",
            ],
            "Lyrics & mood": [
                "What moods dominate my library?",
                "Find me dark melancholic songs about death",
                "Show me uplifting songs about friendship or love",
                "Which tracks have the most varied vocabulary?",
                "What languages are in my library?",
                "Which artists mention New York most in their lyrics?",
            ],
            "Visualise (Agent)": [
                "Graph my top 10 artists and the genres they belong to",
                "Graph my top 10 most played tracks and the artists behind them",
                "Graph which artists are featured on my top 5 artists' tracks",
                "Graph my genres and the top 3 artists in each",
                "Graph my top artists and the eras their music comes from",
            ],
        }

        for category, queries in EXAMPLE_QUERIES.items():
            with st.expander(category):
                for q in queries:
                    if st.button(q, key=f"eq_{q}", use_container_width=True):
                        st.session_state["prefill_question"] = q

        if mode == "Agent":
            st.caption("The agent will call the query engine multiple times to build a comprehensive answer.")

    st.title("♪ TuneMap")
    if mode == "Simple":
        st.caption("Ask a question and get a direct answer.")
    else:
        st.caption("Agent mode — ask a broad question and the agent will run multiple queries to answer it.")

    # Pick up any pre-filled question from sidebar button clicks
    if "prefill_question" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("prefill_question")

    question = st.text_input(
        label="Question",
        placeholder="e.g. What genres does Kanye West span? · What are my most played tracks?",
        label_visibility="collapsed",
        key="question_input",
    )

    col_ask, col_clear, _ = st.columns([1, 1, 4])
    with col_ask:
        ask = st.button("Ask", use_container_width=True)
    with col_clear:
        clear = st.button("Clear history", use_container_width=True)

    if clear:
        st.session_state.history = []

    st.divider()

    if ask and question.strip():

        QUERY_TIMEOUT = 180  # seconds

        if mode == "Simple":
            engine = get_engine()

            with st.spinner("Generating Cypher and querying the graph ..."):
                try:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = pool.submit(engine.query, question.strip())
                    response = future.result(timeout=QUERY_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    pool.shutdown(wait=False)
                    st.error("Request timed out after 2 minutes. Try a simpler question.")
                    st.stop()

                cypher = None
                for node in getattr(response, "source_nodes", []):
                    meta = node.node.metadata
                    if meta.get("query"):
                        cypher = meta["query"]
                        break

            st.session_state.history.append({
                "question": question.strip(),
                "answer":   str(response),
                "cypher":   cypher,
                "mode":     "Simple",
                "steps":    [],
            })

            st.markdown("#### Answer")
            st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
            if cypher:
                st.markdown("#### Cypher")
                st.code(cypher, language="cypher")

        else:  # Agent mode
            run_agent = get_agent()

            with st.status("Agent is thinking ...", expanded=True) as status:
                try:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = pool.submit(run_agent, question.strip())
                    answer, steps, viz = future.result(timeout=QUERY_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    pool.shutdown(wait=False)
                    status.update(label="Timed out", state="error")
                    st.error("Request timed out after 2 minutes. Try a more focused question.")
                    st.stop()

                for step in steps:
                    st.markdown(f"**→ {step['question']}**")
                    st.caption(step["result"])

                graph_label = " · graph ready" if viz else ""
                status.update(label=f"Done — {len(steps)} quer{'y' if len(steps) == 1 else 'ies'} run{graph_label}", state="complete")

            if viz:
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("#### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                with right:
                    st.markdown("#### Graph")
                    st.caption(viz["description"])
                    components.html(neovis_html(viz["cypher"]), height=600, scrolling=False)
                    with st.expander("Cypher", expanded=False):
                        st.code(viz["cypher"], language="cypher")
            else:
                st.markdown("#### Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "question": question.strip(),
                "answer":   answer,
                "cypher":   viz["cypher"] if viz else None,
                "mode":     "Agent",
                "steps":    steps,
            })

    elif ask and not question.strip():
        st.warning("Please enter a question.")

    if st.session_state.history:
        st.divider()
        st.markdown("#### Query History")
        for i, entry in enumerate(reversed(st.session_state.history)):
            idx   = len(st.session_state.history) - i
            label = f"{idx}. [{entry.get('mode', 'Standard')}] {entry['question']}"
            with st.expander(label):
                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
                if entry.get("cypher"):
                    st.code(entry["cypher"], language="cypher")
                if entry.get("steps"):
                    st.markdown("**Queries run by agent:**")
                    for step in entry["steps"]:
                        st.markdown(f"- *{step['question']}*")

# ── Entry point ───────────────────────────────────────────────────────────────

_stage = st.session_state.pipeline["stage"]
if _stage in ("parsing", "ingesting", "done", "lyrics", "ingesting_lyrics", "error") or st.session_state.graph_empty:
    show_onboarding()
else:
    show_main_app()
