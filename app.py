"""
app.py

Streamlit app — Apple Music Knowledge Graph explorer.

Left panel:  natural language answer (LlamaIndex + vLLM)
Right panel: neovis.js interactive subgraph of the query result

Run:
    streamlit run app.py
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Apple Music KG",
    page_icon="🎵",
    layout="wide",
)

st.markdown("""
<style>
  /* Dark background to match the graph visualisation */
  [data-testid="stAppViewContainer"] { background: #0d0d0d; }
  [data-testid="stHeader"]           { background: transparent; }
  section[data-testid="stSidebar"]   { background: #111; }
  h1, h2, h3, p, label              { color: #e0e0e0 !important; }
  .stTextInput input                 { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; }
  .stButton button                   { background: #1DB954; color: #000; font-weight: 600; border: none; }
  .stButton button:hover             { background: #17a349; }
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

# ── Query engine / agent (cached — built once per session) ────────────────────

@st.cache_resource(show_spinner="Connecting to Neo4j and vLLM ...")
def get_engine():
    from query_engine import build_engine
    return build_engine()

@st.cache_resource(show_spinner="Building agent ...")
def get_agent():
    from query_engine import build_agent
    return build_agent(engine=get_engine())

# ── neovis.js HTML template ───────────────────────────────────────────────────

def neovis_html(cypher: str) -> str:
    # Escape backticks and backslashes in the Cypher string for JS template literal
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
  <div class="legend-item"><div class="dot" style="background:#1DB954"></div><span class="legend-label">Artist</span></div>
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
          static: {{ color: {{ background: "#1DB954", border: "#1DB954" }}, size: 30 }}
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
    """
    The LLM-generated Cypher returns scalar data (counts, names).
    For neovis we need a query that returns actual graph nodes and relationships.
    We wrap the original query as a subquery and re-fetch the involved nodes.

    Fallback: if we can't determine the right nodes, show a default subgraph.
    """
    # neovis needs RETURN of nodes/rels, not scalars.
    # Simple heuristic: re-run the subgraph for whatever entities the Cypher references.
    # The safest approach: run the original query inside a WITH and pull node objects.

    # Strip any trailing LIMIT / ORDER BY / RETURN to replace the RETURN clause
    upper = query_cypher.upper()
    if "MATCH" not in upper:
        return query_cypher  # pass through if it's already structured

    # Wrap in a subquery that returns graph elements
    # We detect what node variables are used in the MATCH and return them with relationships
    wrapped = f"""
CALL {{
  {query_cypher}
}}
RETURN *
"""
    return wrapped


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of {question, answer, cypher, mode, steps}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio(
        "",
        ["Standard", "Agent"],
        label_visibility="collapsed",
        help="Standard: single query → answer + graph.\nAgent: LLM runs multiple queries autonomously then synthesises.",
    )
    st.divider()
    if mode == "Agent":
        st.caption("The agent will call the query engine multiple times to build a comprehensive answer.")

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎵 Apple Music Knowledge Graph")
if mode == "Standard":
    st.caption("Ask a question about your library — get a natural language answer and an interactive graph.")
else:
    st.caption("Agent mode — ask a broad question and the agent will run multiple queries to answer it.")

question = st.text_input(
    label="Question",
    placeholder="e.g. What genres does Kanye West span? · What are my most played tracks?",
    label_visibility="collapsed",
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

    if mode == "Standard":
        engine = get_engine()

        with st.spinner("Generating Cypher and querying the graph ..."):
            response = engine.query(question.strip())

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
            "mode":     "Standard",
            "steps":    [],
        })

        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Answer")
            st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
            if cypher:
                st.markdown("#### Cypher")
                st.code(cypher, language="cypher")

        with right:
            st.markdown("#### Graph")
            if cypher:
                graph_cypher = make_graph_cypher(cypher)
                components.html(neovis_html(graph_cypher), height=600, scrolling=False)
            else:
                st.info("No graph available for this query.")

    else:  # Agent mode
        run_agent = get_agent()
        steps = []

        with st.status("Agent is thinking ...", expanded=True) as status:
            def on_step(step):
                st.markdown(f"**→ {step['question']}**")
                st.caption(step["result"])
                steps.append(step)

            answer, steps, viz = run_agent(question.strip(), on_step=on_step)
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

# ── Query history ──────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.markdown("#### Query History")
    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        label = f"{idx}. [{entry.get('mode', 'Standard')}] {entry['question']}"
        with st.expander(label):
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
            if entry.get("cypher"):
                st.code(entry["cypher"], language="cypher")
            if entry.get("steps"):
                st.markdown("**Queries run by agent:**")
                for step in entry["steps"]:
                    st.markdown(f"- *{step['question']}*")
