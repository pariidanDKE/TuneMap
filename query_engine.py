"""
query_engine.py

Tests the LlamaIndex PropertyGraphQueryEngine against the Apple Music
Knowledge Graph using a local vLLM endpoint (Qwen3.5-9B-AWQ).

Usage:
    python query_engine.py
    python query_engine.py --question "What genres does Kanye West span?"
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# ── Config ────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL",  "http://localhost:8000/v1")
VLLM_MODEL     = os.getenv("VLLM_MODEL",     "qwen3.5-9b-awq")

# ── Schema prompt ─────────────────────────────────────────────────────────────
# Teaches the LLM the exact graph schema so it generates correct Cypher.

SCHEMA_PROMPT = """
You are a Cypher query expert for a Neo4j Knowledge Graph built from an Apple Music library.

## Graph Schema

### Node types and their properties:
- (:Track)    — name, year, release_date, duration_ms, play_count, skip_count, loved (bool), explicit (bool), date_added, track_number,
                lyrics_found (bool), language (ISO 639-1 code e.g. "en", "ru", "fr"),
                total_words (int), unique_words (int), type_token_ratio (float), repetition_rate (float)
- (:Artist)   — name
- (:Album)    — title, year
- (:Single)   — name
- (:Genre)    — name
- (:Era)      — name  (values: "Pre-90s", "90s", "2000s", "2010s", "2020s")
- (:Playlist) — name
- (:Mood)     — name  (values: "melancholic", "aggressive", "uplifting", "introspective", "boastful", "celebratory", "dark", "nostalgic", "anxious", "romantic", "defiant", "vulnerable", "cynical")
- (:Topic)    — name  (values: "god", "spirituality", "mother", "father", "death", "love", "sex", "desire", "friendship", "loyalty", "betrayal", "money", "fame", "freedom", "struggle", "fun", "heartbreak", "resentment", "redemption", "ego", "nostalgia", "dreams", "jealousy", "isolation", "insecurity", "mental health", "materialism", "pleasure", "moving on", "gang", "drugs", "prison", "police", "hustle", "government", "rebellion", "anger", "war", "identity", "alienation", "toxic relationship", "confidence", "youth", "passion", "technology", "escapism", "hometown", "hardship", "nature", "improvisation")
- (:Place)    — name  (lowercase city/country/region e.g. "new york", "toronto", "los angeles")

### Relationships:
- (:Track)-[:BY]------------->(:Artist)     primary artist of the track
- (:Track)-[:FEATURES]------->(:Artist)     featured/collaborating artist
- (:Track)-[:ON]------------->(:Album)      track belongs to this album
- (:Track)-[:IS_SINGLE]------>(:Single)     track is a standalone single
- (:Track)-[:IN_GENRE]------->(:Genre)      genre of the track
- (:Track)-[:IN_ERA]--------->(:Era)        decade era based on release year
- (:Track)-[:IN_PLAYLIST]---->(:Playlist)   track is in this user playlist
- (:Track)-[:HAS_MOOD]------->(:Mood)       mood(s) expressed in the lyrics
- (:Track)-[:HAS_TOPIC]------>(:Topic)      topic(s) referenced in the lyrics
- (:Track)-[:MENTIONS_PLACE]->(:Place)      real-world place mentioned in the lyrics
- (:Album)-[:BY]------------->(:Artist)     album's primary artist
- (:Artist)-[:IN_GENRE {track_count}]-->(:Genre)  derived genre affiliation (weighted by track count)

### Important notes:
- Artist identity is based on the Album Artist field (canonical). The Artist field may contain featured artists.
- Album titles have been normalised — edition suffixes (Deluxe, Remaster, etc.) are stripped.
- Genre has been normalised — all Hip-Hop variants map to "Hip-Hop/Rap".
- A track either has an [:ON]->(:Album) edge OR an [:IS_SINGLE]->(:Single) edge, never both.
- play_count and skip_count are integers (0 if never played/skipped).
- loved is a boolean — true means the user marked the track as a favourite.
- Mood, Topic and Place nodes only exist for tracks where lyrics_found = true (~65% of library).
- Topic and Mood names are always lowercase in the graph.
- type_token_ratio measures vocabulary richness (0–1): higher = more varied vocabulary.
- repetition_rate measures chorus/hook density (0–1): higher = more repeated lines.

### Cypher syntax rules (important — do not violate):
- This is Neo4j Cypher, NOT SQL. Never use SELECT, FROM, WHERE as top-level clauses. Use MATCH, RETURN, WITH, WHERE.
- String values that contain double quotes MUST use single-quote delimiters:
  WRONG: {name: "The "Spring" Concerto"}
  RIGHT: {name: 'The "Spring" Concerto'}
- Cypher does NOT have GROUP BY. Aggregation is implicit: any non-aggregated variable in RETURN acts as the grouping key automatically. Never write GROUP BY.
- Use WITH to pass variables between clauses and to filter/order before a subsequent MATCH.
- LIMIT always goes at the end after ORDER BY.
- NEVER produce a cartesian product. Every MATCH clause must share at least one variable with another clause.
  WRONG (cartesian — no shared variable between t1 and t2):
    MATCH (t1:Track {language: 'en'})-[:BY]->(a1:Artist)
    MATCH (t2:Track {language: 'ru'})-[:BY]->(a2:Artist)
    RETURN a1.name, a2.name
  RIGHT (joined via shared topic node):
    MATCH (t1:Track {language: 'en'})-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track {language: 'ru'})
    MATCH (t1)-[:BY]->(a1:Artist), (t2)-[:BY]->(a2:Artist)
    RETURN t1.name, a1.name, t2.name, a2.name, tp.name AS shared_topic
  RIGHT (joined via shared artist):
    MATCH (a:Artist)<-[:BY]-(t1:Track {language: 'en'})
    MATCH (a)<-[:BY]-(t2:Track {language: 'ru'})
    RETURN a.name, count(DISTINCT t1) AS english_tracks, count(DISTINCT t2) AS russian_tracks

### Hints for taste/summary questions:
- To understand overall taste, use the WHOLE library (all tracks), not just playlist tracks.
- play_count is the strongest signal for preference — weight by it wherever possible.
- loved = true tracks are explicit favourites — always worth highlighting separately.
- For broad taste summaries, aggregate across genres and eras weighted by play_count:
  MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
  RETURN g.name AS genre, sum(t.play_count) AS total_plays, count(t) AS track_count
  ORDER BY total_plays DESC
- For top artists by engagement:
  MATCH (t:Track)-[:BY]->(a:Artist)
  RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count
  ORDER BY total_plays DESC LIMIT 20

### Graph traversal philosophy:
- This is a Knowledge Graph, not a relational database. Prefer returning specific nodes and paths
  that reveal interesting connections over aggregate counts.
- Think in traversals: follow edges across multiple hops to surface non-obvious relationships.
- A good KG query discovers WHICH specific songs/artists are connected and HOW, not just HOW MANY.
- Use aggregation (count, sum) only when the question explicitly asks for statistics or rankings.

### Example queries:
Q: What genres does Kanye West span?
MATCH (a:Artist {name:"Kanye West"})<-[:BY]-(t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count ORDER BY track_count DESC

Q: What are my most played tracks?
MATCH (t:Track) WHERE t.play_count > 0
RETURN t.name AS track, t.play_count AS plays ORDER BY plays DESC LIMIT 10

Q: Which artists feature most frequently in my library?
MATCH (t:Track)-[:FEATURES]->(a:Artist)
RETURN a.name AS featured_artist, count(t) AS appearances ORDER BY appearances DESC LIMIT 10

Q: Summarise my music taste
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WITH g.name AS genre, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 10
RETURN genre, total_plays, track_count

Q: Find me dark melancholic songs about death
MATCH (t:Track)-[:HAS_MOOD]->(m1:Mood {name:"dark"})
MATCH (t)-[:HAS_MOOD]->(m2:Mood {name:"melancholic"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic {name:"death"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist ORDER BY t.play_count DESC LIMIT 20

Q: Which artists mention New York most in their lyrics?
MATCH (t:Track)-[:MENTIONS_PLACE]->(pl:Place {name:"new york"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS track_count ORDER BY track_count DESC LIMIT 10

Q: What moods dominate my Hip-Hop tracks?
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name:"Hip-Hop/Rap"})
MATCH (t)-[:HAS_MOOD]->(m:Mood)
RETURN m.name AS mood, count(t) AS track_count ORDER BY track_count DESC

Q: Which tracks have the most varied vocabulary?
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.type_token_ratio IS NOT NULL
RETURN t.name AS track, a.name AS artist, t.type_token_ratio AS ttr
ORDER BY ttr DESC LIMIT 10

Q: Show me uplifting songs about friendship or love
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name:"uplifting"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic)
WHERE tp.name IN ["friendship", "love"]
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, collect(DISTINCT tp.name) AS topics
ORDER BY t.play_count DESC LIMIT 20

Q: What languages are in my library?
MATCH (t:Track)
WHERE t.language IS NOT NULL
RETURN t.language AS language, count(t) AS track_count ORDER BY track_count DESC

Q: Which albums contain songs about both love and death? (topic co-occurrence on album)
MATCH (al:Album)<-[:ON]-(t1:Track)-[:HAS_TOPIC]->(tp1:Topic {name: "love"})
MATCH (al)<-[:ON]-(t2:Track)-[:HAS_TOPIC]->(tp2:Topic {name: "death"})
MATCH (al)-[:BY]->(a:Artist)
RETURN al.title AS album, a.name AS artist
ORDER BY al.year DESC LIMIT 15

Q: Which artists appear both as a primary artist and as a featured guest in my library?
MATCH (a:Artist)<-[:BY]-(t1:Track)
MATCH (t2:Track)-[:FEATURES]->(a)
RETURN a.name AS artist, count(DISTINCT t1) AS own_tracks, count(DISTINCT t2) AS guest_appearances
ORDER BY guest_appearances DESC LIMIT 15

Q: Which of my playlists contain the most songs about money?
MATCH (t:Track)-[:HAS_TOPIC]->(tp:Topic {name: "money"})
MATCH (t)-[:IN_PLAYLIST]->(p:Playlist)
RETURN p.name AS playlist, count(t) AS money_tracks
ORDER BY money_tracks DESC LIMIT 10

Q: Find artists who collaborated with Drake and what moods their own songs have
MATCH (drake:Artist {name: "Drake"})<-[:BY]-(t1:Track)-[:FEATURES]->(collab:Artist)
MATCH (collab)<-[:BY]-(t2:Track)-[:HAS_MOOD]->(m:Mood)
RETURN collab.name AS collaborator, collect(DISTINCT m.name) AS moods, count(DISTINCT t2) AS track_count
ORDER BY track_count DESC LIMIT 15

Q: Find English and Russian songs that share a topic and mood (multi-hop traversal)
MATCH (t1:Track {language: 'en'})-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track {language: 'ru'})
MATCH (t1)-[:HAS_MOOD]->(m:Mood)<-[:HAS_MOOD]-(t2)
MATCH (t1)-[:BY]->(a1:Artist), (t2)-[:BY]->(a2:Artist)
RETURN a1.name AS english_artist, t1.name AS english_track,
       tp.name AS shared_topic, m.name AS shared_mood,
       t2.name AS russian_track, a2.name AS russian_artist
ORDER BY t1.play_count DESC LIMIT 20

Q: Which artists are connected through shared collaborators? (artist bridge traversal)
MATCH (a1:Artist)<-[:BY]-(t1:Track)-[:FEATURES]->(bridge:Artist)<-[:FEATURES]-(t2:Track)-[:BY]->(a2:Artist)
WHERE a1 <> a2 AND a1.name < a2.name
RETURN a1.name, bridge.name AS shared_collaborator, a2.name, count(*) AS strength
ORDER BY strength DESC LIMIT 15

Q: What topics connect artists across different genres? (cross-genre topic bridge)
MATCH (a1:Artist)<-[:BY]-(t1:Track)-[:IN_GENRE]->(g1:Genre)
MATCH (t1)-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track)-[:IN_GENRE]->(g2:Genre)
MATCH (t2)-[:BY]->(a2:Artist)
WHERE g1 <> g2 AND a1 <> a2
RETURN tp.name AS topic, g1.name AS genre1, a1.name AS artist1,
       g2.name AS genre2, a2.name AS artist2
ORDER BY t1.play_count DESC LIMIT 20

Now generate a Cypher query for the following question. Return ONLY the Cypher query, no explanation.
"""

# ── Self-healing retriever ────────────────────────────────────────────────────

class SelfHealingCypherRetriever:
    """
    Wraps TextToCypherRetriever with a retry loop.
    If Neo4j raises a CypherSyntaxError, the bad query + error are fed back
    to the LLM so it can fix and retry (up to MAX_RETRIES times).
    """
    MAX_RETRIES = 2

    def __init__(self, retriever, llm, graph_store, cypher_template):
        self._retriever    = retriever
        self._llm          = llm
        self._graph_store  = graph_store
        self._template     = cypher_template

    def retrieve(self, question: str):
        from neo4j.exceptions import CypherSyntaxError
        from llama_index.core.schema import QueryBundle
        from llama_index.core.prompts import PromptTemplate

        query_bundle = QueryBundle(query_str=question)
        last_error   = None
        last_cypher  = None

        for attempt in range(1 + self.MAX_RETRIES):
            if attempt == 0:
                nodes = self._retriever.retrieve_from_graph(query_bundle)
                # retrieve_from_graph catches errors internally and returns a node
                # with text — check the metadata for the query, then validate
                if nodes:
                    last_cypher = nodes[0].node.metadata.get("query")
                    # If execution succeeded, results are already in the node
                    # (structured_query errors bubble up, so reaching here = success)
                return nodes

            # Retry: ask the LLM to fix the bad Cypher
            fix_prompt = PromptTemplate(
                "The following Cypher query for a Neo4j database raised a syntax error.\n\n"
                "Bad Cypher:\n{bad_cypher}\n\n"
                "Error:\n{error}\n\n"
                "Schema:\n{schema}\n\n"
                "Please rewrite the Cypher to fix the error. "
                "Return ONLY the corrected Cypher query, no explanation.\n"
                "Original question: {question}"
            )
            schema = self._graph_store.get_schema_str()
            fixed  = self._llm.predict(
                fix_prompt,
                bad_cypher=last_cypher or "",
                error=str(last_error),
                schema=schema,
                question=question,
            )
            try:
                results  = self._graph_store.structured_query(fixed)
                last_cypher = fixed
                # Build a minimal response mirroring what TextToCypherRetriever returns
                from llama_index.core.schema import NodeWithScore, TextNode
                return [NodeWithScore(
                    node=TextNode(
                        text=str(results),
                        metadata={"query": fixed, "response": results},
                    ),
                    score=1.0,
                )]
            except CypherSyntaxError as e:
                last_error  = e
                last_cypher = fixed

        raise last_error


FIX_PROMPT = (
    "The Cypher query below raised a syntax error in Neo4j.\n\n"
    "Bad Cypher:\n{bad_cypher}\n\n"
    "Error:\n{error}\n\n"
    "Schema:\n{schema}\n\n"
    "Rewrite the Cypher to fix the error. "
    "Return ONLY the corrected Cypher, no explanation.\n"
    "Original question: {question}"
)


def _wrap_with_self_healing(retriever, llm, graph_store):
    """
    Patch both retrieve_from_graph (sync) and aretrieve_from_graph (async)
    — LlamaIndex uses the async path, but we cover both to be safe.
    """
    from neo4j.exceptions import CypherSyntaxError
    from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
    from llama_index.core.prompts import PromptTemplate

    fix_template = PromptTemplate(FIX_PROMPT)

    def _fix_cypher(bad_cypher, error, question):
        schema = graph_store.get_schema_str()
        fixed  = llm.predict(
            fix_template,
            bad_cypher=bad_cypher,
            error=str(error),
            schema=schema,
            question=question,
        )
        return fixed.strip().strip("```").strip()

    def _make_node(cypher, results):
        return [NodeWithScore(
            node=TextNode(text=str(results), metadata={"query": cypher, "response": results}),
            score=1.0,
        )]

    # ── sync patch ────────────────────────────────────────────────────────────
    orig_sync = retriever.retrieve_from_graph

    def sync_with_retry(query_bundle: QueryBundle):
        try:
            return orig_sync(query_bundle)
        except CypherSyntaxError as e:
            bad = getattr(e, "query", None) or str(e)
            for _ in range(2):
                fixed = _fix_cypher(bad, e, query_bundle.query_str)
                try:
                    return _make_node(fixed, graph_store.structured_query(fixed))
                except CypherSyntaxError as e2:
                    bad, e = fixed, e2
            raise e

    # ── async patch ───────────────────────────────────────────────────────────
    orig_async = retriever.aretrieve_from_graph

    async def async_with_retry(query_bundle: QueryBundle):
        try:
            return await orig_async(query_bundle)
        except CypherSyntaxError as e:
            bad = getattr(e, "query", None) or str(e)
            for _ in range(2):
                fixed = _fix_cypher(bad, e, query_bundle.query_str)
                try:
                    results = await graph_store.astructured_query(fixed)
                    return _make_node(fixed, results)
                except CypherSyntaxError as e2:
                    bad, e = fixed, e2
            raise e

    retriever.retrieve_from_graph  = sync_with_retry
    retriever.aretrieve_from_graph = async_with_retry
    return retriever


# ── Build engine ──────────────────────────────────────────────────────────────

def build_engine():
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.core import PropertyGraphIndex, Settings
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import TextToCypherRetriever
    from llama_index.core.prompts import PromptTemplate
    from llama_index.llms.openai_like import OpenAILike

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )

    print(f"Connecting to vLLM at {VLLM_BASE_URL} ...")
    llm = OpenAILike(
        model=VLLM_MODEL,
        api_base=VLLM_BASE_URL,
        api_key="not-needed",          # vLLM doesn't require a key
        is_chat_model=True,
        temperature=0.0,               # deterministic Cypher generation
        max_tokens=512,
        context_window=8192,
        additional_kwargs={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},  # skip <think> tokens for Cypher
        },
    )

    Settings.llm = llm
    Settings.embed_model = None        # no embeddings — pure Text-to-Cypher

    # Build index from existing graph store (no re-ingestion)
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        llm=llm,
        embed_kg_nodes=False,
    )

    cypher_template = PromptTemplate(
        "Schema:\n{schema}\n\n" + SCHEMA_PROMPT + "\nThe question is:\n{question}"
    )

    # Explicitly use TextToCypherRetriever with our schema prompt baked in
    cypher_retriever = TextToCypherRetriever(
        graph_store=graph_store,
        llm=llm,
        text_to_cypher_template=cypher_template,
        summarize_response=True,
        include_raw_response_as_metadata=True,
    )

    # Wrap with self-healing: retry on CypherSyntaxError, feeding error back to LLM
    cypher_retriever = _wrap_with_self_healing(cypher_retriever, llm, graph_store)

    query_engine = index.as_query_engine(
        sub_retrievers=[cypher_retriever],
        llm=llm,
    )

    return query_engine


# ── Agent ─────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "You are a music taste analyst for a personal Apple Music library stored as a Knowledge Graph. "
    "Answer the user's question by calling music_kg_query multiple times with specific, focused questions, "
    "then synthesise a comprehensive answer. Always query the graph — never answer from memory.\n\n"
    "The graph contains the following queryable dimensions:\n"
    "- Genres and play counts\n"
    "- Artists (primary and featured) and play counts\n"
    "- Albums and tracks (play count, skip count, loved, explicit)\n"
    "- Eras (Pre-90s, 90s, 2000s, 2010s, 2020s)\n"
    "- Playlists the user has curated\n"
    "- Moods expressed in lyrics (e.g. defiant, melancholic, aggressive, romantic)\n"
    "- Topics referenced in lyrics (e.g. love, death, ego, loyalty, drugs, money)\n"
    "- Places mentioned in lyrics\n"
    "- Language of tracks (ISO codes, e.g. en, fr, ru)\n"
    "- Lyric richness metrics (vocabulary variety, repetition rate)\n\n"
    "For broad taste summaries, always cover: genres, top artists, eras, moods, topics, "
    "language diversity, and loved/most-played tracks. "
    "Deliver the final answer directly — do not narrate your reasoning process.\n\n"
    "IMPORTANT RULES:\n"
    "- Always add LIMIT to every query — never fetch more than 60 records at a time.\n"
    "- Keep your responses concise — do not over-explain, do not repeat data you already have.\n"
    "- Make at most 6 targeted queries then synthesise — do not loop excessively.\n"
    "- Call render_graph at most once. You MUST call it if the user's message contains words like 'graph', 'visualize', 'visualise', 'render', 'show', or 'map'. Otherwise call it only if a visualisation genuinely adds value."
)

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "music_kg_query",
            "description": (
                "Query the Apple Music knowledge graph with a natural language question. "
                "Returns a summarised answer derived from the graph data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A specific natural language question about the music library",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_graph",
            "description": (
                "Optionally call this if a graph visualisation would meaningfully complement your answer — "
                "e.g. a network of artists, a genre cluster, a mood map. "
                "Do NOT call it for simple factual answers where a graph adds no value. "
                "The loop continues after this call — you must still give a final text answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Precise description of what the graph should show. "
                            "You MUST explicitly name: "
                            "(1) the anchor node type(s) to centre the graph on — e.g. Artist, Genre, Mood, Topic, Era, Playlist; "
                            "(2) which relationships to traverse — e.g. BY, FEATURES, IN_GENRE, HAS_MOOD, HAS_TOPIC, IN_ERA, MENTIONS_PLACE; "
                            "(3) any secondary node types to expand into — e.g. Album, Track, Genre, Mood, Topic. "
                            "IMPORTANT: if you already know the specific names from your prior queries, list them explicitly "
                            "so the graph anchors on exactly those nodes — do NOT ask the graph to re-derive them. "
                            "Example: 'Anchor on these specific tracks: [Sunflower, Ric Flair Drip, Roses]. "
                            "Expand via FEATURES to their collaborating Artist nodes.' "
                            "The more specific you are, the better the visualisation."
                        ),
                    }
                },
                "required": ["description"],
            },
        },
    },
]

VIZ_WORKER_PROMPT = SCHEMA_PROMPT.replace(
    "Now generate a Cypher query for the following question. Return ONLY the Cypher query, no explanation.",
    """Now generate a Cypher query for neovis.js visualisation based on the description below.

### Technical rules — must follow all of them:
- RETURN only node and relationship variables — NEVER scalars, aliases, or property accesses in RETURN.
  WRONG: RETURN t.name, r, f.name, feat_count
  RIGHT:  RETURN t, r, f
- Do NOT mix node variables with scalar expressions in RETURN — neovis.js will silently produce an empty graph.
- Every variable used in RETURN must be explicitly bound in a MATCH or OPTIONAL MATCH clause. Never reference a variable in RETURN that was not assigned — e.g. MATCH (a)<-[r1:BY]-(t) not MATCH (a)<-[:BY]-(t) if you intend to RETURN r1
- Once a WITH clause is used, only the listed variables remain in scope. Never reference a variable that was consumed in a previous WITH.
  WRONG: WITH t, count(f) AS cnt  /  WITH t, collect(f) AS feats   ← f is out of scope after first WITH
  RIGHT:  WITH t, count(f) AS cnt, collect(f) AS feats              ← collect in the same WITH where f is still in scope
- Use OPTIONAL MATCH for all secondary patterns to avoid dropping anchor nodes
- Limit anchor nodes to 5–10 with WITH ... LIMIT before expanding
- After anchoring, limit track expansion to the top 3–5 tracks per anchor node ordered by play_count DESC — never expand to all tracks of an artist
- Keep the total number of returned rows under 80 — add a final LIMIT if necessary
- Prefer the collect() anchor pattern:
    MATCH (a:Artist)<-[:BY]-(t:Track)
    WITH a, count(t) AS tc ORDER BY tc DESC LIMIT 10
    WITH collect(a.name) AS top
    MATCH (a:Artist)<-[r1:BY]-(t:Track) WHERE a.name IN top
    WITH a, r1, t ORDER BY t.play_count DESC
    WITH a, collect({r:r1, t:t})[..4] AS top_tracks
    UNWIND top_tracks AS row
    WITH a, row.r AS r1, row.t AS t
    OPTIONAL MATCH (t)-[r2:ON]->(al:Album)
    OPTIONAL MATCH (t)-[r3:IN_GENRE]->(g:Genre)
    RETURN a, r1, t, r2, al, r3, g
- For "top-N then expand" patterns, collect names in a WITH clause THEN re-MATCH using that list:
    MATCH (t:Track)-[r:FEATURES]->(f:Artist)
    WITH t, count(f) AS feat_count, collect(f.name) AS feat_names
    ORDER BY feat_count DESC LIMIT 10
    WITH collect(t.name) AS top_tracks
    MATCH (t:Track)-[r:FEATURES]->(f:Artist) WHERE t.name IN top_tracks
    RETURN t, r, f
- If the description lists specific names (e.g. "Anchor on these tracks: [Sunflower, Ric Flair Drip]"),
  use them directly as a hardcoded IN list — do NOT re-derive via aggregation:
    MATCH (t:Track)-[r:FEATURES]->(f:Artist)
    WHERE t.name IN ['Sunflower', 'Ric Flair Drip']
    RETURN t, r, f
- Return ONLY the Cypher query, no explanation, no markdown fences.

### FEATURES relationship — critical:
- FEATURES connects a Track to a featured Artist: (:Track)-[:FEATURES]->(:Artist)
- There is NO (:Artist)-[:FEATURES]->(:Artist) relationship — never write it
- To show collaborations between artists, always go through the Track:
    WRONG: MATCH (a:Artist)-[r:FEATURES]->(f:Artist)
    RIGHT: MATCH (t:Track)-[rb:BY]->(a:Artist), (t)-[rf:FEATURES]->(f:Artist) RETURN t, rb, a, rf, f

### Visual richness — aim for these:
- Include at least 2–3 relationship types in the same query
- Mix node types (Artist + Track + Genre + Mood + Topic) — a graph with only one node type is boring
- FEATURES relationships (collaborations between artists) are visually striking — include whenever relevant
- Mood and Topic nodes create natural clusters — always include for artist or taste queries
- Era nodes group tracks by decade — useful for any temporal or library-wide query

### Visually impressive patterns — use as inspiration:
- Artist collaboration web:
    anchor on top artists via BY, expand with FEATURES to pull in collaborators and their genres
- Artist taste profile:
    anchor on one artist via BY, expand with IN_GENRE + HAS_MOOD + HAS_TOPIC on a sample of their tracks
- Genre landscape:
    anchor on Genre nodes, expand with BY to show top artists per genre and their shared tracks
- Mood/topic cluster:
    anchor on Mood or Topic nodes, expand with HAS_MOOD/HAS_TOPIC + BY to show artists and tracks that share them
- Era snapshot:
    anchor on Era nodes via IN_ERA, expand with BY + IN_GENRE to show which artists and genres define each decade"""
)


def _parse_cypher(raw: str) -> str:
    """Strip markdown fences from a raw LLM Cypher response."""
    cypher = raw.strip()
    if cypher.startswith("```"):
        cypher = cypher.split("```")[1]
        if cypher.lower().startswith("cypher"):
            cypher = cypher[6:]
    return cypher.strip()


def _has_graph_return(cypher: str) -> bool:
    """
    Returns True if the RETURN clause contains only bare node/relationship
    variables — no property accesses (t.name) or function calls (count()).
    neovis.js silently produces an empty graph if scalars are returned.
    """
    import re
    match = re.search(
        r'\bRETURN\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|\bSKIP\b|$)',
        cypher, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    return_clause = match.group(1)
    # Property access (t.name) or function calls (count(...)) → scalars
    return not re.search(r'\w+\.\w+|\w+\s*\(', return_clause)


def build_agent(engine=None, max_steps: int = 8):
    from openai import OpenAI as OpenAIClient

    if engine is None:
        engine = build_engine()

    client = OpenAIClient(base_url=VLLM_BASE_URL, api_key="not-needed")

    _VIZ_USER_SUFFIX = (
        "\n\nCRITICAL: your RETURN clause must contain ONLY bare node and relationship "
        "variables (e.g. RETURN t, r, a, g). No property accesses (t.name), no function "
        "calls (count(), collect()), no aliases. neovis.js will produce an empty graph otherwise."
    )

    def viz_worker(description: str) -> str:
        user_msg = f"Generate a neovis.js visualisation Cypher for: {description}{_VIZ_USER_SUFFIX}"
        messages = [
            {"role": "system", "content": VIZ_WORKER_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        _llm_kwargs = dict(
            model=VLLM_MODEL,
            temperature=0.0,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        response = client.chat.completions.create(messages=messages, **_llm_kwargs)
        cypher = _parse_cypher(response.choices[0].message.content)
        _viz_tokens = response.usage.completion_tokens if response.usage else '?'
        print(f"[viz_worker attempt=1] tokens={_viz_tokens} cypher={repr(cypher[:400])}", flush=True)

        if not _has_graph_return(cypher):
            print("[viz_worker] RETURN contains scalars — retrying with correction", flush=True)
            messages.append({"role": "assistant", "content": cypher})
            messages.append({
                "role": "user",
                "content": (
                    "WRONG — your RETURN clause contains property accesses or function calls "
                    "(e.g. t.name, count(), collect()). neovis.js cannot render scalars. "
                    "Rewrite the query so RETURN contains ONLY bare node and relationship "
                    "variables — nothing else. Example: RETURN t, r, a, g"
                ),
            })
            response = client.chat.completions.create(messages=messages, **_llm_kwargs)
            cypher = _parse_cypher(response.choices[0].message.content)
            _viz_tokens = response.usage.completion_tokens if response.usage else '?'
            print(f"[viz_worker attempt=2] tokens={_viz_tokens} cypher={repr(cypher[:400])}", flush=True)

        return cypher

    def run_agent(question: str, on_step=None):
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        steps = []
        pending_viz = None   # {"cypher": ..., "description": ...} if agent calls render_graph

        while True:
            # If we've hit the step cap, stop offering tools and force a final answer
            tools_param = AGENT_TOOLS if len(steps) < max_steps else None
            if tools_param is None and len(steps) >= max_steps:
                messages.append({
                    "role": "user",
                    "content": "You have gathered enough data. Now synthesise a comprehensive answer based on everything you have found so far.",
                })

            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                **({"tools": tools_param} if tools_param else {}),
                temperature=0.0,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = response.choices[0].message
            _usage = response.usage.completion_tokens if response.usage else '?'
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[agent step={len(steps)}] tool={tc.function.name} args={tc.function.arguments[:200]} tokens={_usage}", flush=True)
            else:
                print(f"[agent step={len(steps)}] final answer tokens={_usage} preview={repr((msg.content or '')[:300])}", flush=True)

            # Serialise assistant message explicitly for vLLM compatibility
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                return msg.content, steps, pending_viz

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)

                if call.function.name == "render_graph":
                    description = args["description"]
                    pending_viz = {
                        "cypher":      viz_worker(description),
                        "description": description,
                    }
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Graph queued for rendering.",
                    })

                elif call.function.name == "music_kg_query":
                    result = str(engine.query(args["question"]))
                    step = {"question": args["question"], "result": result}
                    steps.append(step)
                    if on_step:
                        on_step(step)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })

    return run_agent


# ── Run ───────────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "What genres does Kanye West span?",
    "What are my top 10 most played tracks?",
    "Which artists feature most on my tracks?",
    "How many tracks do I have per era?",
]

def run(question: str):
    engine = build_engine()
    print(f"\nQuestion: {question}")
    print("-" * 60)
    response = engine.query(question)
    print(f"Answer:\n{response}")
    if hasattr(response, "metadata") and response.metadata:
        cypher = response.metadata.get("query")
        if cypher:
            print(f"\nCypher used:\n{cypher}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", default=None)
    args = parser.parse_args()

    if args.question:
        run(args.question)
    else:
        for q in DEFAULT_QUESTIONS:
            run(q)
            print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
