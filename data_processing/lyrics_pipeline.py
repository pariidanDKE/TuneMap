"""
lyrics_pipeline.py

Fetches lyrics for every track in library.json via LRCLIB, then fires 3
concurrent LLM calls (moods, places, topics) per track to a local vLLM instance.

Results are saved to Data/track_themes.json keyed by Track ID.
The script is resumable — already-processed tracks are skipped on re-run.

Run:
  python data_processing/lyrics_pipeline.py
"""

import asyncio
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import aiohttp
from langdetect import detect, LangDetectException

# ── Defaults (used by main() CLI only) ────────────────────────────────────────

_DEFAULT_LRCLIB_SEARCH = "https://lrclib.net/api/search"
_DEFAULT_LLM_URL       = "http://localhost:8000/v1/chat/completions"
_DEFAULT_LLM_MODEL     = "qwen3.5-9b-awq"

# ── Fixed vocabularies ────────────────────────────────────────────────────────

MOODS = [
    "melancholic", "aggressive", "uplifting", "introspective", "boastful",
    "celebratory", "dark", "nostalgic", "anxious", "romantic", "defiant",
    "vulnerable", "cynical",
]
MOODS_SET = set(MOODS)

BASE_TOPICS = [
    "God", "Spirituality", "Mother", "Father", "Death", "Love", "Sex",
    "Desire", "Friendship", "Loyalty", "Betrayal", "Money", "Fame",
    "Freedom", "Struggle", "Fun", "Heartbreak", "Resentment", "Redemption",
    "Ego", "Nostalgia", "Dreams", "Jealousy", "Isolation", "Insecurity",
    "Mental Health", "Materialism", "Pleasure", "Moving On",
]

GENRE_TOPIC_EXTRAS: dict[str, list[str]] = {
    "hip-hop":    ["Gang", "Drugs", "Prison", "Police", "Hustle", "Government"],
    "rock":       ["Rebellion", "Anger", "War", "Identity", "Alienation"],
    "metal":      ["Rebellion", "Anger", "War", "Identity", "Death"],
    "pop":        ["Toxic Relationship", "Confidence", "Youth"],
    "r&b":        ["Toxic Relationship", "Confidence", "Youth", "Passion"],
    "electronic": ["Technology", "Escapism"],
    "country":    ["Hometown", "Hardship", "Nature"],
    "jazz":       ["Freedom", "Improvisation"],
}

GENRE_GROUP: dict[str, str] = {
    "hip-hop/rap":             "hip-hop",
    "rap":                     "hip-hop",
    "hip-hop":                 "hip-hop",
    "old school rap":          "hip-hop",
    "hardcore rap":            "hip-hop",
    "alternative rap":         "hip-hop",
    "dirty south":             "hip-hop",
    "latin rap":               "hip-hop",
    "south african hip-hop":   "hip-hop",
    "hip-hop in russian":      "hip-hop",
    "rock":                    "rock",
    "alternative":             "rock",
    "hard rock":               "rock",
    "indie rock":              "rock",
    "folk-rock":               "rock",
    "pop/rock":                "rock",
    "metal":                   "metal",
    "hardcore":                "metal",
    "death metal/black metal": "metal",
    "industrial":              "metal",
    "pop":                     "pop",
    "french pop":              "pop",
    "j-pop":                   "pop",
    "k-pop":                   "pop",
    "indie pop":               "pop",
    "vocal pop":               "pop",
    "adult contemporary":      "pop",
    "pop latino":              "pop",
    "urbano latino":           "pop",
    "r&b/soul":                "r&b",
    "soul":                    "r&b",
    "contemporary r&b":        "r&b",
    "electronic":              "electronic",
    "dance":                   "electronic",
    "house":                   "electronic",
    "techno":                  "electronic",
    "electronica":             "electronic",
    "jungle/drum'n'bass":      "electronic",
    "country":                 "country",
    "americana":               "country",
    "traditional folk":        "country",
    "alternative folk":        "country",
    "jazz":                    "jazz",
    "blues":                   "jazz",
}

PLACE_ROAD_WORDS = {
    "road", "street", "avenue", "ave", "boulevard", "blvd", "lane", "drive",
    "dr", "way", "court", "ct", "place", "pl", "terrace", "alley", "highway",
    "hwy", "freeway", "parkway", "pkwy", "side",
}


def topics_for_genre(genre: str) -> list[str]:
    group  = GENRE_GROUP.get(genre.lower())
    extras = GENRE_TOPIC_EXTRAS.get(group, []) if group else []
    seen, result = set(), []
    for t in BASE_TOPICS + extras:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _is_road(place: str) -> bool:
    return bool(set(place.lower().split()) & PLACE_ROAD_WORDS)


# ── Async LLM helper ──────────────────────────────────────────────────────────

async def llm_call(session: aiohttp.ClientSession, system: str, user: str, llm_url: str, llm_model: str) -> dict | None:
    payload = {
        "model":           llm_model,
        "messages":        [{"role": "system", "content": system},
                            {"role": "user",   "content": user}],
        "temperature":     0.1,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with session.post(llm_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as r:
            data = await r.json()
            return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"    LLM call failed: {e}")
        return None


# ── Three extraction coroutines ───────────────────────────────────────────────

async def extract_moods(session: aiohttp.ClientSession, name: str, artist: str, lyrics: str, llm_url: str, llm_model: str) -> list[str]:
    system = (
        "Extract moods from the song lyrics. "
        "Return JSON: {\"moods\": [...]}. "
        "Pick ONLY from this exact list, nothing else: "
        + ", ".join(MOODS)
    )
    result = await llm_call(session, system, f"Song: {name} by {artist}\n\nLyrics:\n{lyrics}", llm_url, llm_model)
    values = [v.lower() for v in result.get("moods", [])] if result else []
    return [v for v in values if v in MOODS_SET]


async def extract_places(session: aiohttp.ClientSession, name: str, artist: str, lyrics: str, llm_url: str, llm_model: str) -> list[str]:
    system = (
        "Extract real-world cities, countries, and regions mentioned in the lyrics. "
        "Valid examples: 'Toronto', 'Los Angeles', 'Brooklyn', 'France', 'the Bronx'. "
        "Do NOT include: streets, roads, avenues, boulevards, highways, or any address-like location. "
        "Do NOT include people's names, events, or abstract concepts. "
        "If unsure whether something is a city/country/region, leave it out. "
        "Return JSON: {\"places\": [...]}"
    )
    result = await llm_call(session, system, f"Song: {name} by {artist}\n\nLyrics:\n{lyrics}", llm_url, llm_model)
    values = [v.lower() for v in result.get("places", [])] if result else []
    return [v for v in values if not _is_road(v)]


async def extract_topics(session: aiohttp.ClientSession, name: str, artist: str, genre: str, lyrics: str, llm_url: str, llm_model: str) -> list[str]:
    allowed     = topics_for_genre(genre)
    allowed_set = {t.lower() for t in allowed}
    system = (
        "Extract concrete topics and subjects referenced in the lyrics. "
        "Use ONLY entries from this list, do not add anything outside it: "
        + ", ".join(allowed)
        + ". Return JSON: {\"topics\": [...]}"
    )
    result = await llm_call(session, system, f"Song: {name} by {artist}\n\nLyrics:\n{lyrics}", llm_url, llm_model)
    values = [v.lower() for v in result.get("topics", [])] if result else []
    return [v for v in values if v in allowed_set]


# ── Deterministic lyrics features ────────────────────────────────────────────

def lyrics_features(lyrics: str) -> dict:
    words       = re.findall(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF]+", lyrics.lower())
    total_words  = len(words)
    unique_words = len(set(words))
    ttr          = round(unique_words / total_words, 4) if total_words else 0.0

    lines        = [l.strip() for l in lyrics.splitlines() if l.strip()]
    total_lines  = len(lines)
    repeated     = sum(c for c in Counter(lines).values() if c > 1)
    repetition   = round(repeated / total_lines, 4) if total_lines else 0.0

    try:
        language = detect(lyrics)
    except LangDetectException:
        language = "unknown"

    return {
        "language":         language,
        "total_words":      total_words,
        "unique_words":     unique_words,
        "type_token_ratio": ttr,
        "repetition_rate":  repetition,
    }


# ── LRCLIB (sync, offloaded to executor) ─────────────────────────────────────

def fetch_lyrics_sync(track_name: str, artist: str, lrclib_url: str = _DEFAULT_LRCLIB_SEARCH) -> str | None:
    params = urllib.parse.urlencode({"q": track_name, "artistName": artist})
    req    = urllib.request.Request(
        f"{lrclib_url}?{params}",
        headers={"User-Agent": "AppleMusicKG/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            results = json.loads(r.read().decode())
        for r in results:
            if (
                r.get("plainLyrics")
                and not r.get("instrumental", False)
                and r.get("trackName", "").lower() == track_name.lower()
                and artist.lower() in r.get("artistName", "").lower()
            ):
                return r["plainLyrics"]
    except Exception as e:
        print(f"    LRCLIB error: {e}")
    return None


# ── Callable run() for use from app.py ───────────────────────────────────────

async def run(
    library: dict,
    output_path: Path,
    llm_url: str,
    llm_model: str,
    lrclib_url: str = _DEFAULT_LRCLIB_SEARCH,
    on_progress: Callable[[int, int, int, str], None] | None = None,
):
    """
    Run the full lyrics enrichment pipeline on a parsed library dict.
    Results are saved incrementally to output_path (resumable).
    on_progress(current, total, found, track_name) called after each track.
    """
    tracks = library["tracks"]

    results: dict = {}
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            pass

    already_done = set(results.keys())
    pending      = [t for t in tracks if str(t.get("Track ID")) not in already_done]
    total        = len(tracks)
    found        = sum(1 for v in results.values() if v.get("lyrics_found"))

    loop = asyncio.get_event_loop()

    async with aiohttp.ClientSession() as session:
        processed = len(already_done)
        for t in pending:
            track_id = str(t.get("Track ID"))
            name     = t.get("Name", "")
            artist   = t.get("Album Artist") or t.get("Artist", "")
            genre    = t.get("Genre", "Unknown")

            lyrics = await loop.run_in_executor(
                None, fetch_lyrics_sync, name, artist, lrclib_url
            )

            if not lyrics:
                results[track_id] = {
                    "track_id": int(track_id), "name": name, "artist": artist,
                    "genre": genre, "lyrics_found": False,
                    "language": None, "total_words": None, "unique_words": None,
                    "type_token_ratio": None, "repetition_rate": None,
                    "moods": [], "places": [], "topics": [],
                }
            else:
                features, (moods, places, topics) = await asyncio.gather(
                    loop.run_in_executor(None, lyrics_features, lyrics),
                    asyncio.gather(
                        extract_moods(session, name, artist, lyrics, llm_url, llm_model),
                        extract_places(session, name, artist, lyrics, llm_url, llm_model),
                        extract_topics(session, name, artist, genre, lyrics, llm_url, llm_model),
                    ),
                )
                found += 1
                results[track_id] = {
                    "track_id": int(track_id), "name": name, "artist": artist,
                    "genre": genre, "lyrics_found": True,
                    **features,
                    "moods": moods, "places": places, "topics": topics,
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            processed += 1
            if on_progress:
                on_progress(processed, total, found, name)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    library_path = Path("Data/library.json")
    output_path  = Path("Data/track_themes.json")

    with open(library_path, encoding="utf-8") as f:
        library = json.load(f)

    def on_progress(current, total, found, name):
        print(f"[{current}/{total}] {name} (found: {found})")

    await run(
        library, output_path,
        llm_url=_DEFAULT_LLM_URL,
        llm_model=_DEFAULT_LLM_MODEL,
        on_progress=on_progress,
    )
    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
