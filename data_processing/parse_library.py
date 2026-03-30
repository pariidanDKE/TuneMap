"""
parse_library.py

Parses an Apple Music Library.xml (plist format) and outputs:
  - library.json  : full structured data (tracks + playlists)
  - overview.json : summary stats (counts, genres, artists, years, etc.)
"""

import argparse
import datetime
import json
import plistlib
from collections import Counter
from pathlib import Path


def load_plist(path: Path) -> dict:
    with open(path, "rb") as f:
        return plistlib.load(f)


def serialize_value(v):
    """Convert non-JSON-serializable plist values."""
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    return v


def parse_tracks(raw: dict) -> list[dict]:
    tracks = []
    for t in raw.values():
        tracks.append({k: serialize_value(v) for k, v in t.items()})
    return tracks


def parse_playlists(raw: list) -> list[dict]:
    playlists = []
    for p in raw:
        pl = {}
        for k, v in p.items():
            if k == "Playlist Items":
                pl[k] = [item.get("Track ID") for item in v]
            else:
                pl[k] = serialize_value(v)
        playlists.append(pl)
    return playlists


def build_overview(tracks: list[dict], playlists: list[dict]) -> dict:
    genres = Counter(t.get("Genre", "Unknown") for t in tracks)
    artists = Counter(t.get("Artist", "Unknown") for t in tracks)
    albums = Counter(t.get("Album", "Unknown") for t in tracks)
    years = Counter(t.get("Year") for t in tracks if t.get("Year"))
    kinds = Counter(t.get("Kind", "Unknown") for t in tracks)

    total_ms = sum(t.get("Total Time", 0) for t in tracks)
    total_hours = round(total_ms / 1000 / 3600, 2)

    explicit_count = sum(1 for t in tracks if t.get("Explicit"))
    apple_music_count = sum(1 for t in tracks if t.get("Apple Music"))

    all_fields = sorted({k for t in tracks for k in t})

    playlist_names = [p.get("Name", "Unnamed") for p in playlists if not p.get("Master")]

    return {
        "totals": {
            "tracks": len(tracks),
            "playlists": len(playlist_names),
            "unique_artists": len(artists),
            "unique_albums": len(albums),
            "unique_genres": len(genres),
            "total_playtime_hours": total_hours,
            "explicit_tracks": explicit_count,
            "apple_music_tracks": apple_music_count,
        },
        "top_genres": dict(genres.most_common(20)),
        "top_artists": dict(artists.most_common(30)),
        "top_albums": dict(albums.most_common(20)),
        "tracks_by_year": dict(sorted(years.items())),
        "file_kinds": dict(kinds.most_common()),
        "playlist_names": sorted(playlist_names),
        "available_track_fields": all_fields,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse Apple Music Library.xml")
    parser.add_argument(
        "--input",
        default="Data/Library.xml",
        help="Path to Library.xml (default: Data/Library.xml)",
    )
    parser.add_argument(
        "--output-dir",
        default="Data",
        help="Directory to write output files (default: Data)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    plist = load_plist(input_path)

    tracks = parse_tracks(plist.get("Tracks", {}))
    playlists = parse_playlists(plist.get("Playlists", []))

    library = {
        "library_info": {
            "date": serialize_value(plist.get("Date")),
            "application_version": plist.get("Application Version"),
            "music_folder": plist.get("Music Folder"),
        },
        "tracks": tracks,
        "playlists": playlists,
    }

    overview = build_overview(tracks, playlists)

    library_out = output_dir / "library.json"
    overview_out = output_dir / "overview.json"

    with open(library_out, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"Written: {library_out}")

    with open(overview_out, "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print(f"Written: {overview_out}")

    # Print summary to stdout
    t = overview["totals"]
    print(f"\n=== Library Overview ===")
    print(f"  Tracks          : {t['tracks']}")
    print(f"  Playlists       : {t['playlists']}")
    print(f"  Unique artists  : {t['unique_artists']}")
    print(f"  Unique albums   : {t['unique_albums']}")
    print(f"  Unique genres   : {t['unique_genres']}")
    print(f"  Total playtime  : {t['total_playtime_hours']} hours")
    print(f"  Explicit tracks : {t['explicit_tracks']}")

    print(f"\nTop 10 Genres:")
    for g, c in list(overview["top_genres"].items())[:10]:
        print(f"  {g}: {c}")

    print(f"\nTop 10 Artists:")
    for a, c in list(overview["top_artists"].items())[:10]:
        print(f"  {a}: {c}")

    print(f"\nAll {len(overview['available_track_fields'])} available track fields:")
    for f in overview["available_track_fields"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
