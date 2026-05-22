import os
import time
import pandas as pd
import requests


# CONFIGURATION

API_KEY        = "paste your TMDB v3 API key here"
IMAGE_SIZE     = "w342"                    # TMDB image size (w92/w185/w342/w500/w780/original)
POSTER_DIR     = os.path.join("static", "posters")
CSV_PATH       = os.path.join("data", "tmdb_5000_movies.csv")
REQUEST_DELAY  = 0.25                      # seconds between requests (4 req/s — well under 40/s limit)
REQUEST_TIMEOUT = 10                       # seconds before a request is considered failed

# TMDB base URLs
TMDB_API_BASE   = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = f"https://image.tmdb.org/t/p/{IMAGE_SIZE}"


def ensure_poster_directory(path: str) -> None:
    """Create the poster directory (and any parent dirs) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def fetch_poster_path(movie_id: int, session: requests.Session) -> str | None:
    """
    Query the TMDB /movie/{id} endpoint and return the poster_path string,
    or None if the movie has no poster or the request fails.
    """
    url = f"{TMDB_API_BASE}/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}

    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("poster_path")

    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP ERROR] Movie {movie_id}: {e}")
    except requests.exceptions.ConnectionError:
        print(f"  [CONNECTION ERROR] Movie {movie_id}: could not reach TMDB API")
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] Movie {movie_id}: request timed out after {REQUEST_TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        print(f"  [REQUEST ERROR] Movie {movie_id}: {e}")

    return None


def download_image(poster_path: str, dest_path: str, session: requests.Session) -> bool:
    """
    Download the poster image from TMDB and write it to `dest_path`.
    Returns True on success, False on any error.
    """
    url = f"{TMDB_IMAGE_BASE}{poster_path}"

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except requests.exceptions.RequestException as e:
        print(f"  [DOWNLOAD ERROR] {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_all_posters() -> None:
    """
    Main entry point.  Reads the CSV, iterates every movie ID, and
    downloads the poster unless one already exists locally.
    """

    if API_KEY == "YOUR_TMDB_API_KEY":
        raise ValueError(
            "Please set your TMDB API key in the API_KEY constant at the top of this file."
        )

    ensure_poster_directory(POSTER_DIR)
    print(f"Posters will be saved to: {os.path.abspath(POSTER_DIR)}\n")

    try:
        df = pd.read_csv(CSV_PATH, usecols=["id"])
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    except ValueError as e:
        raise ValueError(f"CSV is missing the 'id' column: {e}")

    movie_ids = df["id"].dropna().astype(int).unique().tolist()
    total = len(movie_ids)
    print(f"Found {total} unique movie IDs in {CSV_PATH}\n")
    print("─" * 60)

    downloaded = 0
    skipped    = 0
    failed     = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Flask-Movie-Recommender/1.0"})

        for index, movie_id in enumerate(movie_ids, start=1):
            dest_path = os.path.join(POSTER_DIR, f"{movie_id}.jpg")

            if os.path.exists(dest_path):
                print(f"[{index:>4}/{total}] SKIP   {movie_id} (already exists)")
                skipped += 1
                continue

            print(f"[{index:>4}/{total}] FETCH  {movie_id} ...", end=" ", flush=True)

            poster_path = fetch_poster_path(movie_id, session)

            if not poster_path:
                print("NO POSTER")
                failed += 1
                time.sleep(REQUEST_DELAY)
                continue

            success = download_image(poster_path, dest_path, session)

            if success:
                print(f"OK  →  {dest_path}")
                downloaded += 1
            else:
                failed += 1

            time.sleep(REQUEST_DELAY)

    #Summary
    print("\n" + "─" * 60)
    print("Download complete!")
    print(f"  ✓  Downloaded : {downloaded}")
    print(f"  ⏭  Skipped    : {skipped}  (already on disk)")
    print(f"  ✗  Failed     : {failed}   (no poster or API error)")
    print(f"  ─  Total      : {total}")
    print("─" * 60)



if __name__ == "__main__":
    download_all_posters()
