# 🎬 CineMatch AI

<div align="center">

![CineMatch AI Banner](https://img.shields.io/badge/CineMatch-AI-e50914?style=for-the-badge&logo=film&logoColor=white)

**An AI-powered cinematic movie recommendation platform with a Netflix-inspired dark UI**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![TMDB](https://img.shields.io/badge/TMDB-Dataset-01B4E4?style=flat-square&logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Features](#-features) · [ML Architecture](#-machine-learning-architecture) · [Tech Stack](#-tech-stack) · [Installation](#-local-installation) · [Project Structure](#-project-structure) · [Screenshots](#-screenshots)

</div>

---

## 🌟 Introduction

**CineMatch AI** is a full-stack movie recommendation web application that uses machine learning to suggest films you'll actually love. It combines **Content-Based Filtering**, **TF-IDF Vectorization**, and **Cosine Similarity** to analyze movie metadata — genres, cast, director, keywords, plot, and more — delivering smart, context-aware recommendations in milliseconds.

Built on the **TMDB 5000 Movie Dataset** and served via a **Flask REST API**, the app pairs a powerful ML backend with a cinematic, Netflix-inspired dark UI crafted in **Tailwind CSS** and vanilla JavaScript. Features include fuzzy search with autocomplete, genre-based discovery, trending movies, director filmographies, movie comparison, and advanced multi-filter search — all in a fully responsive interface.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 **Smart Search** | Autocomplete with fuzzy matching — finds movies even with typos |
| 🤖 **AI Recommendations** | Hybrid content + numeric ML model with a per-result match score |
| 🎭 **Genre Discovery** | Browse top-rated movies by any genre |
| 📈 **Trending Movies** | Bayesian-weighted ranking surfaces the genuinely best films |
| 🎲 **Random Movie** | Discover something new with a single click |
| 🎬 **Director Filmography** | Explore every film by a specific director |
| 🖼️ **TMDB Posters** | Batch-downloaded poster images for every movie |
| ⚡ **Instant Results** | Pre-computed similarity matrix means zero wait time |
| ⚡ **Compare Movies** | Side-by-side complete film analysis tool |

---

## 🧠 Machine Learning Architecture

CineMatch AI uses a **hybrid two-arm recommendation model** that blends rich content signals with numeric production-profile signals.

### Arm 1 — Content-Based Filtering (70%)

Movie metadata is merged into a single weighted text field. Features are repeated to reflect their relative importance before TF-IDF vectorization:

| Signal | Weight | Example |
|---|---|---|
| Genres | 4× | `Action_Adventure_SciFi` |
| Keywords | 3× | `space_opera dystopia rebellion` |
| Director | 3× | `Christopher_Nolan` |
| Top 5 Cast | 3× | `Tom_Hanks Meryl_Streep` |
| Overview / Plot | 2× | full plot synopsis |
| Tagline | 1× | marketing tagline |

A **TF-IDF Vectorizer** (30,000 features, 1–2 ngrams, sublinear TF scaling) converts the combined field into sparse vectors. **Cosine Similarity** is then computed across all ~4,800 movie pairs.

### Arm 2 — Numeric Feature Similarity (30%)

Five production-level features provide a complementary signal:

> Budget · Revenue · Runtime · Popularity · Vote Average

All values are **MinMax scaled** and compared via cosine similarity, surfacing films with comparable production scale and audience reception.

### Hybrid Blending

```
hybrid_score = 0.70 × content_similarity + 0.30 × numeric_similarity
```

### Final Re-Ranking

Before results are returned, each candidate score is further adjusted:

- **Year Proximity (Gaussian Decay)** — films closer in release year receive a boost (σ = 15 years), keeping recommendations era-coherent.
- **Popularity Boost** — a subtle +5% nudge for popular titles prevents obscure-but-similar films from dominating results.
- **Vote Threshold** — films with fewer than 10 votes are excluded entirely.

### Bayesian Weighted Rating

Trending and genre rankings use a **Bayesian average** instead of raw `vote_average`, so acclaimed niche films rank above merely popular but mediocre ones:

```
bayesian_score = (v / (v + m)) × R + (m / (v + m)) × C
```

`v` = vote count · `m` = 60th-percentile vote count · `R` = film rating · `C` = dataset mean rating

### Fuzzy Title Matching

Search and lookup use **RapidFuzz** token set ratio (≥ 60% threshold), so users can find films with approximate or misspelled titles via a three-tier strategy: exact match → substring match → fuzzy match.

---

## 🛠️ Tech Stack

**Backend**
- Python 3.9+, Flask, Flask-CORS
- Scikit-learn — TF-IDF, Cosine Similarity, MinMaxScaler
- Pandas, NumPy
- RapidFuzz — fuzzy string matching
- Pickle — model serialization

**Frontend**
- HTML5, vanilla JavaScript
- Tailwind CSS (CDN, dark mode)
- Google Fonts: Bebas Neue · Barlow Condensed · DM Sans
- Material Symbols icons

**Data & APIs**
- TMDB 5000 Movie Dataset (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)
- TMDB REST API (poster image downloads)

---

## 📋 Prerequisites

- Python 3.9 or higher
- pip
- A free [TMDB API key](https://www.themoviedb.org/settings/api) *(only required for poster downloads)*

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/balaji624/cinematch-ai.git
cd cinematch-ai
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Download movie posters

Open `download_posters.py`, set your TMDB API key, then run:

```python
API_KEY = "your_tmdb_api_key_here"
```

```bash
python download_posters.py
```

Posters are saved to `static/posters/<movie_id>.jpg`. The app runs without them, but they significantly enhance the UI.

### 5. Run the app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

> On first launch, the ML model is built automatically from the CSVs (~20 seconds). All subsequent starts load the cached `.pkl` files instantly.

---

## 📁 Project Structure

```
cinematch-ai/
│
├── data/                          # Dataset CSVs
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── model/                         # Auto-generated model cache
│   ├── movies.pkl                 # Preprocessed DataFrame
│   └── similarity.pkl             # Hybrid similarity matrix
│
├── static/
│   └── posters/                   # Downloaded TMDB poster images
│
├── templates/
│   ├── index.html                 # Main SPA (served at /)
│   └── about.html                 # ML explainer / about page
│
├── app.py                         # Flask app & all REST API routes
├── recommendation.py              # ML engine: preprocessing, training, inference
├── download_posters.py            # Utility: batch-downloads TMDB posters
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment config (Heroku / Railway)
└── README.md
```

---

## 📄 File Descriptions

### `app.py` — Flask Application & API Layer

The main entry point. Loads the ML model on startup and exposes the full REST API:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recommend` | AI recommendations for a given movie title |
| `GET` | `/search` | Quick fuzzy title search (up to 8 results) |
| `GET` | `/search/advanced` | Multi-filter search — year, rating, genre, language, votes |
| `GET` | `/random` | A random well-rated movie |
| `GET` | `/trending` | Top N movies by Bayesian score |
| `GET` | `/genre/<name>` | Top movies for a genre |
| `GET` | `/director/<name>` | Top movies by a director |
| `GET` | `/stats` | Dataset-wide statistics |

---

### `recommendation.py` — ML Engine

The core of the project. Key functions:

- **`load_and_preprocess()`** — merges CSVs, extracts text/numeric features, computes Bayesian scores
- **`build_model(df)`** — builds TF-IDF + numeric matrices, blends into hybrid similarity, saves to `model/`
- **`load_model()`** — loads cached `.pkl` files, or builds from scratch on first run
- **`get_recommendations()`** — main inference function with year decay and popularity boost
- **`search_movies()`** — three-tier search: exact → contains → fuzzy
- **`get_random_movie()`** — samples from films with ≥ 50 votes
- **`get_trending()`** — top N by Bayesian score
- **`get_by_genre()`** — genre-filtered top films
- **`get_by_director()`** — partial director name match
- **`get_stats()`** — aggregate dataset statistics

---

### `templates/index.html` — Main Frontend

A ~2,700-line single-page application with a cinematic dark UI:

- Hero section with animated search bar and live autocomplete dropdown
- Horizontal-scroll trending movies section
- Genre discovery grid
- Recommendation results with poster, title, year, genres, and match score
- Movie comparison tool
- Advanced search panel with multiple simultaneous filters
- Fully responsive via Tailwind CSS utility classes

All content is rendered dynamically via `fetch()` calls to the Flask API.

---

### `templates/about.html` — About Page

An informational page covering the ML methodology, tech stack, dataset details, and algorithm design — presented in the same cinematic dark UI style as the main app.

---

### `download_posters.py` — Poster Downloader Utility

A standalone batch script that:

1. Reads all movie IDs from `tmdb_5000_movies.csv`
2. Queries the TMDB API for each film's `poster_path`
3. Downloads images to `static/posters/<id>.jpg`
4. Skips files already on disk — safe to re-run at any time
5. Respects TMDB rate limits with a polite 0.25s delay between requests

Configure `API_KEY`, `IMAGE_SIZE`, and `POSTER_DIR` at the top of the file before running.

---

## ⚙️ Configuration

Tune these constants in `recommendation.py` to adjust model behavior:

| Constant | Default | Description |
|---|---|---|
| `CONTENT_WEIGHT` | `0.70` | TF-IDF content similarity weight in hybrid blend |
| `NUMERIC_WEIGHT` | `0.30` | Numeric feature similarity weight |
| `MIN_VOTES_TO_RECOMMEND` | `10` | Minimum vote count for a film to appear in results |
| `YEAR_DECAY_SIGMA` | `15` | Gaussian σ controlling year-proximity decay (years) |
| `POPULARITY_BOOST` | `0.05` | Re-ranking multiplier for TMDB popularity |

---

## 🖼️ Screenshots

**Homepage — Cinematic Hero Section**
![Homepage](/static/image.png)

**Search Autocomplete — Fuzzy-Matched Dropdown**
![Search Autocomplete](/static/image-1.png)

**Recommendation Results — Movie Cards with Match Score**
![Recommendations](/static/image-2.png)

**Individual Movie Detail Card**
![individual movie detail card](/static/image-4.png)

**Genre Discovery — Top Movies Grid**
![Genre Discovery](/static/image-3.png)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) — dataset and poster API
- [Kaggle TMDB 5000 Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Scikit-learn](https://scikit-learn.org/) — TF-IDF, cosine similarity, and scaling tools
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — fast fuzzy string matching
- [Tailwind CSS](https://tailwindcss.com/) — responsive dark UI framework

---

<div align="center">
  Made with ❤️ and 🎬 by <strong>Balaji624</strong>
  <br>
  <a href="https://github.com/balaji624/cinematch-ai">⭐ Star this repo if you found it useful!</a>
</div>