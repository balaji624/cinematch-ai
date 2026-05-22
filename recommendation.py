import os
import re
import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, MinMaxScaler
from rapidfuzz import fuzz

#Paths
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

#Blend weights
CONTENT_WEIGHT  = 0.70   # TF-IDF multi-field
NUMERIC_WEIGHT  = 0.30   # budget, revenue, runtime, popularity, vote_average

MIN_VOTES_TO_RECOMMEND = 10     # hide truly obscure films
YEAR_DECAY_SIGMA       = 15     # Gaussian year-proximity σ (years)
POPULARITY_BOOST       = 0.05   # small multiplier on normalised TMDB popularity




def _extract_names(json_str, key='name', limit=None):
    """Parse a TMDB JSON list string and return space-joined names."""
    try:
        items = ast.literal_eval(str(json_str))
        names = [item[key] for item in items if key in item]
        if limit:
            names = names[:limit]
        return ' '.join(n.replace(' ', '_') for n in names)
    except Exception:
        return ''


def _extract_director(crew_str):
    """Return director name(s) from crew JSON."""
    try:
        crew = ast.literal_eval(str(crew_str))
        directors = [
            m['name'].replace(' ', '_')
            for m in crew
            if m.get('job') == 'Director'
        ]
        return ' '.join(directors)
    except Exception:
        return ''


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


#Preprocessing

def load_and_preprocess():
    movies  = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_movies.csv'))
    credits = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_credits.csv'))

    # credits has movie_id; movies has id — merge on those
    credits = credits.rename(columns={'movie_id': 'id', 'title': 'credit_title'})
    df = movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')

    df['genres_str']    = df['genres'].apply(lambda x: _extract_names(x))
    df['keywords_str']  = df['keywords'].apply(lambda x: _extract_names(x))
    df['cast_str']      = df['cast'].apply(lambda x: _extract_names(x, limit=5))
    df['director_str']  = df['crew'].apply(_extract_director)
    df['overview_str']  = df['overview'].fillna('').str.lower()
    df['tagline_str']   = df['tagline'].fillna('').str.lower()
    df['orig_lang']     = df['original_language'].fillna('en')

    df['budget']        = df['budget'].apply(_safe_float)
    df['revenue']       = df['revenue'].apply(_safe_float)
    df['runtime']       = df['runtime'].apply(_safe_float)
    df['popularity']    = df['popularity'].apply(_safe_float)
    df['vote_average']  = df['vote_average'].apply(_safe_float)
    df['vote_count']    = df['vote_count'].apply(_safe_float)

    df['budget']  = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)

    #Bayesian weighted rating
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.60)
    df['bayesian_score'] = (
        (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] +
        (m / (df['vote_count'] + m)) * C
    )

    #Year
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    #Clean title
    df['clean_title'] = df['title'].str.strip()

    df['combined'] = (
        df['genres_str']   + ' ' + df['genres_str']   + ' ' +
        df['genres_str']   + ' ' + df['genres_str']   + ' ' +
        df['keywords_str'] + ' ' + df['keywords_str'] + ' ' +
        df['keywords_str'] + ' ' +
        df['director_str'] + ' ' + df['director_str'] + ' ' +
        df['director_str'] + ' ' +
        df['cast_str']     + ' ' + df['cast_str']     + ' ' +
        df['cast_str']     + ' ' +
        df['overview_str'] + ' ' + df['overview_str'] + ' ' +
        df['tagline_str']
    )

    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df)} movies from TMDB 5000 dataset.")
    return df


#Model building

def build_model(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("  [1/2] Building content model (TF-IDF)...")
    tfidf = TfidfVectorizer(
        max_features=30000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    content_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"     Content matrix: {content_sim.shape}")


    print("  [2/2] Building numeric feature model...")
    num_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']
    num_df   = df[num_cols].copy()

   
    for col in num_cols:
        num_df[col] = num_df[col].fillna(num_df[col].median())

    scaler    = MinMaxScaler()
    num_norm  = scaler.fit_transform(num_df)
    num_sim   = cosine_similarity(num_norm, num_norm)
    print(f"     Numeric matrix: {num_sim.shape}")

    #3. Blend
    print("  [3/3] Blending hybrid matrix...")
    hybrid_sim = (
        CONTENT_WEIGHT * content_sim +
        NUMERIC_WEIGHT * num_sim
    )
    print(f"     Hybrid matrix: {hybrid_sim.shape}")

    #Save
    with open(os.path.join(MODEL_DIR, 'similarity.pkl'), 'wb') as f:
        pickle.dump(hybrid_sim, f)
    with open(os.path.join(MODEL_DIR, 'movies.pkl'), 'wb') as f:
        pickle.dump(df, f)

    print(f"Model saved. ({len(df)} movies)")
    return df, hybrid_sim


#Load

def load_model():
    sim_path = os.path.join(MODEL_DIR, 'similarity.pkl')
    mov_path = os.path.join(MODEL_DIR, 'movies.pkl')

    if os.path.exists(sim_path) and os.path.exists(mov_path):
        with open(sim_path, 'rb') as f:
            similarity = pickle.load(f)
        with open(mov_path, 'rb') as f:
            df = pickle.load(f)
        print(f"TMDB hybrid model loaded: {len(df)} movies")
        return df, similarity

    print("Building TMDB hybrid model for first time (~20 sec)...")
    df = load_and_preprocess()
    return build_model(df)



def _find_movie_idx(title, df):
    t = title.lower().strip()

    for col in ['clean_title', 'title', 'original_title']:
        if col not in df.columns:
            continue
        mask = df[col].str.lower() == t
        if mask.any():
            return df[mask].index[0]

    for col in ['clean_title', 'title']:
        mask = df[col].str.lower().str.contains(re.escape(t), regex=True, na=False)
        if mask.any():
            candidates = df[mask]
            return candidates['vote_count'].idxmax()

    best_score, best_idx = 0, None
    for idx, row in df.iterrows():
        score = fuzz.token_set_ratio(t, str(row['clean_title']).lower())
        if score > best_score:
            best_score, best_idx = score, idx
    if best_score >= 60:
        return best_idx
    return None


def _year_weight(source_year, target_year, sigma=YEAR_DECAY_SIGMA):
    if pd.isna(source_year) or pd.isna(target_year):
        return 1.0
    delta = abs(float(source_year) - float(target_year))
    return float(np.exp(-0.5 * (delta / sigma) ** 2))


def _row_to_dict(row):
    """Serialise a DataFrame row to a JSON-safe dict with TMDB-rich fields."""
    genres_raw = row.get('genres', '')
    try:
        genres_list = [g['name'] for g in ast.literal_eval(str(genres_raw))]
    except Exception:
        genres_list = []

    keywords_raw = row.get('keywords', '')
    try:
        kw_list = [k['name'] for k in ast.literal_eval(str(keywords_raw))][:10]
    except Exception:
        kw_list = []

    budget  = row.get('budget', 0)
    revenue = row.get('revenue', 0)

    return {
        'id':            int(row['id']),
        'title':         str(row['clean_title']),
        'original_title': str(row.get('original_title') or row.get('clean_title') or row.get('title', '')),
        'year':          str(int(row['year'])) if pd.notna(row.get('year')) else 'N/A',
        'genres':        genres_list,
        'genre_str':     ', '.join(genres_list),
        'keywords':      kw_list,
        'overview':      str(row.get('overview', ''))[:400],
        'tagline':       str(row.get('tagline', '')),
        'director':      str(row.get('director_str', '')).replace('_', ' '),
        'cast':          str(row.get('cast_str', '')).replace('_', ' '),
        'language':      str(row.get('original_language', 'en')),
        'runtime':       int(row['runtime']) if pd.notna(row.get('runtime')) and _safe_float(row.get('runtime')) > 0 else None,
        'budget':        int(budget) if pd.notna(budget) and budget > 0 else None,
        'revenue':       int(revenue) if pd.notna(revenue) and revenue > 0 else None,
        'popularity':    round(float(row.get('popularity', 0)), 2),
        'vote_average':  round(float(row.get('vote_average', 0)), 1),
        'vote_count':    int(row.get('vote_count', 0)),
        'bayesian_score': round(float(row.get('bayesian_score', 0)), 3),
        'tmdb_id':       int(row['id']),
        'homepage':      str(row.get('homepage', '')) if pd.notna(row.get('homepage')) else '',
    }



def get_recommendations(movie_title, df, similarity, n=10, language_filter=None):
    idx = _find_movie_idx(movie_title, df)
    if idx is None:
        return None, f"Movie '{movie_title}' not found in the dataset."

    source_row  = df.iloc[idx]
    source_year = source_row.get('year')

    raw_scores = list(enumerate(similarity[idx]))

    scored = []
    for i, sim_score in raw_scores:
        if i == idx:
            continue
        row = df.iloc[i]
        if int(row.get('vote_count', 0)) < MIN_VOTES_TO_RECOMMEND:
            continue
        if language_filter and str(row.get('original_language', 'en')) != language_filter:
            continue

        yr_w = _year_weight(source_year, row.get('year'))
        # Small popularity boost (normalised 0–1 within dataset)
        pop_norm   = float(row.get('popularity', 0)) / (df['popularity'].max() + 1e-9)
        final      = sim_score * (0.85 + 0.15 * yr_w) * (1 + POPULARITY_BOOST * pop_norm)
        scored.append((i, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n]

    results = []
    for i, score in top:
        rec = _row_to_dict(df.iloc[i])
        rec['match_score'] = min(round(float(score) * 100, 1), 99.0)
        results.append(rec)

    return {'source': _row_to_dict(source_row), 'recommendations': results}, None


def search_movies(query, df, limit=8):
    q = query.lower().strip()
    results  = []
    seen_ids = set()

    def _add(rows_iter):
        for _, row in rows_iter:
            mid = int(row['id'])
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            results.append(row)

    exact = df[df['clean_title'].str.lower() == q]
    _add(exact.iterrows())

    if len(results) < limit:
        contains = df[
            df['clean_title'].str.lower().str.contains(re.escape(q), regex=True, na=False) &
            ~df['id'].isin(seen_ids)
        ]
        _add(contains.iterrows())

    if len(results) < limit:
        fuzzy_matches = []
        for idx, row in df.iterrows():
            if row['id'] in seen_ids:
                continue
            score = fuzz.token_set_ratio(q, str(row['clean_title']).lower())
            if score >= 70:
                fuzzy_matches.append((row, score))
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        for row, _ in fuzzy_matches:
            mid = int(row['id'])
            if mid not in seen_ids:
                seen_ids.add(mid)
                results.append(row)

    output = []
    for row in results[:limit]:
        genres_raw = row.get('genres', '')
        try:
            genres_list = [g['name'] for g in ast.literal_eval(str(genres_raw))]
        except Exception:
            genres_list = []
        output.append({
            'id':          int(row['id']),
            'title':       str(row['clean_title']),
            'year':        str(int(row['year'])) if pd.notna(row.get('year')) else 'N/A',
            'genres':      genres_list,
            'vote_average': round(float(row.get('vote_average', 0)), 1),
            'vote_count':  int(row.get('vote_count', 0)),
            'popularity':  round(float(row.get('popularity', 0)), 2),
        })

    output.sort(key=lambda x: x['vote_count'] * x['vote_average'], reverse=True)
    return output


def get_random_movie(df):
    pool = df[df['vote_count'] >= 50]
    return _row_to_dict(pool.sample(1).iloc[0])


def get_trending(df, n=10):
    return [_row_to_dict(row) for _, row in
            df.nlargest(n, 'bayesian_score').iterrows()]


def get_by_genre(genre, df, n=12):
    gdf = df[df['genres'].str.contains(genre, case=False, na=False)].copy()
    if gdf.empty:
        return []
    return [_row_to_dict(row) for _, row in
            gdf.nlargest(n, 'bayesian_score').iterrows()]


def get_by_director(director_name, df, n=12):
    """Return top movies by a given director."""
    mask = df['director_str'].str.replace('_', ' ').str.lower().str.contains(
        director_name.lower(), na=False
    )
    gdf = df[mask].copy()
    if gdf.empty:
        return []
    return [_row_to_dict(row) for _, row in
            gdf.nlargest(n, 'bayesian_score').iterrows()]


def get_stats(df):
    genres_all = []
    for g_str in df['genres'].dropna():
        try:
            genres_all.extend(g['name'] for g in ast.literal_eval(str(g_str)))
        except Exception:
            pass

    languages = df['original_language'].value_counts().head(10).to_dict()

    return {
        'total_movies':    len(df),
        'total_votes':     int(df['vote_count'].sum()),
        'avg_rating':      round(float(df['vote_average'].mean()), 2),
        'genres':          sorted(set(genres_all)),
        'year_range':      [int(df['year'].min()), int(df['year'].max())] if df['year'].notna().any() else [None, None],
        'languages':       {k: int(v) for k, v in languages.items()},
    }
