from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import ast
from recommendation import (
    load_model, get_recommendations, search_movies,
    get_random_movie, get_trending, get_by_genre,
    get_by_director, get_stats
)

app = Flask(__name__)
CORS(app)

print("Loading TMDB recommendation model...")
movies_df, similarity_matrix = load_model()
print("Model ready!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
@app.route('/about.html')
def about():
    return render_template('about.html')


#Recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or 'movie' not in data:
        return jsonify({'error': 'Provide {"movie": "title"} in JSON body'}), 400

    movie_title = data['movie'].strip()
    if not movie_title:
        return jsonify({'error': 'Movie title cannot be empty'}), 400

    n               = int(data.get('n', 10))
    language_filter = data.get('language')   # e.g. "en", "fr", "ja"

    result, error = get_recommendations(
        movie_title, movies_df, similarity_matrix,
        n=n, language_filter=language_filter
    )
    if error:
        return jsonify({'error': error}), 404
    return jsonify(result)


#Search

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'results': []})
    results = search_movies(query, movies_df, limit=8)
    return jsonify({'results': results})


@app.route('/search/advanced', methods=['GET'])
def advanced_search():
    """
    Filter movies by: title query, min/max year, min vote_average,
    genre, language, min_vote_count. Returns up to `limit` results
    sorted by bayesian_score.
    """
    query          = request.args.get('q', '').strip()
    min_rating     = float(request.args.get('min_rating', 0))
    max_year       = int(request.args.get('max_year', 2100))
    min_year       = int(request.args.get('min_year', 1900))
    genre          = request.args.get('genre', '').strip()
    language       = request.args.get('language', '').strip()
    min_votes      = int(request.args.get('min_votes', 0))
    limit          = int(request.args.get('limit', 20))

    res = movies_df.copy()

    if query:
        res = res[res['clean_title'].str.lower().str.contains(
            query.lower(), regex=False, na=False
        )]

    res = res[
        (res['vote_average'] >= min_rating) &
        (res['year'].fillna(0).astype(float) <= max_year) &
        (res['year'].fillna(0).astype(float) >= min_year) &
        (res['vote_count'] >= min_votes)
    ]

    if genre:
        res = res[res['genres'].str.contains(genre, case=False, na=False)]

    if language:
        res = res[res['original_language'] == language]

    res = res.nlargest(limit, 'bayesian_score')

    output = []
    for _, row in res.iterrows():
        try:
            genres_list = [g['name'] for g in ast.literal_eval(str(row['genres']))]
        except Exception:
            genres_list = []
        output.append({
            'id':           int(row['id']),
            'title':        str(row['clean_title']),
            'year':         str(int(row['year'])) if pd.notna(row.get('year')) else 'N/A',
            'genres':       genres_list,
            'vote_average': round(float(row['vote_average']), 1),
            'vote_count':   int(row['vote_count']),
            'popularity':   round(float(row['popularity']), 2),
            'language':     str(row['original_language']),
        })

    return jsonify({'results': output, 'count': len(output)})


#Discovery endpoints

@app.route('/random', methods=['GET'])
def random_movie():
    return jsonify(get_random_movie(movies_df))


@app.route('/trending', methods=['GET'])
def trending():
    n = int(request.args.get('n', 10))
    return jsonify({'movies': get_trending(movies_df, n=n)})


@app.route('/genre/<genre_name>', methods=['GET'])
def by_genre(genre_name):
    n = int(request.args.get('n', 12))
    movies = get_by_genre(genre_name, movies_df, n=n)
    if not movies:
        return jsonify({'error': f'No movies found for genre: {genre_name}'}), 404
    return jsonify({'genre': genre_name, 'movies': movies})


@app.route('/director/<director_name>', methods=['GET'])
def by_director(director_name):
    """Return top films by a director (partial name match)."""
    n = int(request.args.get('n', 12))
    movies = get_by_director(director_name, movies_df, n=n)
    if not movies:
        return jsonify({'error': f'No movies found for director: {director_name}'}), 404
    return jsonify({'director': director_name, 'movies': movies})


#Stats
@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(get_stats(movies_df))


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
