import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_model(df):
    """Membangun model similarity menggunakan TF-IDF pada synopsis_clean"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['synopsis_clean'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_anime(title, cosine_sim, df, top_n=5):
    """
    Memberikan rekomendasi anime berdasarkan kemiripan sinopsis (cosine similarity).
    Tidak memfilter berdasarkan rating atau genre di tahap ini.

    Args:
        title (str): Judul anime sebagai query.
        cosine_sim (ndarray): Matrix cosine similarity.
        df (DataFrame): DataFrame anime.
        top_n (int): Jumlah rekomendasi teratas yang diambil.

    Returns:
        DataFrame atau str jika error.
    """
    try:
        indices = pd.Series(df.index, index=df['title'].str.lower())
        title = title.lower()

        if title not in indices.index:
            return f"Anime '{title}' tidak ditemukan dalam database."

        idx = indices[title] if not isinstance(
            indices[title], pd.Series) else indices[title].iloc[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1:top_n+1]

        anime_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        result_cols = ['title', 'genres', 'score', 'synopsis', 'image_url']
        rekomendasi = df.iloc[anime_indices][result_cols].copy()
        rekomendasi['similarity'] = similarity_scores

        return rekomendasi

    except Exception as e:
        return f"Terjadi error dalam sistem rekomendasi: {str(e)}"
