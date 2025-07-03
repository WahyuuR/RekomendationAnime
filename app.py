import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Import fungsi rekomendasi
from Model.rekomendasi import build_model, recommend_anime

# Import login dan bookmark modules
from Modules.login import login, logout
from Modules.bookmark import add_bookmark, get_bookmarks, remove_multiple_bookmarks

# ================== Fungsi Utama ==================


@st.cache_data
def load_data():
    """Memuat dan memproses data anime"""
    try:
        df = pd.read_csv('Dataset/anime_cleaned_token.csv')

        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['genres'] = df['genres'].fillna('Unknown')

        if 'image_url' not in df.columns:
            df['image_url'] = 'https://via.placeholder.com/150x200?text=No+Image'

        df['image_url'] = df['image_url'].apply(
            lambda x: x if str(x).startswith('http')
            else 'https://via.placeholder.com/150x200?text=No+Image'
        )

        df['synopsis_clean'] = df['synopsis_clean'].fillna('')

        return df

    except Exception as e:
        st.error(f"Gagal memuat dataset: {str(e)}")
        st.stop()

# ================== Fungsi Load CSS ==================


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================== Fungsi Display Card ==================


def display_anime_card(title, genres, score, similarity, synopsis, image_url):
    """Menampilkan card anime dalam layout yang menarik"""
    card = st.container(border=True)
    col1, col2 = card.columns([1, 3])

    with col1:
        st.image(image_url, use_container_width=True)

    with col2:
        st.markdown(f"### {title}")
        st.caption(f"**Genre:** {genres}")

        col_score, col_sim = st.columns(2)
        with col_score:
            st.metric("Rating", f"{score:.2f}")
        with col_sim:
            st.metric("Similarity", f"{similarity:.2f}")

        with st.expander("📜 Sinopsis"):
            st.write(synopsis if pd.notna(synopsis)
                     else "Sinopsis tidak tersedia")

        if st.session_state.get('logged_in'):
            if st.button(f"🔖 Bookmark {title}", key=f"bookmark_{title}"):
                add_bookmark(title)

# ================== Aplikasi Utama ==================


def main():
    st.set_page_config(
        page_title="Anime Recommender Pro",
        layout="wide",
        page_icon="🎬",
        initial_sidebar_state="expanded"
    )

    # Load external CSS
    local_css("assets/style.css")

    # Login check
    if not st.session_state.get('logged_in'):
        login()
        return
    else:
        st.sidebar.markdown(
            f"👤 **Logged in as {st.session_state['username']}**")
        logout()

    # Title
    st.title("🎬 Anime Recommendation Engine")
    st.markdown("""
    <div class="main-title-container">
    <h4>
    Temukan anime serupa berdasarkan kesamaan sinopsis dengan teknologi AI
    </h4>
    </div>
    """, unsafe_allow_html=True)

    # Load data & model
    df = load_data()
    cosine_sim = build_model(df)

    # Sidebar - Menu
    menu = st.sidebar.radio("Menu", ["Rekomendasi", "Bookmark"])

    if menu == "Rekomendasi":
        with st.sidebar:
            st.title("🔍 Rekomendasi Berdasarkan Sinopsis")
            selected_anime = st.selectbox(
                "Pilih Anime Favorit Anda",
                sorted(df['title'].unique()),
                help="Pilih anime untuk mendapatkan rekomendasi berbasis sinopsis"
            )

            top_n = st.slider(
                "Jumlah Rekomendasi",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Jumlah anime yang ditampilkan dalam hasil rekomendasi"
            )

            if st.button("🎯 Dapatkan Rekomendasi", type="primary"):
                st.session_state['show_recommendations'] = True

        if st.session_state.get('show_recommendations'):
            with st.spinner('⏳ Sistem sedang memproses rekomendasi Anda... Mohon tunggu...'):
                recommendations = recommend_anime(
                    selected_anime, cosine_sim, df, top_n=top_n
                )

                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    selected_info = df[df['title'].str.lower()
                                       == selected_anime.lower()].iloc[0]
                    st.subheader(f"🔍 Anda memilih: {selected_anime}")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(selected_info['image_url'],
                                 use_container_width=True)

                    with col2:
                        st.markdown(f"**🎭 Genre:** {selected_info['genres']}")
                        st.markdown(
                            f"**⭐ Rating:** {selected_info['score']}/10")
                        with st.expander("📖 Sinopsis Lengkap"):
                            st.write(selected_info['synopsis']
                                     if pd.notna(selected_info['synopsis'])
                                     else "Sinopsis tidak tersedia")

                    st.subheader("✨ Rekomendasi Untuk Anda")
                    for _, row in recommendations.iterrows():
                        display_anime_card(
                            row['title'], row['genres'],
                            row['score'], row['similarity'],
                            row['synopsis'], row['image_url']
                        )

    elif menu == "Bookmark":
        st.subheader("🔖 Daftar Bookmark Anda")
        bookmarks = get_bookmarks()
        if bookmarks:
            selected_to_delete = st.multiselect(
                "Pilih bookmark yang ingin dihapus",
                options=bookmarks,
                help="Pilih satu atau lebih bookmark untuk dihapus"
            )
            if st.button("🗑️ Hapus Bookmark Terpilih"):
                if selected_to_delete:
                    remove_multiple_bookmarks(selected_to_delete)
                else:
                    st.info("Tidak ada bookmark yang dipilih untuk dihapus.")

            # Tampilkan semua bookmark
            for title in bookmarks:
                anime = df[df['title'] == title].iloc[0]
                display_anime_card(
                    anime['title'], anime['genres'],
                    anime['score'], 1.0,  # similarity dummy
                    anime['synopsis'], anime['image_url']
                )
        else:
            st.info("Bookmark Anda masih kosong.")


if __name__ == "__main__":
    main()
