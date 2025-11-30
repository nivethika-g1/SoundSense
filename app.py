import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide",
)


# ---------------- DATA LOADING & CLEANING ----------------

@st.cache_data
def load_data():
    df1 = pd.read_csv("../data/Audible_Catlog.csv")
    df2 = pd.read_csv("../data/Audible_Catlog_Advanced_Features.csv")

    df = pd.merge(
        df1,
        df2,
        on=["Book Name", "Author"],
        how="inner",
        suffixes=("_x", "_y"),
    )

    # ---- detect Rating column ----
    rating_col = None
    for c in df.columns:
        if "rating" in c.lower():
            rating_col = c
            break

    if rating_col is None:
        st.error("Rating column not found.")
        st.stop()

    df["Rating"] = pd.to_numeric(df[rating_col], errors="coerce")

    # ---- detect Description column ----
    desc_col = None
    for c in df.columns:
        if "desc" in c.lower():
            desc_col = c
            break

    if desc_col:
        df["Description"] = df[desc_col].fillna("")
    else:
        df["Description"] = ""

    # ---- detect Reviews column ----
    reviews_col = None
    for c in df.columns:
        if "review" in c.lower():
            reviews_col = c
            break

    if reviews_col:
        df["Reviews"] = pd.to_numeric(df[reviews_col], errors="coerce")

    # clean
    df = df.dropna(subset=["Rating"])
    df = df.drop_duplicates(subset=["Book Name", "Author"])
    df = df.reset_index(drop=True)

    return df, reviews_col


# ---------------- MODEL BUILDING ----------------

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["Description"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    indices = pd.Series(
        df.index, index=df["Book Name"].str.lower()
    ).drop_duplicates()

    return cosine_sim, indices


# ---------------- HELPER FUNCTIONS ----------------

def book_card(row):
    st.markdown(
        f"""
        **{row['Book Name']}**  
        *by {row['Author']}*  
        ⭐ **{row['Rating']:.2f}**
        """.strip()
    )


def recommend_by_book(df, cosine_sim, indices, min_rating):
    st.subheader("🎯 Recommend by Book")

    search = st.text_input("Search book title")

    book_list = sorted(df["Book Name"].unique())
    if search:
        matches = [b for b in book_list if search.lower() in b.lower()]
        if matches:
            book_list = matches

    selected_book = st.selectbox("Choose a book", book_list)

    if st.button("Recommend"):
        key = selected_book.lower()
        if key not in indices:
            st.error("Book not found.")
            return

        idx = indices[key]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = [
            (i, s) for (i, s) in sim_scores[1:]
            if df.loc[i, "Rating"] >= min_rating
        ][:5]

        if not results:
            st.warning("No similar books found with this rating filter.")
            return

        idxs = [i for i, _ in results]

        st.markdown("### Selected Book")
        book_card(df.loc[idx])

        st.markdown("### Recommended Books")
        st.dataframe(
            df.loc[idxs, ["Book Name", "Author", "Rating"]],
            use_container_width=True
        )


def hidden_gems(df, reviews_col):
    st.subheader("💎 Hidden Gems")

    if not reviews_col or "Reviews" not in df.columns:
        st.info("Review count not available in this dataset.")
        return

    max_reviews = st.slider("Maximum reviews", 10, 1000, 200, 10)
    min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.5, 0.1)

    gems = df[
        (df["Reviews"] <= max_reviews) &
        (df["Rating"] >= min_rating)
    ].sort_values("Rating", ascending=False)

    st.markdown(
        f"Found **{len(gems)}** books with rating ≥ {min_rating} "
        f"and reviews ≤ {max_reviews}"
    )

    st.dataframe(
        gems[["Book Name", "Author", "Rating", "Reviews"]].head(20),
        use_container_width=True
    )


# ---------------- MAIN APP ----------------

def main():
    df, reviews_col = load_data()
    cosine_sim, indices = build_model(df)

    st.markdown(
        """
        <h1 style="text-align:center;">🎧 Audible Insights</h1>
        <p style="text-align:center;">
        Smart Audiobook Recommendations using NLP
        </p>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Books", len(df))
    col2.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    if "Reviews" in df.columns:
        col3.metric("Total Reviews", int(df["Reviews"].sum()))
    else:
        col3.metric("Total Reviews", "N/A")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose",
        ["🎯 Recommendation", "💎 Hidden Gems", "ℹ️ About"]
    )

    global_rating = st.sidebar.slider(
        "Minimum rating filter",
        1.0,
        5.0,
        3.5,
        0.1
    )

    if page.startswith("🎯"):
        recommend_by_book(df, cosine_sim, indices, global_rating)
    elif page.startswith("💎"):
        hidden_gems(df, reviews_col)
    else:
        st.subheader("ℹ️ About")
        st.write(
            """
            This project demonstrates a content-based audiobook recommendation system.

            **Core Technology**
            - Natural Language Processing using TF-IDF
            - Cosine similarity for recommendations
            - Streamlit interface

            **Notes**
            - Genre-based filtering is removed due to poor-quality data.
            - Recommendation quality depends on book descriptions.
            """
        )


if __name__ == "__main__":
    main()
