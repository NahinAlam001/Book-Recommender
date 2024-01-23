import pickle
import streamlit as st
import numpy as np

def load_data():
    try:
        model = pickle.load(open('artifacts/model.pkl', 'rb'))
        book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
        final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
        book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}")
        return None, None, None, None

    return model, book_names, final_rating, book_pivot

def fetch_poster(suggestion, final_rating):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(final_rating.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx].get('img_url', 'default_image.png')
        poster_url.append(url)

    return poster_url

def recommend_book(book_name, model, book_pivot, final_rating):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion, final_rating)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)

    return books_list, poster_url

# Load data
model, book_names, final_rating, book_pivot = load_data()

# Streamlit app
st.header('Book Recommender System Using Machine Learning')

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books, model, book_pivot, final_rating)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])
