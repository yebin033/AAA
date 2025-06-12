import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

def extract_director(crew_data):
    try:
        crew = ast.literal_eval(crew_data)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except:
        return np.nan

def preprocess_data(movies, credits):
    credits['director'] = credits['crew'].apply(extract_director)
    credits = credits[['id', 'director']]

    movies['id'] = movies['id'].astype(int)
    credits['id'] = credits['id'].astype(int)

    df = pd.merge(movies, credits, on='id')

    df = df[['budget', 'popularity', 'runtime', 'revenue', 'director', 'genres']]
    df.dropna(inplace=True)

    df = df[df['budget'] > 0]
    df = df[df['revenue'] > 0]

    # 장르에서 첫 번째 값만 추출
    def get_first_genre(genres):
        try:
            genre_list = ast.literal_eval(genres)
            if genre_list:
                return genre_list[0]['name']
        except:
            return 'Unknown'

    df['genre'] = df['genres'].apply(get_first_genre)
    df.drop('genres', axis=1, inplace=True)

    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    # 원-핫 인코딩
    director_encoder = OneHotEncoder(sparse_output=False)
    genre_encoder = OneHotEncoder(sparse_output=False)

    director_encoded = director_encoder.fit_transform(df[['director']])
    genre_encoded = genre_encoder.fit_transform(df[['genre']])

    director_cols = [f'director_{cat}' for cat in director_encoder.categories_[0]]
    genre_cols = [f'genre_{cat}' for cat in genre_encoder.categories_[0]]

    df_encoded = pd.concat([
        df[['budget', 'popularity', 'runtime']],
        pd.DataFrame(director_encoded, columns=director_cols, index=df.index),
        pd.DataFrame(genre_encoded, columns=genre_cols, index=df.index)
    ], axis=1)

    X = df_encoded
    y = df['revenue']

    return X, y, director_encoder, genre_encoder

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_revenue(budget, popularity, runtime, director, genre, director_enc, genre_enc, model):
    input_dict = {
        'budget': [budget],
        'popularity': [popularity],
        'runtime': [runtime]
    }

    # 인코딩
    director_array = director_enc.transform([[director]])
    genre_array = genre_enc.transform([[genre]])

    director_cols = [f'director_{cat}' for cat in director_enc.categories_[0]]
    genre_cols = [f'genre_{cat}' for cat in genre_enc.categories_[0]]

    input_df = pd.DataFrame(input_dict)
    director_df = pd.DataFrame(director_array, columns=director_cols)
    genre_df = pd.DataFrame(genre_array, columns=genre_cols)

    full_input = pd.concat([input_df, director_df, genre_df], axis=1)

    # 누락된 컬럼 보완
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in full_input.columns:
            full_input[col] = 0

    full_input = full_input[model_columns]

    prediction = model.predict(full_input)[0]
    return prediction

def main():
    st.title("🎬 영화 수익 예측기")

    movies, credits = load_data()
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)
    model = train_model(X, y)

    st.subheader("입력값을 바탕으로 수익 예측")
    budget = st.number_input("예산 (USD)", value=1_000_000)
    popularity = st.slider("인기도", 0.0, 100.0, 10.0)
    runtime = st.slider("러닝타임 (분)", 30, 240, 120)

    director = st.selectbox("감독", director_enc.categories_[0])
    genre = st.selectbox("장르", genre_enc.categories_[0])

    if st.button("수익 예측하기"):
        revenue = predict_revenue(budget, popularity, runtime, director, genre, director_enc, genre_enc, model)
        st.success(f"예측된 수익: ${revenue:,.0f}")

if __name__ == "__main__":
    main()
