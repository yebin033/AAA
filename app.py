import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

def extract_director(crew_json):
    try:
        crew = json.loads(crew_json.replace("'", '"'))
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except Exception:
        return np.nan
    return np.nan

def extract_genres(genres_json):
    try:
        genres = json.loads(genres_json.replace("'", '"'))
        return [genre['name'] for genre in genres]
    except Exception:
        return []

def preprocess_data(movies, credits):
    # 감독 정보 추출
    credits['director'] = credits['crew'].apply(extract_director)

    # 장르 정보 추출
    movies['genres_list'] = movies['genres'].apply(extract_genres)

    # 데이터 병합
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 필요한 컬럼 선택 및 결측치 제거
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres_list', 'revenue']]
    df = df.dropna()

    # 감독 상위 20명으로 한정, 나머지는 'Other'
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    # 장르를 멀티라벨 원핫인코딩
    unique_genres = set(genre for sublist in df['genres_list'] for genre in sublist)
    for genre in unique_genres:
        df[f'genre_{genre}'] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)

    # 감독 원핫인코딩
    director_encoder = OneHotEncoder(sparse_output=False)
    director_encoded = director_encoder.fit_transform(df[['director']])
    director_cols = [f'director_{cat}' for cat in director_encoder.categories_[0]]
    df_director = pd.DataFrame(director_encoded, columns=director_cols, index=df.index)

    # 최종 데이터프레임 합치기
    df_final = pd.concat([df.drop(['director', 'genres_list'], axis=1), df_director], axis=1)

    # 설명 변수와 타깃 변수 분리
    X = df_final.drop('revenue', axis=1)
    y = df_final['revenue']

    return X, y, director_encoder, list(unique_genres)

def train_model(X, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

def predict_revenue(model, director_encoder, unique_genres, budget, popularity, runtime, director, genres):
    # 감독 처리
    if director not in director_encoder.categories_[0]:
        director = 'Other'
    director_encoded = director_encoder.transform([[director]])
    director_cols = [f'director_{cat}' for cat in director_encoder.categories_[0]]

    # 장르 처리
    genre_dict = {f'genre_{genre}': 0 for genre in unique_genres}
    for g in genres:
        if g in unique_genres:
            genre_dict[f'genre_{g}'] = 1

    # 입력 데이터프레임 만들기
    input_df = pd.DataFrame({
        'budget': [budget],
        'popularity': [popularity],
        'runtime': [runtime],
        **genre_dict
    })

    director_df = pd.DataFrame(director_encoded, columns=director_cols)
    input_final = pd.concat([input_df, director_df], axis=1)

    # 모델 예측
    prediction = model.predict(input_final)[0]
    return prediction

def main():
    st.title("영화 수익 예측 앱")

    movies, credits = load_data()
    X, y, director_encoder, unique_genres = preprocess_data(movies, credits)
    model = train_model(X, y)

    st.header("영화 데이터 입력")
    budget = st.number_input("예산 (budget)", min_value=0, value=10000000, step=1000000)
    popularity = st.number_input("인기도 (popularity)", min_value=0.0, value=10.0, step=0.1)
    runtime = st.number_input("상영시간 (runtime, 분)", min_value=1, value=120, step=1)
    director = st.selectbox("감독", options=list(director_encoder.categories_[0]))
    genres = st.multiselect("장르", options=unique_genres)

    if st.button("수익 예측하기"):
        if len(genres) == 0:
            st.error("적어도 하나의 장르는 선택해야 합니다.")
        else:
            revenue_pred = predict_revenue(model, director_encoder, unique_genres, budget, popularity, runtime, director, genres)
            st.success(f"예상 수익: {int(revenue_pred):,} 달러")

if __name__ == "__main__":
    main()
