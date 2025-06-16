import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import json

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

def extract_director(crew_json_str):
    try:
        crew = json.loads(crew_json_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except:
        return 'Unknown'
    return 'Unknown'

def extract_genres(genres_json_str):
    try:
        genres = json.loads(genres_json_str)
        return [genre['name'] for genre in genres]
    except:
        return []

def preprocess_data(movies, credits):
    # 👉 감독 정보 먼저 추가
    credits['director'] = credits['crew'].apply(extract_director)

    # 👉 id 기준으로 병합
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 👉 장르 정보 전처리
    df['genres'] = df['genres'].apply(lambda x: ','.join(extract_genres(x)))

    # 👉 필요한 컬럼만 선택
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']]

    # 👉 결측치 제거
    df = df.dropna()

    # 👉 라벨 인코딩
    director_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    df['director_encoded'] = director_enc.fit_transform(df['director'])
    df['genres_encoded'] = genre_enc.fit_transform(df['genres'])

    X = df[['budget', 'popularity', 'runtime', 'director_encoded', 'genres_encoded']]
    y = df['revenue']

    return X, y, director_enc, genre_enc

def main():
    st.title("🎬 영화 수익 예측 앱")

  

    # 데이터 로드
    movies, credits = load_data()

    # 데이터 전처리
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)

    # 학습 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 예측 및 RMSE 출력
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.success(f"✅ 모델 RMSE: {rmse:,.0f} 달러")

    # 사용자 입력
    st.subheader("🎯 영화 정보 입력")
    budget = st.number_input("예산 (budget)", min_value=0, step=1000, value=10000000)
    popularity = st.number_input("인기도 (popularity)", min_value=0.0, step=0.1, value=10.0)
    runtime = st.number_input("상영 시간 (runtime)", min_value=1, step=1, value=120)
    director = st.selectbox("감독 (director)", director_enc.classes_)
    genre = st.selectbox("장르 (genres)", genre_enc.classes_)

    if st.button("📈 예측하기"):
        director_code = director_enc.transform([director])[0]
        genre_code = genre_enc.transform([genre])[0]
        input_data = np.array([[budget, popularity, runtime, director_code, genre_code]])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 예측된 수익: {int(prediction):,} 달러")

if __name__ == "__main__":
    main()
