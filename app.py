import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

# 데이터 불러오기 (캐시)
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

# 감독 추출 함수
def extract_director(crew_str):
    try:
        crew = json.loads(crew_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except:
        return 'Unknown'
    return 'Unknown'

# 장르 추출 함수
def extract_genres(genres_str):
    try:
        genres = json.loads(genres_str)
        return [genre['name'] for genre in genres]
    except:
        return []

# 데이터 전처리 함수
def preprocess_data(movies, credits):
    credits['director'] = credits['crew'].apply(extract_director)
    df = pd.merge(movies, credits[['id', 'director']], on='id')
    df['genres'] = df['genres'].apply(lambda x: ','.join(extract_genres(x)))
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()

    director_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    df['director_encoded'] = director_enc.fit_transform(df['director'])
    df['genres_encoded'] = genre_enc.fit_transform(df['genres'])

    X = df[['budget', 'popularity', 'runtime', 'director_encoded', 'genres_encoded']]
    y = df['revenue']
    return X, y, director_enc, genre_enc

# 메인 앱
def main():
    st.title("🎬 영화 수익 예측 앱")

    movies, credits = load_data()
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.success(f"✅ 모델 RMSE (예측 오차): {rmse:,.0f} 달러")

    st.subheader("🎯 영화 정보 입력")
    budget = st.number_input("예산 (budget)", min_value=0, step=1000, value=10000000)
    popularity = st.number_input("인기도 (popularity)", min_value=0.0, step=0.1, value=10.0)
    runtime = st.number_input("상영 시간 (runtime)", min_value=1, step=1, value=120)
    director = st.selectbox("감독 (director)", director_enc.classes_)
    genre = st.selectbox("장르 (genres)", genre_enc.classes_)

    if st.button("📈 수익 예측하기"):
        director_code = director_enc.transform([director])[0]
        genre_code = genre_enc.transform([genre])[0]
        input_data = np.array([[budget, popularity, runtime, director_code, genre_code]])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 예측 수익: {int(prediction):,} 달러")

# 실행
if __name__ == '__main__':
    main()
