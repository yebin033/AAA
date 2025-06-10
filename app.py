import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 데이터 로딩 (같은 폴더에 CSV 파일 있어야 함)
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    return movies, credits

# 데이터 전처리 및 모델 학습
@st.cache_data
def train_model(movies, credits):
    # credits에서 감독 이름만 추출
    credits['director'] = credits['crew'].apply(lambda x: 
        next((i['name'] for i in eval(x) if i['job']=='Director'), 'Unknown'))
    
    # movies와 credits 합치기
    df = pd.merge(movies, credits[['movie_id', 'director']], left_on='id', right_on='movie_id')

    # 필요한 컬럼 추출 및 결측치 처리
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()
    df = df[df['revenue'] > 0]  # 수익이 0 이상인 데이터만

    # 장르명만 추출 (genres 컬럼은 JSON 형태)
    df['genres'] = df['genres'].apply(lambda x: 
        next(iter(eval(x)), {'name': 'Unknown'})['name'] if x != '[]' else 'Unknown')

    # 상위 감독 20명만 유지, 나머지는 Other 처리
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    # 피처와 타겟 분리
    X = df[['budget', 'popularity', 'runtime', 'director', 'genres']]
    y = df['revenue']

    # 전처리 파이프라인
    categorical_features = ['director', 'genres']
    numeric_features = ['budget', 'popularity', 'runtime']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    # 파이프라인에 전처리 + 선형 회귀 모델
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    return model

# 예측 함수
def predict_revenue(model, budget, popularity, runtime, director, genre):
    input_df = pd.DataFrame({
        'budget': [budget],
        'popularity': [popularity],
        'runtime': [runtime],
        'director': [director],
        'genres': [genre]
    })
    pred = model.predict(input_df)[0]
    return max(0, int(pred))  # 수익은 음수가 될 수 없으니 0 이상으로

# Streamlit 앱 UI
def main():
    st.title('영화 흥행 수익 예측 앱 (광고 효과 분석용)')
    st.write('예산, 감독, 장르, 러닝타임, 인기도를 입력하면 영화 수익을 예측합니다.')

    movies, credits = load_data()
    model = train_model(movies, credits)

    budget = st.number_input('예산 (Budget, 단위: 달러)', min_value=0, value=50000000, step=1000000)
    popularity = st.slider('인기도 (Popularity)', min_value=0.0, max_value=100.0, value=10.0)
    runtime = st.number_input('러닝타임 (Runtime, 분)', min_value=1, max_value=500, value=120)
    
    # 감독과 장르 선택지를 데이터에서 추출
    directors = ['Other'] + list(model.named_steps['preprocessor'].transformers_[0][1].categories_[0])
    genres = list(model.named_steps['preprocessor'].transformers_[0][1].categories_[1])

    director = st.selectbox('감독 (Director)', directors)
    genre = st.selectbox('장르 (Genre)', genres)

    if st.button('수익 예측하기'):
        revenue = predict_revenue(model, budget, popularity, runtime, director, genre)
        st.success(f'예상 영화 흥행 수익: ${revenue:,} (달러)')

if __name__ == '__main__':
    main()
