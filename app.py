import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    return movies, credits

# 감독 추출 함수
def extract_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
    except:
        return np.nan
    return np.nan

# 장르 이름 리스트 추출 함수
def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g['name'] for g in genres]
    except:
        return []

def preprocess_data(movies, credits):
    # 감독 추출
    credits['director'] = credits['crew'].apply(extract_director)

    # 필요한 컬럼만 선택
    credits_small = credits[['id', 'director']]

    # 장르 추출
    movies['genres_list'] = movies['genres'].apply(extract_genres)

    # movies와 credits 합치기
    df = pd.merge(movies, credits_small, on='id')

    # 장르와 감독 중 상위 20개만 사용, 나머지는 'Other'로 처리
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    all_genres = [g for sublist in df['genres_list'] for g in sublist]
    top_genres = pd.Series(all_genres).value_counts().head(20).index
    def clean_genres(genres):
        return [g if g in top_genres else 'Other' for g in genres]
    df['clean_genres'] = df['genres_list'].apply(clean_genres)

    # 감독 one-hot 인코딩
    director_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    director_encoded = director_encoder.fit_transform(df[['director']])
    director_cols = [f'director_{d}' for d in director_encoder.categories_[0]]
    df_director = pd.DataFrame(director_encoded, columns=director_cols, index=df.index)

    # 장르 one-hot 인코딩 (multi-label 처리)
    genres_exploded = df.explode('clean_genres')
    genre_dummies = pd.get_dummies(genres_exploded['clean_genres'], prefix='genre')
    genre_dummies = genre_dummies.groupby(genres_exploded.index).max()  # 한 영화에 여러 장르 있을 수 있으니 max

    # 숫자 피처 선택 (budget, runtime, popularity)
    features_num = df[['budget', 'runtime', 'popularity']]

    # 피처 합치기
    X = pd.concat([features_num, df_director, genre_dummies], axis=1)

    # 타겟
    y = df['revenue']

    # 결측치 제거
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y, director_encoder, top_directors, top_genres

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

def main():
    st.title("영화 수익 예측기 (TMDB 데이터 기반)")

    # 데이터 로드
    movies, credits = load_data()

    # 데이터 전처리 및 피처 생성
    with st.spinner('데이터 전처리 중...'):
        X, y, director_enc, top_directors, top_genres = preprocess_data(movies, credits)

    # 모델 학습
    with st.spinner('모델 학습 중...'):
        model, rmse, r2 = train_model(X, y)

    st.write(f"모델 성능: RMSE = {rmse:,.0f}, R^2 = {r2:.3f}")

    st.header("예측 입력값 설정")

    # UI: 입력값 받기
    budget = st.number_input('예산 (budget)', min_value=0, max_value=1_000_000_000, value=100_000_000, step=1_000_000)
    runtime = st.number_input('상영시간 (runtime, 분)', min_value=30, max_value=300, value=120, step=1)
    popularity = st.number_input('인기도 (popularity)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    director = st.selectbox('감독 선택', ['Other'] + list(top_directors))
    genre = st.multiselect('장르 선택 (최대 3개)', options=list(top_genres) + ['Other'], default=['Action'])

    # 입력값을 모델 입력 피처에 맞게 변환
    input_dict = {
        'budget': budget,
        'runtime': runtime,
        'popularity': popularity,
    }

    # 감독 원핫
    director_vec = np.zeros(len(director_enc.categories_[0]))
    if director in director_enc.categories_[0]:
        idx = list(director_enc.categories_[0]).index(director)
        director_vec[idx] = 1
    director_df = pd.DataFrame([director_vec], columns=[f'director_{d}' for d in director_enc.categories_[0]])

    # 장르 원핫 (multi-label)
    genre_vec = np.zeros(len(top_genres))
    for g in genre:
        if g in top_genres:
            idx = list(top_genres).index(g)
            genre_vec[idx] = 1
    genre_df = pd.DataFrame([genre_vec], columns=[f'genre_{g}' for g in top_genres])

    # 숫자 피처 데이터프레임
    input_num_df = pd.DataFrame([{k: v for k, v in input_dict.items()}])

    # 모델 입력 피처 조합
    input_df = pd.concat([input_num_df, director_df, genre_df], axis=1)

    # 없는 컬럼(훈련 데이터에만 있던)이 있으면 0으로 채우기
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]  # 컬럼 순서 맞추기

    # 예측 실행
    if st.button('수익 예측하기'):
        pred = model.predict(input_df)[0]
        st.success(f"예측된 수익: {pred:,.0f} 달러")

if __name__ == "__main__":
    main()
