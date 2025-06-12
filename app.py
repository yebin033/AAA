import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 감독 이름 추출 함수
def extract_director(crew_data):
    try:
        crew_list = ast.literal_eval(crew_data)
        for person in crew_list:
            if person['job'] == 'Director':
                return person['name']
    except:
        return np.nan

# 장르 이름 추출 함수
def get_first_genre(genres):
    try:
        genre_list = ast.literal_eval(genres)
        if genre_list:
            return genre_list[0]['name']
    except:
        return 'Unknown'

# 데이터 불러오기
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

# 데이터 전처리 함수
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

    df['genre'] = df['genres'].apply(get_first_genre)
    df.drop('genres', axis=1, inplace=True)

    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

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

# 예측 함수
def predict_revenue(budget, popularity, runtime, director, genre, model, director_enc, genre_enc):
    input_data = pd.DataFrame({
        'budget': [budget],
        'popularity': [popularity],
        'runtime': [runtime]
    })

    # 감독 처리
    director_array = director_enc.transform([[director]])
    director_cols = [f'director_{cat}' for cat in director_enc.categories_[0]]
    director_df = pd.DataFrame(director_array, columns=director_cols)

    # 장르 처리
    genre_array = genre_enc.transform([[genre]])
    genre_cols = [f'genre_{cat}' for cat in genre_enc.categories_[0]]
    genre_df = pd.DataFrame(genre_array, columns=genre_cols)

    # 누락 컬럼 보정
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = pd.concat([input_data, director_df, genre_df], axis=1).reindex(columns=model.feature_names_in_, fill_value=0)

    return model.predict(input_data)[0]

# Streamlit 인터페이스
def main():
    st.title('🎬 영화 수익 예측기 (TMDB 기반)')
    st.write("제작비, 감독, 장르 등을 바탕으로 영화의 예산 대비 수익을 예측합니다.")

    movies, credits = load_data()
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.sidebar.header('🎥 입력 영화 정보')
    budget = st.sidebar.number_input('제작비 (달러)', min_value=100000, value=10000000, step=100000)
    popularity = st.sidebar.slider('인기도 (popularity)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    runtime = st.sidebar.number_input('상영 시간 (분)', min_value=30, value=100, step=1)
    director = st.sidebar.selectbox('감독', list(director_enc.categories_[0]))
    genre = st.sidebar.selectbox('장르', list(genre_enc.categories_[0]))

    if st.sidebar.button('수익 예측하기'):
        revenue = predict_revenue(budget, popularity, runtime, director, genre, model, director_enc, genre_enc)
        st.subheader(f'📈 예상 수익: ${revenue:,.0f}')

    st.markdown("---")
    st.subheader("📊 모델 평가 지표")
    y_pred = model.predict(X_test)
    st.write(f"✅ RMSE: {mean_squared_error(y_test, y_pred, squared=False):,.2f}")
    st.write(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")

if __name__ == '__main__':
    main()
