import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

def preprocess_data(movies, credits):
    credits['crew'] = credits['crew'].apply(ast.literal_eval)

    def get_director(crew_list):
        for crew_member in crew_list:
            if crew_member.get('job') == 'Director':
                return crew_member.get('name')
        return np.nan

    credits['director'] = credits['crew'].apply(get_director)
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 장르 추출
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['genre'] = df['genres'].apply(lambda x: x[0]['name'] if len(x) > 0 else 'Unknown')

    # 필요한 열만 추출
    df = df[['budget', 'vote_average', 'runtime', 'revenue', 'director', 'genre']].dropna()

    # 상위 감독만 남기고 나머지는 Other
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    # 원핫 인코딩
    director_encoder = OneHotEncoder(sparse_output=False)
    genre_encoder = OneHotEncoder(sparse_output=False)

    director_encoded = director_encoder.fit_transform(df[['director']])
    genre_encoded = genre_encoder.fit_transform(df[['genre']])

    director_cols = [f'director_{cat}' for cat in director_encoder.categories_[0]]
    genre_cols = [f'genre_{cat}' for cat in genre_encoder.categories_[0]]

    df_director = pd.DataFrame(director_encoded, columns=director_cols, index=df.index)
    df_genre = pd.DataFrame(genre_encoded, columns=genre_cols, index=df.index)

    df_final = pd.concat([df[['budget', 'vote_average', 'runtime']], df_director, df_genre], axis=1)
    y = df['revenue']

    return df_final, y, director_encoder, genre_encoder

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_revenue(model, director_encoder, genre_encoder, budget, vote_avg, runtime, director, genre):
    input_df = pd.DataFrame([[budget, vote_avg, runtime]], columns=['budget', 'vote_average', 'runtime'])

    # 인코딩
    director_encoded = director_encoder.transform([[director]])
    genre_encoded = genre_encoder.transform([[genre]])

    dir_cols = [f'director_{cat}' for cat in director_encoder.categories_[0]]
    gen_cols = [f'genre_{cat}' for cat in genre_encoder.categories_[0]]

    df_dir = pd.DataFrame(director_encoded, columns=dir_cols)
    df_gen = pd.DataFrame(genre_encoded, columns=gen_cols)

    input_all = pd.concat([input_df, df_dir, df_gen], axis=1)

    # 누락된 열 채우기
    missing_cols = set(model.feature_names_in_) - set(input_all.columns)
    for col in missing_cols:
        input_all[col] = 0

    input_all = input_all[model.feature_names_in_]

    return model.predict(input_all)[0]

# ======================== Streamlit 앱 ========================

def main():
    st.title("🎬 영화 수익 예측기")
    st.write("영화의 기본 정보를 입력하면 예측 수익을 알려줍니다!")

    movies, credits = load_data()
    X, y, director_encoder, genre_encoder = preprocess_data(movies, credits)
    model = train_model(X, y)

    st.sidebar.header("입력값을 설정하세요")

    budget = st.sidebar.number_input("예산 (USD)", min_value=0, value=10000000, step=1000000)
    vote_avg = st.sidebar.slider("평균 평점", 0.0, 10.0, 7.0, 0.1)
    runtime = st.sidebar.number_input("상영시간 (분)", min_value=30, value=100, step=5)

    directors = list(director_encoder.categories_[0])
    genres = list(genre_encoder.categories_[0])

    director = st.sidebar.selectbox("감독", directors)
    genre = st.sidebar.selectbox("장르", genres)

    if st.sidebar.button("수익 예측하기"):
        revenue = predict_revenue(model, director_encoder, genre_encoder, budget, vote_avg, runtime, director, genre)
        st.success(f"📈 예측된 수익: **${int(revenue):,} USD**")

        # 평가 점수 표시
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        st.write(f"📊 모델 성능 - RMSE: {rmse:,.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()
