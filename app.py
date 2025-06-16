import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import ast

# 감독 이름 추출 함수
def extract_director(crew_str):
    crew = ast.literal_eval(crew_str)
    for member in crew:
        if member.get('job') == 'Director':
            return member.get('name')
    return 'Unknown'

# 장르 리스트 추출 함수
def extract_genres(genres_str):
    genres = ast.literal_eval(genres_str)
    return [genre['name'] for genre in genres]

# 데이터 전처리
def preprocess_data(movies, credits):
    # credits 컬럼명 변경
    credits = credits.rename(columns={'movie_id': 'id'})

    # 감독 정보 추출
    credits['director'] = credits['crew'].apply(extract_director)

    # movies, credits 병합 (id 기준)
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 장르 문자열 -> 리스트 -> 문자열 (콤마 구분)
    df['genres'] = df['genres'].apply(lambda x: ','.join(extract_genres(x)))

    # 필요한 컬럼만 선택 및 결측치 제거
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()

    # 라벨 인코딩
    director_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    df['director_encoded'] = director_enc.fit_transform(df['director'])
    df['genres_encoded'] = genre_enc.fit_transform(df['genres'])

    # 설명 변수, 타겟 변수 분리
    X = df[['budget', 'popularity', 'runtime', 'director_encoded', 'genres_encoded']]
    y = df['revenue']

    return X, y, director_enc, genre_enc

def main():
    st.title("영화 수익 예측 앱")

    # CSV 파일 로드 (작업 경로 내 존재)
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')

    # 데이터 전처리
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)

    # 학습용/테스트용 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 테스트 예측 및 성능 출력
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write(f"테스트 RMSE: {rmse:,.0f}")

    st.header("영화 수익 예측 입력")

    # 사용자 입력 받기
    budget = st.number_input('예산', min_value=0, value=10000000, step=1000000)
    popularity = st.number_input('인기도', min_value=0.0, value=10.0, step=0.1)
    runtime = st.number_input('상영 시간(분)', min_value=1, value=120, step=1)
    director_name = st.text_input('감독 이름')
    genre_name = st.text_input('장르 이름(콤마로 구분)')

    if st.button("예측하기"):
        # 감독, 장르 인코딩 시도
        try:
            director_encoded = director_enc.transform([director_name])[0]
        except ValueError:
            st.warning("감독 이름이 데이터에 없습니다. 기본값으로 처리합니다.")
            director_encoded = 0

        try:
            genre_encoded = genre_enc.transform([genre_name])[0]
        except ValueError:
            st.warning("장르 이름이 데이터에 없습니다. 기본값으로 처리합니다.")
            genre_encoded = 0

        # 입력값으로 데이터프레임 생성
        input_data = pd.DataFrame({
            'budget': [budget],
            'popularity': [popularity],
            'runtime': [runtime],
            'director_encoded': [director_encoded],
            'genres_encoded': [genre_encoded]
        })

        # 예측
        pred_revenue = model.predict(input_data)[0]
        st.success(f"예측 수익: {pred_revenue:,.0f} 달러")

if __name__ == "__main__":
    main()
