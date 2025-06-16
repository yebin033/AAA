import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import ast

# 감독 이름을 crew JSON 문자열에서 추출하는 함수
def extract_director(crew_str):
    crew = ast.literal_eval(crew_str)  # 문자열을 리스트로 변환
    for member in crew:
        if member.get('job') == 'Director':  # 감독 역할 찾기
            return member.get('name')  # 감독 이름 반환
    return 'Unknown'  # 감독 정보 없으면 'Unknown'

# 장르 리스트를 genres JSON 문자열에서 추출하는 함수
def extract_genres(genres_str):
    genres = ast.literal_eval(genres_str)  # 문자열을 리스트로 변환
    return [genre['name'] for genre in genres]  # 장르 이름 리스트 반환

# 데이터 전처리 함수: movies, credits 합치고 인코딩 수행
def preprocess_data(movies, credits):
    credits = credits.rename(columns={'movie_id': 'id'})  # 영화 id 컬럼명 통일
    credits['director'] = credits['crew'].apply(extract_director)  # 감독 이름 추출

    # movies와 credits 데이터프레임 합침 (id 기준)
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # genres 컬럼에서 장르 이름들을 ,로 연결한 문자열로 변환
    df['genres'] = df['genres'].apply(lambda x: ','.join(extract_genres(x)))

    # 예측에 필요한 컬럼만 선택하고 결측값 제거
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()

    # 감독명과 장르명을 숫자 레이블로 변환 (머신러닝 입력용)
    director_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    df['director_encoded'] = director_enc.fit_transform(df['director'])
    df['genres_encoded'] = genre_enc.fit_transform(df['genres'])

    # 특성(입력) 변수와 목표 변수 분리
    X = df[['budget', 'popularity', 'runtime', 'director_encoded', 'genres_encoded']]
    y = df['revenue']

    return df, X, y, director_enc, genre_enc

def main():
    # Streamlit 앱 기본 설정
    st.set_page_config(page_title="영화 수익 예측 앱", page_icon="🎬", layout="centered")
    st.title("🎥 영화 수익 예측 앱")
    st.write("영화의 예산, 인기도, 상영 시간, 감독, 장르 정보를 선택하면 수익을 예측합니다.")

    # 데이터 파일 읽기 (크기 줄인 버전 사용 권장)
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')

    # 데이터 전처리 및 인코더 생성
    df, X, y, director_enc, genre_enc = preprocess_data(movies, credits)

    # 학습용, 테스트용 데이터 분리 (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 회귀 모델 학습
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터 예측 및 RMSE(평균 오차) 계산
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    st.write(f"🔍 테스트 데이터 RMSE: **{rmse:,.0f} 달러**")

    st.header("영화 정보 입력")

    # 예산, 인기도, 상영 시간 숫자 입력
    budget = st.number_input('예산 (달러)', min_value=0, value=10000000, step=1000000)
    popularity = st.number_input('인기도', min_value=0.0, value=10.0, step=0.1)
    runtime = st.number_input('상영 시간 (분)', min_value=1, value=120, step=1)

    # 감독 선택: 데이터 내 고유 감독 목록에서 선택
    directors = sorted(df['director'].unique())
    director_name = st.selectbox('감독 선택', directors)

    # 장르 선택: 데이터 내 고유 장르 목록에서 선택
    genres = sorted(df['genres'].unique())
    genre_name = st.selectbox('장르 선택', genres)

    if st.button("🎯 예측하기"):
        # 선택된 감독과 장르를 인코딩된 숫자로 변환
        director_encoded = director_enc.transform([director_name])[0]
        genre_encoded = genre_enc.transform([genre_name])[0]

        # 모델에 입력할 데이터 프레임 생성
        input_data = pd.DataFrame({
            'budget': [budget],
            'popularity': [popularity],
            'runtime': [runtime],
            'director_encoded': [director_encoded],
            'genres_encoded': [genre_encoded]
        })

        # 영화 수익 예측
        pred_revenue = model.predict(input_data)[0]
        st.success(f"예측된 영화 수익: **{pred_revenue:,.0f} 달러** 🎉")

if __name__ == "__main__":
    main()
