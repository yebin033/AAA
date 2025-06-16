import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import ast

# 감독 이름을 crew JSON 문자열에서 추출하는 함수
def extract_director(crew_str):
    crew = ast.literal_eval(crew_str)  # 문자열 -> 리스트 변환
    for member in crew:
        if member.get('job') == 'Director':  # 감독 찾기
            return member.get('name')
    return 'Unknown'  # 없으면 Unknown 처리

# 장르 리스트를 genres JSON 문자열에서 추출하는 함수
def extract_genres(genres_str):
    genres = ast.literal_eval(genres_str)  # 문자열 -> 리스트 변환
    return [genre['name'] for genre in genres]  # 장르 이름 리스트 반환

# 데이터 전처리 함수: 필요한 컬럼 선택, 인코딩, 결합
def preprocess_data(movies, credits):
    # credits 데이터프레임 컬럼명 변경 (movie_id -> id)
    credits = credits.rename(columns={'movie_id': 'id'})

    # 감독 이름 추출하여 새로운 컬럼 'director' 생성
    credits['director'] = credits['crew'].apply(extract_director)

    # movies와 credits 합치기 (id 기준)
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 장르 리스트를 문자열로 변환 (콤마로 연결)
    df['genres'] = df['genres'].apply(lambda x: ','.join(extract_genres(x)))

    # 사용할 컬럼만 선택하고 결측치 제거
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()

    # 감독명과 장르명을 숫자로 인코딩
    director_enc = LabelEncoder()
    genre_enc = LabelEncoder()
    df['director_encoded'] = director_enc.fit_transform(df['director'])
    df['genres_encoded'] = genre_enc.fit_transform(df['genres'])

    # 특성(feature)와 타깃(target) 분리
    X = df[['budget', 'popularity', 'runtime', 'director_encoded', 'genres_encoded']]
    y = df['revenue']

    return X, y, director_enc, genre_enc

def main():
    st.set_page_config(page_title="영화 수익 예측 앱", page_icon="🎬", layout="centered")
    st.title("🎥 영화 수익 예측 앱")
    st.write("영화의 예산, 인기도, 상영 시간, 감독, 장르 정보를 입력하면 수익을 예측합니다.")

    # 데이터 불러오기 (사이즈 줄인 버전으로 가정)
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')

    # 데이터 전처리 및 인코더 반환
    X, y, director_enc, genre_enc = preprocess_data(movies, credits)

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 학습
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터로 예측 및 RMSE 계산
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    st.write(f"🔍 테스트 데이터 RMSE (오차의 평균 크기): **{rmse:,.0f} 달러**")

    # 사용자 입력 폼
    st.header("영화 정보 입력")
    budget = st.number_input('예산 (달러 단위)', min_value=0, value=10000000, step=1000000, help="영화 제작에 사용된 예산")
    popularity = st.number_input('인기도 지수', min_value=0.0, value=10.0, step=0.1, help="영화의 TMDB 인기 점수")
    runtime = st.number_input('상영 시간 (분)', min_value=1, value=120, step=1, help="영화의 총 상영 시간")
    director_name = st.text_input('감독 이름', help="감독 이름을 정확히 입력해주세요")
    genre_name = st.text_input('장르 이름 (콤마로 구분)', help="예: Action,Adventure,Fantasy")

    # 예측 버튼 클릭 시
    if st.button("🎯 예측하기"):
        # 감독 이름 인코딩 시도, 실패 시 경고 및 기본값(0) 대입
        try:
            director_encoded = director_enc.transform([director_name])[0]
        except ValueError:
            st.warning("감독 이름이 데이터에 없습니다. 기본값으로 처리합니다.")
            director_encoded = 0

        # 장르 이름 인코딩 시도, 실패 시 경고 및 기본값(0) 대입
        try:
            genre_encoded = genre_enc.transform([genre_name])[0]
        except ValueError:
            st.warning("장르 이름이 데이터에 없습니다. 기본값으로 처리합니다.")
            genre_encoded = 0

        # 입력값을 데이터프레임으로 만들어 예측
        input_data = pd.DataFrame({
            'budget': [budget],
            'popularity': [popularity],
            'runtime': [runtime],
            'director_encoded': [director_encoded],
            'genres_encoded': [genre_encoded]
        })

        pred_revenue = model.predict(input_data)[0]
        st.success(f"예측된 영화 수익은 **{pred_revenue:,.0f} 달러** 입니다! 🎉")

if __name__ == "__main__":
    main()
