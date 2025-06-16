import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 데이터 로드 함수
def load_data():
    # 실제 환경에 맞게 파일 경로 조정 필요
    movies = pd.read_csv("tmdb_5000_movies_small.csv")
    credits = pd.read_csv("tmdb_5000_credits_small.csv")
    return movies, credits

# 전처리 및 피처 엔지니어링 함수
def preprocess_data(movies, credits):
    # 감독 이름 추출 (예: credits에 감독 정보가 있을 경우)
    # 여기서는 movies에 'director' 컬럼이 있다고 가정
    # 없으면 credits에서 처리 필요
    if 'director' not in movies.columns:
        # 간단 예시: credits에서 감독 추출 후 movies에 합치기
        # 실제 구조에 맞게 수정 필요
        director_dict = {}
        for idx, row in credits.iterrows():
            # credits의 crew 컬럼은 json 문자열 형태일 가능성 있음
            # 필요시 json.loads 처리 후 감독 추출 로직 추가
            pass
        # movies['director'] = movies['id'].map(director_dict)

    # 필요한 컬럼만 추출
    df = movies[['title', 'budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'revenue', 'director']].copy()

    # 결측치 처리: runtime, revenue, budget 등
    df = df.dropna(subset=['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'revenue', 'director'])

    # 감독 라벨 인코딩
    le = LabelEncoder()
    df['director_encoded'] = le.fit_transform(df['director'])

    return df, le

# 머신러닝 모델 학습 및 예측 함수
def train_model(df):
    # 설명 변수와 타깃 변수 분리
    X = df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'director_encoded']]
    y = df['revenue']

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse

# 선택한 영화 정보로 예측 입력 데이터 만들기
def make_features(movie, le):
    # 감독 인코딩 처리
    director_enc = -1
    if movie['director'] in le.classes_:
        director_enc = le.transform([movie['director']])[0]

    features = pd.DataFrame({
        'budget': [movie['budget']],
        'popularity': [movie['popularity']],
        'runtime': [movie['runtime']],
        'vote_average': [movie['vote_average']],
        'vote_count': [movie['vote_count']],
        'director_encoded': [director_enc]
    })
    return features

def main():
    st.set_page_config(page_title="영화 수익 예측 대시보드", layout="wide")
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("탭 선택", ["데이터 요약", "수익 예측", "감독 정보"])

    # 데이터 로드
    movies, credits = load_data()

    # 전처리 및 라벨 인코더 준비
    df, le = preprocess_data(movies, credits)

    # 데이터 요약 탭
    if page == "데이터 요약":
        st.header("영화 데이터 요약 및 시각화")

        # 감독별 평균 수익 계산
        director_revenue = df.groupby("director")["revenue"].mean().sort_values(ascending=False).head(10)

        # Plotly 막대그래프
        fig = px.bar(director_revenue,
                     x=director_revenue.index,
                     y=director_revenue.values,
                     labels={"x": "감독", "y": "평균 수익"},
                     title="상위 10명 감독의 평균 수익")
        st.plotly_chart(fig, use_container_width=True)

    # 수익 예측 탭
    elif page == "수익 예측":
        st.header("영화 수익 예측")

        # 머신러닝 모델 학습 (간단히 매번 재학습, 실 서비스면 캐싱 등 필요)
        with st.spinner("모델 학습 중... 잠시만 기다려 주세요"):
            model, rmse = train_model(df)

        st.write(f"모델 학습 완료. RMSE: {rmse:,.0f} 달러")

        # 영화 선택
        movie_title = st.selectbox("예측할 영화 선택", df["title"].tolist())

        # 선택 영화 데이터
        movie = df[df["title"] == movie_title].iloc[0]

        st.write("선택한 영화 정보:")
        st.write(movie[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'director']])

        # 예측 실행 버튼
        if st.button("수익 예측 실행"):
            X_pred = make_features(movie, le)
            pred_revenue = model.predict(X_pred)[0]
            st.success(f"예측된 수익: {pred_revenue:,.0f} 달러")

    # 감독 정보 탭
    elif page == "감독 정보":
        st.header("감독 상세 정보")

        director_list = df["director"].dropna().unique()
        selected_director = st.selectbox("감독 선택", director_list)

        st.write(f"선택된 감독: {selected_director}")

        director_movies = df[df["director"] == selected_director]
        st.write("작품 목록:")
        st.dataframe(director_movies[["title", "release_date", "revenue"]])

        avg_revenue = director_movies["revenue"].mean()
        st.write(f"평균 수익: {avg_revenue:,.0f} 달러")

if __name__ == "__main__":
    main()
