import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 데이터 전처리 함수
@st.cache_data
def preprocess_data():
    # TMDB 영화 데이터 로드
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # 'crew' 컬럼에서 감독 이름 추출
    # eval()로 문자열 -> 리스트 변환 후 'job'이 'Director'인 사람 이름 가져오기
    movies['director'] = movies['crew'].apply(
        lambda x: next((i['name'] for i in eval(x) if i['job'] == 'Director'), 'Unknown')
    )

    # 'genres' 컬럼에서 장르 리스트 추출 및 첫번째 장르 선택
    movies['genres'] = movies['genres'].apply(
        lambda x: [i['name'] for i in eval(x)] if pd.notnull(x) else []
    )
    movies['genre'] = movies['genres'].apply(lambda x: x[0] if x else 'Unknown')

    # 필요한 컬럼만 선택하고, 0 이하 값 제외
    movies = movies[['budget', 'director', 'genre', 'runtime', 'revenue']].copy()
    movies = movies[movies['budget'] > 0]
    movies = movies[movies['runtime'] > 0]
    movies = movies[movies['revenue'] > 0]

    # 감독, 장르 문자열을 숫자형으로 변환 (Label Encoding)
    le_director = LabelEncoder()
    movies['director_enc'] = le_director.fit_transform(movies['director'])
    le_genre = LabelEncoder()
    movies['genre_enc'] = le_genre.fit_transform(movies['genre'])
    
    return movies, le_director, le_genre

# 머신러닝 모델 학습 함수
@st.cache_resource
def train_model(data):
    # 독립변수(X), 종속변수(y) 설정
    X = data[['budget', 'director_enc', 'genre_enc', 'runtime']]
    y = data['revenue']

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 랜덤포레스트 회귀 모델 생성 및 학습
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # 테스트 데이터로 예측 후 RMSE 평가
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, rmse, X_train, X_test, y_train, y_test

# 예측 결과를 과거 데이터와 함께 시각화하는 함수
def plot_prediction(movies, prediction, budget, runtime):
    plt.figure(figsize=(10,5))

    # 과거 영화 데이터의 예산-수익 산점도
    plt.scatter(movies['budget'], movies['revenue'], alpha=0.3, label='과거 데이터')

    # 입력한 예산에 대한 예측 결과 빨간 점으로 표시
    plt.scatter(budget, prediction, color='red', s=150, label='예측 결과')

    plt.xlabel('예산 (달러)')
    plt.ylabel('수익 (달러)')
    plt.title(f'예산 대비 수익 분포 및 예측 결과 (러닝타임: {runtime}분)')
    plt.legend()
    plt.grid(True)
    
    # Streamlit에 그래프 출력
    st.pyplot(plt.gcf())
    plt.close()

def main():
    st.title("🎬 영화 수익 예측 앱 (그래프 + 신뢰도 포함)")

    # 데이터 전처리 및 레이블 인코더 로드
    movies, le_director, le_genre = preprocess_data()

    # 머신러닝 모델 학습 및 RMSE 확인
    model, rmse, X_train, X_test, y_train, y_test = train_model(movies)

    # 사용자 입력 UI
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input(
            "예산 (달러)", min_value=0, max_value=500_000_000,
            value=50_000_000, step=1_000_000, format="%d"
        )
        runtime = st.slider("러닝타임 (분)", min_value=30, max_value=300, value=120)
    with col2:
        director = st.selectbox("감독", options=sorted(le_director.classes_))
        genre = st.selectbox("장르", options=sorted(le_genre.classes_))

    # 예측 버튼
    if st.button("수익 예측 실행"):
        if director not in le_director.classes_ or genre not in le_genre.classes_:
            st.error("감독 또는 장르 정보가 올바르지 않습니다.")
            return
        
        # 입력값 인코딩
        director_enc = le_director.transform([director])[0]
        genre_enc = le_genre.transform([genre])[0]

        # 모델 입력 데이터 프레임 생성
        X_new = pd.DataFrame({
            'budget': [budget],
            'director_enc': [director_enc],
            'genre_enc': [genre_enc],
            'runtime': [runtime]
        })

        # 예측값 계산
        prediction = model.predict(X_new)[0]

        # 랜덤포레스트 각 트리별 예측값으로 신뢰도 계산 (표준편차)
        preds_per_tree = np.array([tree.predict(X_new)[0] for tree in model.estimators_])
        std_dev = preds_per_tree.std()

        # 예측 결과와 신뢰도 출력
        st.success(f"예측 수익: ${prediction:,.0f} 달러")
        st.info(f"예측 신뢰도 (표준편차): ±${std_dev:,.0f} 달러")

        # 예측 결과 시각화
        plot_prediction(movies, prediction, budget, runtime)

    # 하단에 모델 성능 정보 출력
    st.markdown("---")
    st.info(f"모델 테스트 세트 RMSE: ${rmse:,.0f} 달러")

if __name__ == "__main__":
    main()
