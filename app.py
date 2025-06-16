import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    # 데이터 로드 함수
    # 파일 경로는 실제 환경에 맞게 조정 필요
    movies = pd.read_csv("tmdb_5000_movies_small.csv")
    credits = pd.read_csv("tmdb_5000_credits_small.csv")
    return movies, credits

def main():
    # 페이지 기본 설정
    st.set_page_config(page_title="영화 수익 예측 대시보드", layout="wide")

    # 사이드바에 메뉴 구성
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("탭 선택", ["데이터 요약", "수익 예측", "감독 정보"])

    # 데이터 로드 (최초 한 번만)
    movies, credits = load_data()

    # 데이터 요약 및 시각화 탭
    if page == "데이터 요약":
        st.header("영화 데이터 요약 및 시각화")

        # 감독별 평균 수익 계산
        # 여기서 'director' 컬럼이 movies 데이터에 있다고 가정
        # 실제 데이터에 맞게 컬럼명 확인 및 수정 필요
        if "director" not in movies.columns or "revenue" not in movies.columns:
            st.error("movies 데이터에 'director' 또는 'revenue' 컬럼이 없습니다.")
            return

        director_revenue = movies.groupby("director")["revenue"].mean().sort_values(ascending=False).head(10)

        # Plotly로 막대 그래프 생성
        fig = px.bar(director_revenue,
                     x=director_revenue.index,
                     y=director_revenue.values,
                     labels={"x":"감독", "y":"평균 수익"},
                     title="상위 10명 감독의 평균 수익")
        st.plotly_chart(fig, use_container_width=True)

    # 수익 예측 탭
    elif page == "수익 예측":
        st.header("영화 수익 예측")

        # 영화 제목 선택 박스
        movie_title = st.selectbox("예측할 영화 선택", movies["title"].tolist())
        st.write("선택한 영화:", movie_title)

        # 예측 실행 버튼
        if st.button("수익 예측 실행"):
            # 실제 모델 코드는 여기서 호출하거나 작성해야 함
            st.info("수익 예측 모델을 실행합니다... (구현 예정)")

    # 감독 정보 탭
    elif page == "감독 정보":
        st.header("감독 상세 정보")

        # 감독 리스트에서 선택
        if "director" not in movies.columns:
            st.error("movies 데이터에 'director' 컬럼이 없습니다.")
            return

        director_list = movies["director"].dropna().unique()
        selected_director = st.selectbox("감독 선택", director_list)

        st.write(f"선택된 감독: {selected_director}")

        # 선택 감독 작품 목록과 평균 수익 계산
        director_movies = movies[movies["director"] == selected_director]
        st.write("작품 목록:")
        st.dataframe(director_movies[["title", "release_date", "revenue"]])

        avg_revenue = director_movies["revenue"].mean()
        st.write(f"평균 수익: {avg_revenue:,.0f} 달러")

if __name__ == "__main__":
    main()
