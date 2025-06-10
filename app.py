import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1) 모델 및 인코더 불러오기
model = joblib.load('model.pkl')
director_encoder = joblib.load('director_encoder.pkl')
top_actors = joblib.load('top_actors.pkl')
all_genres = joblib.load('all_genres.pkl')

st.title("영화 흥행 수익 예측 (광고 마케팅용)")

# 2) 사용자 입력
budget = st.number_input("제작비 (USD)", min_value=0, value=50000000, step=1000000)
popularity = st.number_input("인기 점수", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
runtime = st.number_input("러닝타임 (분)", min_value=1, max_value=500, value=120)

genres = st.multiselect("장르 선택", options=all_genres)
director = st.selectbox("감독 선택", options=list(director_encoder.categories_[0]))
actors = st.multiselect("대표 배우 선택 (최대 3명)", options=top_actors)

# 3) 입력 데이터 전처리
input_data = {}

input_data['budget'] = budget
input_data['popularity'] = popularity
input_data['runtime'] = runtime

# 장르 멀티핫 인코딩
for genre in all_genres:
    input_data[f'genre_{genre}'] = 1 if genre in genres else 0

# 감독 원핫 인코딩
director_array = director_encoder.transform([[director]])[0]
for i, col in enumerate(director_encoder.categories_[0]):
    input_data[f'director_{col}'] = director_array[i]

# 배우 원핫 인코딩
for actor in top_actors:
    input_data[f'actor_{actor}'] = 1 if actor in actors else 0

X_input = pd.DataFrame([input_data])

# 4) 예측 및 출력
prediction = model.predict(X_input)[0]
st.success(f"예상 영화 수익: ${prediction:,.0f}")

st.write("※ 실제 데이터와 차이가 있을 수 있으며, 참고용 예측입니다.")
