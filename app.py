import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

def preprocess_data(movies, credits):
    # 크레딧 데이터에서 감독 이름 추출
    credits['director'] = credits['crew'].apply(
        lambda x: next((person['name'] for person in eval(x) if person['job'] == 'Director'), 'Unknown')
    )
    
    # movies와 credits 병합 (id 기준)
    df = pd.merge(movies, credits[['id', 'director']], on='id')
    
    # 필요한 컬럼만 추출
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()
    
    # genres 컬럼에서 첫 번째 장르만 추출 (장르가 리스트 형식임)
    df['genres'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')

    # 예산, 인기, 상영시간 양수 필터링
    df = df[(df['budget'] > 0) & (df['popularity'] > 0) & (df['runtime'] > 0)]
    
    # 상위 20명의 감독만 유지, 나머지는 'Other'로
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')
    
    return df

def build_model(df):
    X = df[['budget', 'popularity', 'runtime', 'director', 'genres']]
    y = df['revenue']
    
    # 전처리 - 범주형 변수 OneHotEncoding
    categorical_features = ['director', 'genres']
    numeric_features = ['budget', 'popularity', 'runtime']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2

def main():
    st.title('TMDB 영화 수익 예측 앱')
    
    movies, credits = load_data()
    df = preprocess_data(movies, credits)
    
    model, rmse, r2 = build_model(df)
    
    st.write(f'모델 평가: RMSE = {rmse:.2f}, R^2 = {r2:.2f}')
    
    st.header('영화 정보 입력')
    budget = st.number_input('예산', min_value=1000, max_value=1_000_000_000, value=1_000_000, step=1000)
    popularity = st.number_input('인기도', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    runtime = st.number_input('상영 시간(분)', min_value=1, max_value=500, value=120)
    director_list = ['Other'] + sorted(df['director'].unique().tolist())
    director = st.selectbox('감독', director_list)
    genre_list = sorted(df['genres'].unique().tolist())
    genre = st.selectbox('장르', genre_list)
    
    if st.button('수익 예측'):
        input_df = pd.DataFrame({
            'budget': [budget],
            'popularity': [popularity],
            'runtime': [runtime],
            'director': [director],
            'genres': [genre]
        })
        prediction = model.predict(input_df)[0]
        st.success(f'예상 수익: ${prediction:,.0f}')
        st.info('참고: 이 예측은 학습된 데이터에 기반한 모델의 추정치입니다.')
    
if __name__ == '__main__':
    main()
