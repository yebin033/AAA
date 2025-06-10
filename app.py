import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 데이터 로드
def load_data():
    movies = pd.read_csv('tmdb_5000_movies_small.csv')
    credits = pd.read_csv('tmdb_5000_credits_small.csv')
    return movies, credits

# 감독 이름 추출 함수 (crew json string -> director name)
def get_director(crew_str):
    try:
        crew = eval(crew_str)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
    except:
        pass
    return 'Unknown'

# 데이터 전처리
def preprocess_data(movies, credits):
    # credits에서 director 컬럼 생성
    credits['director'] = credits['crew'].apply(get_director)

    # movies와 credits 합치기 (id 기준)
    df = pd.merge(movies, credits[['id', 'director']], on='id')

    # 필요한 컬럼만 추출하고 결측치 제거
    df = df[['budget', 'popularity', 'runtime', 'director', 'genres', 'revenue']].dropna()

    # genres 컬럼은 리스트 형태 json string -> 첫 번째 장르 이름으로 변환
    def get_first_genre(genres_str):
        try:
            genres = eval(genres_str)
            if len(genres) > 0:
                return genres[0]['name']
        except:
            pass
        return 'Unknown'

    df['genres'] = df['genres'].apply(get_first_genre)

    # budget, popularity, runtime이 0 이하인 행 제거
    df = df[(df['budget'] > 0) & (df['popularity'] > 0) & (df['runtime'] > 0)]

    # 감독 수가 적은 경우 'Other'로 통합 (상위 20명만 남김)
    top_directors = df['director'].value_counts().head(20).index
    df['director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')

    return df

# 모델 학습 및 평가
def train_model(df):
    X = df[['budget', 'popularity', 'runtime', 'director', 'genres']]
    y = df['revenue']

    # 범주형 변수 인코딩
    categorical_cols = ['director', 'genres']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_cat = encoder.fit_transform(X[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

    # 수치형 변수와 합치기
    X_num = X[['budget', 'popularity', 'runtime']]
    X_final = pd.concat([X_num, X_cat_df], axis=1)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    return model, encoder, X_final.columns

# 새로운 입력으로 수익 예측 함수
def predict_revenue(model, encoder, feature_columns, budget, popularity, runtime, director, genre):
    input_df = pd.DataFrame({
        'budget': [budget],
        'popularity': [popularity],
        'runtime': [runtime],
        'director': [director],
        'genres': [genre]
    })

    # 인코딩
    X_cat = encoder.transform(input_df[['director', 'genres']])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(['director', 'genres']))

    X_num = input_df[['budget', 'popularity', 'runtime']]

    X_final = pd.concat([X_num, X_cat_df], axis=1)

    # 없는 컬럼을 0으로 채우기 (학습시와 같은 피처셋 맞추기 위해)
    for col in feature_columns:
        if col not in X_final.columns:
            X_final[col] = 0

    X_final = X_final[feature_columns]

    prediction = model.predict(X_final)[0]
    return prediction

def main():
    movies, credits = load_data()
    df = preprocess_data(movies, credits)
    model, encoder, feature_columns = train_model(df)

    # 예측 예시
    example_revenue = predict_revenue(
        model, encoder, feature_columns,
        budget=1_000_000,
        popularity=10,
        runtime=120,
        director='Steven Spielberg',
        genre='Action'
    )

    print(f"예측 수익: ${example_revenue:,.0f}")

if __name__ == "__main__":
    main()
