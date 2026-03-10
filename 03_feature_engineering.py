"""
03_feature_engineering.py
Netflix Retention Modeling - Feature Engineering
세션 단위 → 유저 단위 집계
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. 전처리된 데이터 로드
# ─────────────────────────────────────────
df = pd.read_csv('watch_preprocessed.csv', parse_dates=['watch_date'])

print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)
print(f"행 수   : {len(df):,}  (세션 단위)")
print(f"유저 수 : {df['user_id'].nunique():,}")

# ─────────────────────────────────────────
# 2. 유저 단위 집계
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 유저 단위 집계")
print("=" * 50)

reference_date = df['watch_date'].max()

user_features = df.groupby('user_id').agg(
    # 세션 빈도
    total_sessions       = ('watch_date',              'count'),
    # 시청 시간
    total_watch_time     = ('watch_duration_minutes',  'sum'),
    avg_watch_time       = ('watch_duration_minutes',  'mean'),
    # 완료율
    avg_completion_rate  = ('completion_rate',         'mean'),
    # 날짜
    last_watch_date      = ('watch_date',              'max'),
    first_watch_date     = ('watch_date',              'min'),
    # 장르 다양성
    genre_diversity      = ('genre_primary',           'nunique'),
    # 디바이스 다양성
    device_diversity     = ('device_type',             'nunique'),
    # 완료 횟수
    n_completed          = ('action',   lambda x: (x == 'completed').sum()),
    # 영화 시청 횟수
    n_movies             = ('content_type', lambda x: (x == 'Movie').sum()),
    # 넷플릭스 오리지널 시청 횟수
    n_original           = ('is_netflix_original', 'sum'),
    # Churn 레이블
    churned              = ('churned',             'first'),
).reset_index()

# ─────────────────────────────────────────
# 3. 파생 피처 계산
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 파생 피처 계산")
print("=" * 50)

# Recency (마지막 시청 후 경과일)
user_features['recency_days'] = (
    reference_date - user_features['last_watch_date']
).dt.days

# Active days (첫 시청 ~ 마지막 시청 기간)
user_features['active_days'] = (
    user_features['last_watch_date'] - user_features['first_watch_date']
).dt.days + 1

# Session frequency (하루 평균 세션 수)
user_features['session_frequency'] = (
    user_features['total_sessions'] / user_features['active_days']
)

# Completion ratio (완료 세션 비율)
user_features['completion_ratio'] = (
    user_features['n_completed'] / user_features['total_sessions']
)

# Movie ratio (영화 시청 비율)
user_features['movie_ratio'] = (
    user_features['n_movies'] / user_features['total_sessions']
)

# Netflix original ratio (오리지널 시청 비율)
user_features['original_ratio'] = (
    user_features['n_original'] / user_features['total_sessions']
)

# ─────────────────────────────────────────
# 4. 불필요한 중간 컬럼 정리
# ─────────────────────────────────────────
drop_cols = ['last_watch_date', 'first_watch_date', 'n_completed', 'n_movies', 'n_original']
user_features = user_features.drop(columns=drop_cols)

# ─────────────────────────────────────────
# 5. 최종 피처 확인
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("5. 최종 피처 목록")
print("=" * 50)

FEATURE_COLS = [
    'total_sessions',       # 총 세션 수
    'total_watch_time',     # 총 시청 시간
    'avg_watch_time',       # 평균 세션 길이
    'avg_completion_rate',  # 평균 완료율
    'recency_days',         # 마지막 시청 후 경과일
    'active_days',          # 활동 기간
    'session_frequency',    # 하루 평균 세션 수
    'genre_diversity',      # 장르 다양성
    'device_diversity',     # 디바이스 다양성
    'completion_ratio',     # 완료 세션 비율
    'movie_ratio',          # 영화 시청 비율
    'original_ratio',       # 넷플릭스 오리지널 비율
]

for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {col}")

print(f"\n유저 수 : {len(user_features):,}  (유저 단위)")
print(f"피처 수 : {len(FEATURE_COLS)}개")
print(f"Churned : {user_features['churned'].sum():,}명 ({user_features['churned'].mean()*100:.1f}%)")

print("\n" + "=" * 50)
print("6. 피처 기술 통계")
print("=" * 50)
print(user_features[FEATURE_COLS].describe().round(2).to_string())

# ─────────────────────────────────────────
# 7. 장르별 비율 피처 실험
#    → 유의미한 차이 없으면 genre_diversity만 유지
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("7. 장르별 비율 피처 실험 (상관계수 확인)")
print("=" * 50)

# 장르 One-Hot Encoding
genre_dummies  = pd.get_dummies(df['genre_primary'], prefix='genre')
df_with_genre  = pd.concat([df[['user_id']], genre_dummies], axis=1)
genre_cols     = [c for c in df_with_genre.columns if c.startswith('genre_')]

# 유저 단위 장르 비율 계산
genre_features = df_with_genre.groupby('user_id')[genre_cols].mean().reset_index()
genre_features = genre_features.merge(
    user_features[['user_id', 'churned']], on='user_id', how='left'
)

# churn과 상관계수
corr = genre_features[genre_cols].corrwith(
    genre_features['churned']
).abs().sort_values(ascending=False)

print(f"\n{'장르':<25} {'|상관계수|':>10}")
print("-" * 37)
for genre, val in corr.items():
    bar = '█' * int(val * 500)
    print(f"  {genre:<23} {val:.4f}  {bar}")

print(f"\n최대 상관계수 : {corr.max():.4f}")
print(f"평균 상관계수 : {corr.mean():.4f}")
if corr.max() < 0.1:
    print("→ 장르별 비율 피처 유의미한 차이 없음 → genre_diversity만 유지")
else:
    print("→ 일부 장르 피처 유의미 → 추가 검토 필요")

# ─────────────────────────────────────────
# 8. 저장
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("8. 저장")
print("=" * 50)
user_features.to_csv('user_features.csv', index=False)
print("  → user_features.csv 저장 완료")
