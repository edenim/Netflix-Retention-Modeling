"""
00_data_preparation.py
Netflix Retention Modeling - Data Preparation
watch_history.csv + movies.csv → watch_joined.csv
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)

watch  = pd.read_csv('watch_history.csv', parse_dates=['watch_date'])
movies = pd.read_csv('movies.csv')

print(f"watch_history : {len(watch):,}행 / {watch.shape[1]}컬럼")
print(f"movies        : {len(movies):,}행 / {movies.shape[1]}컬럼")

# ─────────────────────────────────────────
# 2. movies 중복 제거
#    movie_id가 중복된 경우 첫 번째 행만 유지
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. movies 중복 제거")
print("=" * 50)

before = len(movies)
movies = movies.drop_duplicates(subset=['movie_id'], keep='first')
after  = len(movies)
print(f"중복 제거 전 : {before:,}행")
print(f"중복 제거 후 : {after:,}행  ({before-after:,}개 제거)")

# ─────────────────────────────────────────
# 3. watch_history + movies 조인
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("3. watch_history + movies 조인 (LEFT JOIN on movie_id)")
print("=" * 50)

# 불필요한 컬럼 제거 (movies에서)
movies_drop = [
    'production_budget',
    'box_office_revenue',
    'number_of_seasons',
    'number_of_episodes',
    'added_to_platform',
    'content_warning',
]
movies = movies.drop(columns=movies_drop)

df = watch.merge(movies, on='movie_id', how='left')
print(f"조인 결과 : {len(df):,}행 / {df.shape[1]}컬럼")

# ─────────────────────────────────────────
# 4. 파생 컬럼 추가
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("4. 파생 컬럼 추가")
print("=" * 50)

# completion_rate = progress_percentage / 100
df['completion_rate'] = df['progress_percentage'] / 100
print("  completion_rate = progress_percentage / 100  ✅")

# watch_ratio = watch_duration / duration_minutes (최대 3.0으로 cap)
df['watch_ratio'] = df['watch_duration_minutes'] / df['duration_minutes']
df['watch_ratio']  = df['watch_ratio'].clip(upper=3.0)
print("  watch_ratio = watch_duration / duration (cap 3.0)  ✅")

# ─────────────────────────────────────────
# 5. 최종 확인
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("5. 최종 확인")
print("=" * 50)
print(f"행 수   : {len(df):,}")
print(f"컬럼 수 : {df.shape[1]}")
print(f"컬럼 목록: {df.columns.tolist()}")
print(f"\n기간    : {df['watch_date'].min().date()} ~ {df['watch_date'].max().date()}")
print(f"유저 수 : {df['user_id'].nunique():,}")
print(f"콘텐츠 수: {df['movie_id'].nunique():,}")

# ─────────────────────────────────────────
# 6. 저장
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("6. 저장")
print("=" * 50)
df.to_csv('watch_joined.csv', index=False)
print("  → watch_joined.csv 저장 완료")
