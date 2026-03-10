"""
02_preprocessing.py
Netflix Retention Modeling - Data Preprocessing
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
df = pd.read_csv('watch_joined.csv', parse_dates=['watch_date'])

print("=" * 50)
print("1. 원본 데이터")
print("=" * 50)
print(f"행 수   : {len(df):,}")
print(f"컬럼 수 : {df.shape[1]}")

# ─────────────────────────────────────────
# 2. 불필요한 컬럼 DROP
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 불필요한 컬럼 DROP")
print("=" * 50)

drop_cols = [
    'progress_percentage',  # completion_rate와 100% 동일 (중복)
    'watch_ratio',          # 파생 컬럼
    'user_rating',          # 결측 79.9%
    'genre_secondary',      # 결측 64.4%
    'session_id',           # 식별자
]
df = df.drop(columns=drop_cols)
print(f"DROP 완료: {drop_cols}")
print(f"남은 컬럼 수: {df.shape[1]}")

# ─────────────────────────────────────────
# 3. 이상값 제거 (ratio >= 3)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 이상값 제거 (watch_duration / duration >= 3)")
print("=" * 50)

before = len(df)
df_valid = df.dropna(subset=['watch_duration_minutes', 'duration_minutes']).copy()
df_valid['ratio'] = df_valid['watch_duration_minutes'] / df_valid['duration_minutes']
remove_idx = df_valid[df_valid['ratio'] >= 3].index
df = df.drop(index=remove_idx)
df = df.drop(columns=['ratio'], errors='ignore')

after = len(df)
print(f"제거 전: {before:,}행")
print(f"제거 후: {after:,}행  ({before-after:,}행 제거)")

# ─────────────────────────────────────────
# 4. 결측치 처리 (중앙값 대체)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("4. 결측치 처리 (중앙값 대체)")
print("=" * 50)

for col in ['completion_rate', 'watch_duration_minutes']:
    n_missing = df[col].isnull().sum()
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: {n_missing:,}개 결측 → 중앙값 {median_val:.2f}로 대체")

# ─────────────────────────────────────────
# 5. Churn 레이블 생성 (30일 기준)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("5. Churn 레이블 생성 (30일 기준)")
print("=" * 50)

reference_date = df['watch_date'].max()
last_watch     = df.groupby('user_id')['watch_date'].max().reset_index()
last_watch.columns = ['user_id', 'last_watch_date']
last_watch['recency_days'] = (reference_date - last_watch['last_watch_date']).dt.days
last_watch['churned']      = (last_watch['recency_days'] >= 30).astype(int)

df = df.merge(last_watch[['user_id', 'last_watch_date', 'recency_days', 'churned']], on='user_id', how='left')

print(f"기준일  : {reference_date.date()}")
print(f"Churned : {last_watch['churned'].sum():,}명 ({last_watch['churned'].mean()*100:.1f}%)")
print(f"Retained: {(1-last_watch['churned']).sum():,}명 ({(1-last_watch['churned']).mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 6. 최종 확인
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("6. 전처리 완료 후 최종 확인")
print("=" * 50)
print(f"행 수   : {len(df):,}")
print(f"컬럼 수 : {df.shape[1]}")
print(f"잔존 결측치:")
missing = df.isnull().sum()
remaining = missing[missing > 0]
if len(remaining) == 0:
    print("  없음 ✅")
else:
    print(remaining)

# ─────────────────────────────────────────
# 7. 전처리된 데이터 저장
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("7. 저장")
print("=" * 50)
df.to_csv('watch_preprocessed.csv', index=False)
print("  → watch_preprocessed.csv 저장 완료")
