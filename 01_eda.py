"""
01_eda.py
Netflix Retention Modeling - Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 그래프 저장 폴더 생성
os.makedirs('outputs', exist_ok=True)

BG   = '#F8F8F8'
RED  = '#E50914'
DARK = '#221F1F'
GRAY = '#AAAAAA'

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
df = pd.read_csv('watch_joined.csv', parse_dates=['watch_date'])

print("=" * 50)
print("1. 기본 정보")
print("=" * 50)
print(f"행 수    : {len(df):,}")
print(f"컬럼 수  : {df.shape[1]}")
print(f"유저 수  : {df['user_id'].nunique():,}")
print(f"콘텐츠 수: {df['movie_id'].nunique():,}")
print(f"기간     : {df['watch_date'].min().date()} ~ {df['watch_date'].max().date()}")

# ─────────────────────────────────────────
# 2. 결측치 확인
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 결측치 확인")
print("=" * 50)
missing     = df.isnull().sum()
missing_pct = missing / len(df) * 100
result      = pd.DataFrame({
    '결측 수': missing,
    '비율(%)': missing_pct.round(1)
})
missing_df = result[result['결측 수'] > 0]
print(missing_df)

# ─────────────────────────────────────────
# 3. 중복 컬럼 확인
#    completion_rate = progress_percentage / 100
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 중복 컬럼 확인")
print("=" * 50)
both      = df.dropna(subset=['completion_rate', 'progress_percentage'])
match_pct = (
    both['completion_rate'].round(3) ==
    (both['progress_percentage'] / 100).round(3)
).mean() * 100
print(f"completion_rate == progress_percentage/100: {match_pct:.1f}% 일치")
print("→ progress_percentage DROP 결정")

# ─────────────────────────────────────────
# 4. 이상값 확인 (ratio = watch_duration / duration)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("4. 이상값 확인 (watch_duration / duration_minutes)")
print("=" * 50)
df_valid          = df.dropna(subset=['watch_duration_minutes', 'duration_minutes']).copy()
df_valid['ratio'] = df_valid['watch_duration_minutes'] / df_valid['duration_minutes']

total         = len(df)
missing_ratio = total - len(df_valid)
under1        = (df_valid['ratio'] < 1).sum()
one_to3       = ((df_valid['ratio'] >= 1) & (df_valid['ratio'] < 3)).sum()
over3         = (df_valid['ratio'] >= 3).sum()

print(f"전체 세션              : {total:,} (100.0%)")
print(f"결측 (ratio 계산 불가) : {missing_ratio:,} ({missing_ratio/total*100:.1f}%)")
print(f"ratio < 1  → 정상     : {under1:,} ({under1/total*100:.1f}%)")
print(f"ratio 1~3  → 유지     : {one_to3:,} ({one_to3/total*100:.1f}%)")
print(f"ratio >= 3 → 제거     : {over3:,} ({over3/total*100:.1f}%)")

before     = set(df['user_id'].unique())
after      = set(df_valid[df_valid['ratio'] < 3]['user_id'].unique())
lost_users = before - after
print(f"ratio >= 3 제거 후 모든 세션 잃는 유저: {len(lost_users)}명")

# ─────────────────────────────────────────
# 5. 수치형 컬럼 분포
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("5. 수치형 컬럼 분포")
print("=" * 50)
print(df[['watch_duration_minutes', 'completion_rate', 'imdb_rating']].describe().round(2))

# ─────────────────────────────────────────
# 6. 범주형 컬럼 분포
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("6. 범주형 컬럼 분포")
print("=" * 50)
for col in ['device_type', 'action', 'genre_primary', 'content_type']:
    print(f"\n[{col}]")
    print(df[col].value_counts().to_string())

# ─────────────────────────────────────────
# 7. 유저별 세션 수 분포
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("7. 유저별 세션 수 분포")
print("=" * 50)
user_sessions = df.groupby('user_id').size()
print(user_sessions.describe().round(1))

# ─────────────────────────────────────────
# 8. Churn 레이블 미리보기
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("8. Churn 레이블 미리보기 (30일 기준)")
print("=" * 50)
reference_date = df['watch_date'].max()
last_watch     = df.groupby('user_id')['watch_date'].max()
recency_days   = (reference_date - last_watch).dt.days
churned        = (recency_days >= 30).astype(int)
print(f"Churned  : {churned.sum():,}명 ({churned.mean()*100:.1f}%)")
print(f"Retained : {(1-churned).sum():,}명 ({(1-churned).mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 9. DROP 컬럼 정리
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("9. DROP 컬럼 정리")
print("=" * 50)
drop_reasons = {
    'progress_percentage': 'completion_rate와 100% 동일 (중복)',
    'watch_ratio'        : 'watch_duration/duration으로 직접 계산 가능 (파생)',
    'user_rating'        : '결측 79.9% (선택 편향)',
    'genre_secondary'    : '결측 64.4%',
    'session_id'         : '식별자, 학습 불가',
}
for col, reason in drop_reasons.items():
    print(f"  {col:<25} → {reason}")

# ─────────────────────────────────────────
# 10. 그래프 저장 → outputs/
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("10. 그래프 저장 → outputs/")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, fontweight='bold', color=DARK, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(colors='#555555', labelsize=8)

# ① 결측치 비율
ax = axes[0]
cols_m = missing_df.index.tolist()
vals_m = missing_df['비율(%)'].tolist()
bar_colors = [RED if v > 50 else '#FF6B6B' if v > 10 else GRAY for v in vals_m]
ax.barh(cols_m, vals_m, color=bar_colors, edgecolor='white', height=0.6)
for i, v in enumerate(vals_m):
    ax.text(v + 0.5, i, f'{v}%', va='center', fontsize=8)
ax.set_xlabel('결측 비율 (%)', fontsize=9)
style(ax, '① 결측치 비율')

# ② ratio 구간별 비중
ax = axes[1]
ratio_labels = ['ratio < 1\n(정상)', 'ratio 1~3\n(유지)', 'ratio ≥ 3\n(제거)', '결측']
ratio_vals   = [
    under1 / total * 100,
    one_to3 / total * 100,
    over3 / total * 100,
    missing_ratio / total * 100
]
ratio_colors = [GRAY, '#FF6B6B', RED, '#CCCCCC']
ax.bar(ratio_labels, ratio_vals, color=ratio_colors, edgecolor='white')
for i, v in enumerate(ratio_vals):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('비율 (%)', fontsize=9)
style(ax, '② watch_duration / duration 비율')

# ③ Churn vs Retained
ax = axes[2]
churn_vals   = [churned.sum(), (1 - churned).sum()]
churn_labels = [
    f'Churned\n{churned.mean()*100:.1f}%',
    f'Retained\n{(1-churned).mean()*100:.1f}%'
]
ax.pie(churn_vals, labels=churn_labels, colors=[RED, DARK],
       startangle=90, textprops={'fontsize': 9, 'color': 'white'},
       wedgeprops={'edgecolor': BG, 'linewidth': 2})
style(ax, '③ Churn vs Retained (30일 기준)')

# ④ 시청 시간 분포
ax = axes[3]
dur = df['watch_duration_minutes'].dropna().clip(0, 300)
ax.hist(dur, bins=40, color=RED, alpha=0.7, edgecolor='white')
ax.axvline(dur.mean(), color=DARK, linestyle='--', linewidth=1.5, label=f'평균 {dur.mean():.0f}분')
ax.set_xlabel('시청 시간 (분)', fontsize=9)
ax.set_ylabel('빈도', fontsize=9)
ax.legend(fontsize=8)
style(ax, '④ 세션별 시청 시간 분포')

# ⑤ 장르별 세션 수
ax = axes[4]
genre = df['genre_primary'].value_counts().head(8)
ax.barh(genre.index[::-1], genre.values[::-1],
        color=[RED if i == len(genre) - 1 else GRAY for i in range(len(genre))],
        edgecolor='white', height=0.6)
ax.set_xlabel('세션 수', fontsize=9)
style(ax, '⑤ 장르별 세션 수 (Top 8)')

# ⑥ 완료율 분포
ax = axes[5]
comp = df['completion_rate'].dropna()
ax.hist(comp, bins=30, color=DARK, alpha=0.7, edgecolor='white')
ax.axvline(comp.mean(), color=RED, linestyle='--', linewidth=1.5, label=f'평균 {comp.mean():.2f}')
ax.set_xlabel('완료율', fontsize=9)
ax.set_ylabel('빈도', fontsize=9)
ax.legend(fontsize=8)
style(ax, '⑥ 완료율 분포')

fig.suptitle('Netflix Retention Modeling — EDA Overview',
             fontsize=14, fontweight='bold', color=DARK)
plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  → outputs/01_eda_overview.png 저장 완료")
