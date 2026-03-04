# 🎬 Netflix Retention Modeling

> Predicting user churn from behavioral patterns using Logistic Regression and Random Forest

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat)

---

## Overview

Netflix loses subscribers when engagement drops — but churn often becomes visible too late. This project builds a **behavioral churn prediction model** using session-level watch history to identify at-risk users before they leave.

Key questions addressed:
- Which engagement signals best predict churn?
- How early can we detect declining retention?
- Can simple models (Logistic Regression) compete with ensemble methods (Random Forest)?

---

## Dataset

- **Source**: [Kaggle](https://www.kaggle.com/) — Netflix user watch history
- **Size**: 105,000 sessions across 10,000 users
- **Period**: Jan 2024 – Dec 2025
- **Key columns**: `user_id`, `watch_date`, `watch_duration_minutes`, `completion_rate`, `action`, `genre_primary`, `content_type`, `device_type`

---

## Methodology

### 1. Data Cleaning & Column Selection
Removed redundant and low-quality columns:
- `progress_percentage` — exact duplicate of `completion_rate` (correlation = 1.0)
- `watch_ratio` — derived column computable from existing features
- `user_rating` — 79.9% missing, high selection bias
- `session_id` — identifier only

### 2. Churn Label Definition
A user is labeled **churned = 1** if their last watch activity was **30+ days** before the reference date (2025-12-31).

```python
reference_date = df['watch_date'].max()
last_watch = df.groupby('user_id')['watch_date'].max()
recency_days = (reference_date - last_watch).dt.days
churned = (recency_days >= 30).astype(int)
# Result: 65.4% churned / 34.6% retained
```

### 3. Feature Engineering
Aggregated session-level data to **user-level** (1 row per user):

| Feature | Description |
|---|---|
| `total_sessions` | Total number of watch sessions |
| `total_watch_time` | Cumulative watch time (minutes) |
| `avg_watch_time` | Average session duration (minutes) |
| `avg_completion_rate` | Mean content completion rate (0–1) |
| `recency_days` | Days since last watch activity |
| `active_days` | Days between first and last watch |
| `session_frequency` | Sessions per active day |
| `genre_diversity` | Number of distinct genres watched |
| `completion_ratio` | Proportion of fully completed sessions |
| `movie_ratio` | Share of Movie vs TV Series sessions |
| `device_diversity` | Number of distinct devices used |

```python
user_features = df.groupby('user_id').agg(
    total_sessions       = ('session_id', 'count'),
    total_watch_time     = ('watch_duration_minutes', 'sum'),
    avg_watch_time       = ('watch_duration_minutes', 'mean'),
    avg_completion_rate  = ('completion_rate', 'mean'),
    last_watch_date      = ('watch_date', 'max'),
    first_watch_date     = ('watch_date', 'min'),
    genre_diversity      = ('genre_primary', 'nunique'),
    n_movies             = ('content_type', lambda x: (x == 'Movie').sum()),
    n_completed          = ('action', lambda x: (x == 'completed').sum()),
    device_diversity     = ('device_type', 'nunique'),
).reset_index()

user_features['recency_days']      = (reference_date - user_features['last_watch_date']).dt.days
user_features['active_days']       = (user_features['last_watch_date'] - user_features['first_watch_date']).dt.days + 1
user_features['session_frequency'] = user_features['total_sessions'] / user_features['active_days']
user_features['completion_ratio']  = user_features['n_completed'] / user_features['total_sessions']
user_features['movie_ratio']       = user_features['n_movies'] / user_features['total_sessions']
user_features['churned']           = (user_features['recency_days'] >= 30).astype(int)
```

### 4. Modeling *(in progress)*
- Logistic Regression with `class_weight='balanced'`
- Random Forest with `class_weight='balanced'`
- Cross-validation: StratifiedKFold (k=5)
- Evaluation: ROC-AUC, F1-score, Precision-Recall

---

## Results

> 🚧 To be updated after model training

| Model | ROC-AUC | F1 (Churn) | Precision | Recall |
|---|---|---|---|---|
| Baseline (majority class) | — | — | — | — |
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |

---

## Project Structure

```
netflix-retention-modeling/
│
├── data/
│   └── watch_joined.csv          # Raw dataset (from Kaggle)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # User-level feature creation
│   └── 03_modeling.ipynb         # Model training & evaluation
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/netflix-retention-modeling.git
cd netflix-retention-modeling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook notebooks/
```

**requirements.txt**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter
```

---

## Key Findings *(preliminary)*

- **Session frequency** is the strongest early signal of churn risk
- Users who stop completing content (low `completion_ratio`) churn at higher rates
- Genre diversity correlates with longer retention

---

*Dec 2025 – Present · University of Wisconsin–Madison*
