import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.features import load_data, build_features

print("Loading data...")
df = load_data()
features_df = build_features(df)

features_df['form_x_rank']        = features_df['win_rate_diff'] * features_df['rank_diff']
features_df['gdiff_x_rank']       = features_df['goal_diff_delta'] * features_df['rank_diff']
features_df['absolute_rank_diff'] = (features_df['home_rank'] - features_df['away_rank']).abs()
features_df['absolute_goal_diff'] = features_df['goal_diff_delta'].abs()
features_df['is_close_match']     = (features_df['absolute_rank_diff'] < 10).astype(int)

COLS = [
    'home_win_rate','away_win_rate','win_rate_diff',
    'home_goals_for','away_goals_for','home_goals_against','away_goals_against',
    'home_goal_diff','away_goal_diff','goal_diff_delta',
    'home_rank','away_rank','rank_diff','home_pts','away_pts',
    'neutral','form_x_rank','gdiff_x_rank',
    'absolute_rank_diff','absolute_goal_diff','is_close_match'
]

X = features_df[COLS]
y = features_df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42
    ))
])

print("Training...")
pipeline.fit(X_train, y_train)

print(f"\nAccuracy: {accuracy_score(y_test, pipeline.predict(X_test)):.1%}")
print(classification_report(y_test, pipeline.predict(X_test), target_names=['Home Win','Draw','Away Win']))

joblib.dump((pipeline, COLS), 'models/model.pkl')
print("Saved.")