import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import pandas as pd
from src.features import load_data, get_team_stats

model = joblib.load('models/model.pkl')
df = load_data()

def predict_match(home, away, neutral=False):
    today = pd.Timestamp.now()
    hs = get_team_stats(df, home, today)
    as_ = get_team_stats(df, away, today)

    X = [[hs['win_rate'], as_['win_rate'],
          hs['win_rate'] - as_['win_rate'],
          hs['avg_goals_for'], as_['avg_goals_for'],
          hs['avg_goals_against'], as_['avg_goals_against'],
          int(neutral)]]

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    res = {c: p for c, p in zip(classes, proba)}

    print(f"\n{home} vs {away}")
    print(f"  {home} win : {res.get(0,0):.1%}")
    print(f"  Draw       : {res.get(1,0):.1%}")
    print(f"  {away} win : {res.get(2,0):.1%}")

predict_match('Brazil', 'Argentina')
predict_match('England', 'France')