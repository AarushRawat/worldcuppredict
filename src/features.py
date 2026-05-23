import pandas as pd

def load_data(path='data/results.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def load_rankings(path='data/fifa_ranking-2024-06-20.csv'):
    df = pd.read_csv(path)
    df['rank_date'] = pd.to_datetime(df['rank_date'])
    return df.sort_values('rank_date').reset_index(drop=True)

def get_fifa_rank(rankings_df, team, before_date):
    """Get the most recent FIFA rank for a team before a given date."""
    rows = rankings_df[
        (rankings_df['country_full'] == team) &
        (rankings_df['rank_date'] < before_date)
    ]
    if len(rows) == 0:
        return 100, 0  # default if no ranking found
    latest = rows.iloc[-1]
    return latest['rank'], latest['total_points']

def get_team_stats(df, team, before_date, n=10):
    matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < before_date)
    ].tail(n)

    if len(matches) == 0:
        return {'win_rate': 0.5, 'avg_goals_for': 1.0, 'avg_goals_against': 1.0, 'avg_goal_diff': 0.0}

    total_weight = 0
    weighted_wins, goals_for, goals_against = 0, 0, 0
    n_matches = len(matches)

    for i, (_, row) in enumerate(matches.iterrows()):
        weight = 2.0 ** (i / n_matches)
        if row['home_team'] == team:
            gf, ga = row['home_score'], row['away_score']
        else:
            gf, ga = row['away_score'], row['home_score']
        goals_for     += gf * weight
        goals_against += ga * weight
        total_weight  += weight
        if gf > ga:    weighted_wins += 1.0 * weight
        elif gf == ga: weighted_wins += 0.5 * weight

    return {
        'win_rate':          weighted_wins / total_weight,
        'avg_goals_for':     goals_for / total_weight,
        'avg_goals_against': goals_against / total_weight,
        'avg_goal_diff':     (goals_for - goals_against) / total_weight
    }


def build_features(df, min_year=2000):
    rankings_df = load_rankings()
    df = df[df['date'].dt.year >= min_year].copy()
    rows = []

    print(f"Building features for {len(df)} matches...")
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 2000 == 0: print(f"  {i}/{len(df)}")

        home = row['home_team']
        away = row['away_team']
        date = row['date']

        hs  = get_team_stats(df, home, date)
        as_ = get_team_stats(df, away, date)

        home_rank, home_pts = get_fifa_rank(rankings_df, home, date)
        away_rank, away_pts = get_fifa_rank(rankings_df, away, date)

        if row['home_score'] > row['away_score']: outcome = 0
        elif row['home_score'] < row['away_score']: outcome = 2
        else: outcome = 1

        rows.append({
            'home_win_rate':      hs['win_rate'],
            'away_win_rate':      as_['win_rate'],
            'win_rate_diff':      hs['win_rate'] - as_['win_rate'],
            'home_goals_for':     hs['avg_goals_for'],
            'away_goals_for':     as_['avg_goals_for'],
            'home_goals_against': hs['avg_goals_against'],
            'away_goals_against': as_['avg_goals_against'],
            'home_goal_diff':     hs['avg_goal_diff'],
            'away_goal_diff':     as_['avg_goal_diff'],
            'goal_diff_delta':    hs['avg_goal_diff'] - as_['avg_goal_diff'],
            'home_rank':          home_rank,
            'away_rank':          away_rank,
            'rank_diff':          away_rank - home_rank,
            'home_pts':           home_pts,
            'away_pts':           away_pts,
            'neutral':            int(row.get('neutral', False)),
            'outcome':            outcome
        })

    return pd.DataFrame(rows)