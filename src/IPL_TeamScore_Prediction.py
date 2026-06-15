import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import warnings
import joblib


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv("data/IPL.csv")


# removing second innings data
df = df[df['innings'] == 1]


# team-names consistency
mapping_teams = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings'
}

df['batting_team'] = df['batting_team'].replace(mapping_teams)
df['bowling_team'] = df['bowling_team'].replace(mapping_teams)


# seasons fixing
df['season'] = df['season'].astype(str)
df['season'] = df['season'].replace({
    '2007/08': '2008',
    '2009/10': '2010',
    '2020/21': '2020'
})


# venues fixing
venue_mapping = {
    # Chinnaswamy
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
    'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',

    # Punjab / Mohali
    'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',

    # Hyderabad
    'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Uppal',

    # Chennai
    'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium, Chepauk',

    # Delhi
    'Arun Jaitley Stadium': 'Feroz Shah Kotla',
    'Arun Jaitley Stadium, Delhi': 'Feroz Shah Kotla',

    # Mumbai
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
    'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
    'Brabourne Stadium, Mumbai': 'Brabourne Stadium',

    # Ahmedabad
    'Narendra Modi Stadium, Ahmedabad': 'Sardar Patel Stadium, Motera',

    # Pune
    'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',

    # Kolkata
    'Eden Gardens, Kolkata': 'Eden Gardens',

    # Lucknow
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow':
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',

    # Jaipur
    'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',

    # Dharamsala
    'Himachal Pradesh Cricket Association Stadium, Dharamsala':
        'Himachal Pradesh Cricket Association Stadium',

    # Visakhapatnam
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',

    # Chandigarh / Mullanpur
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh':
        'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur'
}

df['venue'] = df['venue'].replace(venue_mapping)


# stage fixing
df['stage'] = df['stage'].replace({'Elimination Final': 'Qualifier 2'})
df['stage'] = df['stage'].replace({'Unknown': 'League Stage'})


# -----------  Feature Engineering -------------

# final_score
df['final_score'] = df.groupby('match_id')['team_runs'].transform('max')

# current_run_rate
df['current_run_rate'] = np.where(
    df['team_balls'] > 0,
    (df['team_runs'] * 6) / df['team_balls'],
    0
)

# balls_remaining
df['balls_remaining'] = 120 - df['team_balls']

# wickets_remaining
df['wickets_remaining'] = 10 - df['team_wicket']


# last_5_over_runs, last_2_over_runs
df['overs'] = df['ball_no'].astype(int) + 1

df_overs = df.groupby(['match_id', 'overs'])['team_runs'].max().reset_index().rename(columns={'team_runs': 'cum_runs_at_end_of_over'})
df_overs['over_runs'] = df_overs.groupby('match_id')['cum_runs_at_end_of_over'].diff().fillna(df_overs['cum_runs_at_end_of_over'])
df_overs['last_5_over_runs'] = (df_overs.groupby('match_id')['over_runs']
                                .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()).fillna(0))
df_overs['last_2_over_runs'] = (df_overs.groupby('match_id')['over_runs']
                                .transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).sum()).fillna(0))

df = pd.merge(df, df_overs[['match_id', 'overs', 'last_5_over_runs', 'last_2_over_runs']], how='left', on=['match_id', 'overs'])


# last_5_over_wickets, last_3_over_wickets
df_wickets = df.groupby(['match_id', 'overs'])['team_wicket'].max().reset_index()
df_wickets['over_wickets'] = df_wickets.groupby('match_id')['team_wicket'].diff().fillna(df_wickets['team_wicket'])
df_wickets['last_5_over_wickets'] = (df_wickets.groupby('match_id')['over_wickets']
                                     .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()).fillna(0))
df_wickets['last_3_over_wickets'] = (df_wickets.groupby('match_id')['over_wickets']
                                     .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()).fillna(0))

df = pd.merge(df, df_wickets[['match_id', 'overs', 'last_5_over_wickets', 'last_3_over_wickets']], on=['match_id', 'overs'], how='left')

df = df.drop(columns=['overs'])


# STRIKER CAREER STRIKE RATE (using only pre-match data to avoid leakage)

df['balls_faced'] = df.groupby(['match_id', 'batter'])['balls_faced'].cumsum()
df_striker = df.groupby(['match_id', 'batter']).agg({'batter_runs': 'max', 'balls_faced': 'max'}).reset_index()
df_striker['striker_career_sr'] = (
    df_striker.groupby('batter')['batter_runs'].transform(lambda x: x.shift(1).cumsum()) /
    df_striker.groupby('batter')['balls_faced'].transform(lambda x: x.shift(1).cumsum()) * 100
)
df_striker['striker_career_sr'] = df_striker['striker_career_sr'].fillna(0)
df = pd.merge(df, df_striker[['striker_career_sr', 'match_id', 'batter']], on=['match_id', 'batter'], how='left')


# NON-STRIKER CAREER STRIKE RATE

df = pd.merge(
    df,
    df_striker[['match_id', 'batter', 'striker_career_sr']].rename(
        columns={'batter': 'non_striker', 'striker_career_sr': 'non_striker_sr'}
    ),
    on=['match_id', 'non_striker'], how='left'
)
df['non_striker_sr'] = df['non_striker_sr'].fillna(0)


# current_partnership
wicket_df = (
    df.groupby(['match_id', 'team_wicket'])['team_runs']
      .min()
      .reset_index()
      .rename(columns={'team_runs': 'partnership_start_runs'})
)

df = pd.merge(df, wicket_df, on=['match_id', 'team_wicket'], how='left')

df['current_partnership'] = df['team_runs'] - df['partnership_start_runs']

df.drop(columns=['partnership_start_runs'], inplace=True)


# ------------------------- Features Tried But Dropped -------------------------------
#
# All features below were engineered, tested, and dropped due to negligible
# improvement (feature importance ~0.01 or model R² decreased).
#
# 1. TEAM AVERAGE SCORE (last 7 matches)
#    Result: R² dropped — made model worse.
#
# team_scores = (
#     df.groupby(['match_id', 'batting_team'])
#       .agg({'final_score': 'max'})
#       .reset_index()
# )
# team_scores['team_avg_score_last_7_matches'] = (
#     team_scores.groupby('batting_team')['final_score']
#     .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
# )
# team_scores['team_avg_score_last_7_matches'] = team_scores['team_avg_score_last_7_matches'].fillna(0)
# df = pd.merge(df, team_scores[['match_id', 'batting_team', 'team_avg_score_last_7_matches']],
#               on=['match_id', 'batting_team'], how='left')
#

#
# 2. STRIKER RECENT FORM INDEX (last 5 matches: avg_runs * sr/100)
#    Result: Feature importance ~0.01, no meaningful improvement.
#
# recent_form_striker_df = (
#     df.groupby(['match_id', 'batter'])
#       .agg({'batter_runs': 'max', 'balls_faced': 'max'})
#       .reset_index()
# )
# recent_form_striker_df['runs_last_5_matches'] = (
#     recent_form_striker_df.groupby('batter')['batter_runs']
#     .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
# )
# recent_form_striker_df['balls_last_5_matches'] = (
#     recent_form_striker_df.groupby('batter')['balls_faced']
#     .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
# )
# recent_form_striker_df['sr_last_5_matches'] = (
#     recent_form_striker_df['runs_last_5_matches'] /
#     recent_form_striker_df['balls_last_5_matches']
# ) * 100
# recent_form_striker_df['sr_last_5_matches'] = recent_form_striker_df['sr_last_5_matches'].fillna(0)
# recent_form_striker_df['striker_recent_form_index'] = (
#     (recent_form_striker_df['runs_last_5_matches'] / 5) *
#     (recent_form_striker_df['sr_last_5_matches'] / 100)
# )
# recent_form_striker_df['striker_recent_form_index'] = recent_form_striker_df['striker_recent_form_index'].fillna(0)
# df = pd.merge(df, recent_form_striker_df[['match_id', 'batter', 'striker_recent_form_index']],
#               on=['match_id', 'batter'], how='left')

# ------------------------------------------------------------------------------------------------


# dropping useless cols
cols_to_drop = [
    'match_type', 'event_name', 'team_type', 'balls_per_over', 'gender',
    'Unnamed: 0', 'date', 'day', 'review_batter', 'team_reviewed', 'review_decision',
    'fielders', 'ball', 'umpires_call', 'superover_winner', 'city', 'method',
    'umpire', 'player_of_match', 'match_won_by', 'win_outcome', 'runs_target',
    'result_type', 'month', 'year', 'match_number', 'event_match_no',
    'runs_not_boundary', 'extra_type', 'over', 'wicket_kind', 'bat_pos',
    'bowler', 'runs_bowler', 'batter', 'non_striker', 'runs_batter', 'balls_faced',
    'non_striker_pos', 'player_out', 'batter_runs', 'batter_balls', 'batting_partners', 'striker_out',
    'bowler_wicket', 'next_batter', 'new_batter',
    'power_surge_start', 'valid_ball', 'runs_extras', 'runs_total',
    'innings', 'toss_winner', 'toss_decision'
]

df = df.drop(columns=cols_to_drop)


# dropping first 5 overs
df = df[df['ball_no'] >= 5]

# analysis df for error analysis
analysis_df = df[[
    'match_id',
    'season',
    'venue',
    'batting_team',
    'final_score'
]].copy()


# ------------------------------ Encoding ---------------------------------------

categorical_cols = ['batting_team', 'bowling_team', 'venue', 'stage', 'season']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df.astype(float)

print(df.head())
print(df.shape)


# ------------------ Train / Test Split --------------------
# GroupShuffleSplit used instead of random split to prevent data leakage —
# balls from the same match must never appear in both train and test sets.

X = df.drop(columns=['final_score', 'match_id'])
Y = df['final_score']

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, Y, groups=df['match_id']))
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
Y_train = Y.iloc[train_idx]
Y_test = Y.iloc[test_idx]


# ---------------------- Model ---------------------------

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter tuning via RandomizedSearchCV (run once, best params saved).
# Best params found:
#   n_estimators=100, max_depth=3, learning_rate=0.1,
#   subsample=0.7, colsample_bytree=0.8

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }
#
# xgb = XGBRegressor(
#     tree_method='hist',
#     device='cuda',
#     random_state=42,
# )
#
# classifier = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_grid,
#     n_iter=10,
#     cv=3,
#     scoring='r2',
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )
#
# classifier.fit(X_train, Y_train)
#
# joblib.dump(classifier.best_estimator_, 'models/ipl_score_predictor.pkl')

model = joblib.load('models/ipl_score_predictor.pkl')

print(model.get_params())

predictions = model.predict(X_test)

print("\n--- XGBoost Model Performance ---")
print("Test R2:", r2_score(Y_test, predictions))
print("Test MAE:", mean_absolute_error(Y_test, predictions))
print("Test MSE:", mean_squared_error(Y_test, predictions))

best_model = model

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n--- Top 30 Feature Importances ---")
print(importance.head(30))

# Baseline comparison: Linear Regression
lm = LinearRegression()
lm.fit(X_train, Y_train)
lmpredictions = lm.predict(X_test)
print("\n--- Linear Regression (Baseline) ---")
print("r2_score:", r2_score(Y_test, lmpredictions))
print("Test MAE:", mean_absolute_error(Y_test, lmpredictions))
print("Test MSE:", mean_squared_error(Y_test, lmpredictions))

# Baseline comparison: Random Forest
rf = RandomForestRegressor(max_depth=None, n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
rfpredictions = rf.predict(X_test)
print("\n--- Random Forest ---")
print("r2_score:", r2_score(Y_test, rfpredictions))
print("Test MAE:", mean_absolute_error(Y_test, rfpredictions))
print("Test MSE:", mean_squared_error(Y_test, rfpredictions))


# ---------------------------- Error Analysis ---------------------------------

test_analysis = analysis_df.iloc[test_idx].copy().reset_index(drop=True)
test_analysis['predicted'] = predictions
test_analysis['error'] = test_analysis['final_score'] - test_analysis['predicted']

match_errors = (
    test_analysis
    .groupby('match_id', as_index=False)
    .agg({
        'season': 'first',
        'batting_team': 'first',
        'venue': 'first',
        'final_score': 'first',
        'predicted': 'mean'
    })
)

match_errors['error'] = match_errors['final_score'] - match_errors['predicted']
match_errors['abs_error'] = match_errors['error'].abs()

print("\n--- Worst Predictions (by absolute error) ---")
print(match_errors.sort_values('abs_error', ascending=False).head(20))

print("\n--- Final Score Distribution ---")
print(match_errors['final_score'].describe())

low_scores = match_errors[match_errors['final_score'] < 120]
high_scores = match_errors[match_errors['final_score'] > 220]

print("\nLow score matches (<120):", len(low_scores))
print("High score matches (>220):", len(high_scores))

# Bias analysis: model systematically overpredicts collapses and
# underpredicts explosions (regression-to-the-mean behaviour).

print("\n--- Low Score Bias (<120) ---")
print(low_scores[['final_score', 'predicted']].describe())
print("Mean error:", low_scores['error'].mean())

print("\n--- High Score Bias (>220) ---")
print(high_scores[['final_score', 'predicted']].describe())
print("Mean error:", high_scores['error'].mean())


