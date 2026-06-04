import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv(r"C:\Users\rithv\OneDrive\Documents\ipl dataset\IPL.csv")


#removing second innings data
df = df[df['innings'] == 1]



#team-names consistency
mapping_teams = {
'Royal Challengers Bangalore' : 'Royal Challengers Bengaluru',
'Rising Pune Supergiant' : 'Rising Pune Supergiants',
'Delhi Daredevils' : 'Delhi Capitals',
'Kings XI Punjab' : 'Punjab Kings'
}

df['batting_team'] = df['batting_team'].replace(mapping_teams)
df['bowling_team'] = df['bowling_team'].replace(mapping_teams)


#seasons fixing
df['season'] = df['season'].astype(str)
df['season'] = df['season'].replace({'2007/08':'2008','2009/10':'2010','2020/21':'2020'})

#venues fixing
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


#stage fixing
df['stage'] = df['stage'].replace({'Elimination Final': 'Qualifier 2'})
df['stage'] = df['stage'].replace({'Unknown': 'League Stage'})

#adding features

#final_score
df['final_score'] = df.groupby('match_id')['team_runs'].transform('max')

# current_run_rate
df['current_run_rate'] = np.where(
    df['team_balls'] > 0,
    (df['team_runs'] * 6) / df['team_balls'],
    0
)

#balls_remaining
df['balls_remaining'] = 120 - df['team_balls']

#wickets_remaining
df['wickets_remaining'] = 10 - df['team_wicket']



#last_5_over_runs,last_2_over_runs

df['overs'] = df['ball_no'].astype(int) + 1

df_overs = df.groupby(['match_id','overs'])['team_runs'].max().reset_index().rename(columns={'team_runs':'cum_runs_at_end_of_over'})
df_overs['over_runs'] = df_overs.groupby('match_id')['cum_runs_at_end_of_over'].diff().fillna(df_overs['cum_runs_at_end_of_over'])
df_overs['last_5_over_runs'] = (df_overs.groupby('match_id')['over_runs'].
                                transform(lambda x: x.shift(1).rolling(window=5,min_periods=1).sum()).fillna(0))
df_overs['last_2_over_runs'] = (df_overs.groupby('match_id')['over_runs'].
                                transform(lambda x: x.shift(1).rolling(window=2,min_periods=1).sum()).fillna(0))


#merging last_5_over_runs,last_2_over_runs to df
df = pd.merge(df,df_overs[['match_id','overs','last_5_over_runs',
                           'last_2_over_runs']],how='left',on=['match_id','overs'])


#last_5_over_wickets,last_3_over_wickets

df_wickets = df.groupby(['match_id','overs'])['team_wicket'].max().reset_index()
df_wickets['over_wickets'] = df_wickets.groupby('match_id')['team_wicket'].diff().fillna(df_wickets['team_wicket'])
df_wickets['last_5_over_wickets'] = (df_wickets.groupby('match_id')['over_wickets'].
                                     transform(lambda x:x.shift(1).rolling(window=5,min_periods=1).sum()).fillna(0))
df_wickets['last_3_over_wickets'] = (df_wickets.groupby('match_id')['over_wickets'].
                                     transform(lambda x:x.shift(1).rolling(window=3,min_periods=1).sum()).fillna(0))

#merging last_5_over_wickets,last_3_over_wickets to df
df = pd.merge(df,df_wickets[['match_id','overs','last_5_over_wickets','last_3_over_wickets']],on=['match_id','overs'],how='left')

df = df.drop(columns=['overs'])


# striker_career_sr

df['balls_faced'] = df.groupby(['match_id','batter'])['balls_faced'].cumsum()
df_striker = df.groupby(['match_id','batter']).agg({'batter_runs':'max','balls_faced':'max'}).reset_index()
df_striker['striker_career_sr'] = (df_striker.groupby('batter')['batter_runs']
                                    .transform(lambda x: x.shift(1).cumsum()) /
                                    df_striker.groupby('batter')['balls_faced']
                                    .transform(lambda x: x.shift(1).cumsum())  * 100)
df_striker['striker_career_sr'] = df_striker['striker_career_sr'].fillna(0)

#merging striker_career_sr to df
df = pd.merge(df,df_striker[['striker_career_sr','match_id','batter']],on=['match_id','batter'],how='left')

#non_striker_sr
df = pd.merge(
    df,
    df_striker[
        ['match_id', 'batter', 'striker_career_sr']
    ].rename(
        columns={
            'batter': 'non_striker',
            'striker_career_sr': 'non_striker_sr'
        }
    ),
    on=['match_id', 'non_striker'],
    how='left'
)

#current_partnership

wicket_df = (
    df.groupby(['match_id','team_wicket'])['team_runs']
      .min()
      .reset_index()
      .rename(columns={'team_runs':'partnership_start_runs'})
)

df = pd.merge(
    df,
    wicket_df,
    on=['match_id','team_wicket'],
    how='left'
)

df['current_partnership'] = (
    df['team_runs'] -
    df['partnership_start_runs']
)

df.drop(columns=['partnership_start_runs'], inplace=True)


#dropping useless cols
cols_to_drop = [
    'match_type', 'event_name', 'team_type' , 'balls_per_over', 'gender',
    'Unnamed: 0', 'date', 'day', 'review_batter', 'team_reviewed', 'review_decision',
    'fielders', 'ball', 'umpires_call', 'superover_winner', 'city', 'method',
    'umpire', 'player_of_match', 'match_won_by', 'win_outcome', 'runs_target',
    'result_type', 'month', 'year', 'match_number', 'event_match_no',
    'runs_not_boundary', 'extra_type', 'over', 'wicket_kind', 'bat_pos',
     'bowler', 'runs_bowler', 'batter' , 'non_striker' , 'runs_batter' , 'balls_faced',
    'non_striker_pos', 'player_out', 'batter_runs' , 'batter_balls' , 'batting_partners' , 'striker_out',
    'bowler_wicket', 'next_batter', 'new_batter',
    'power_surge_start', 'valid_ball', 'runs_extras', 'runs_total',
    'innings', 'toss_winner', 'toss_decision'
]

df = df.drop(columns= cols_to_drop)


#dropping first 5 overs
df = df[df['ball_no'] >= 5]



# encoding

categorical_cols = ['batting_team','bowling_team','venue','stage','season']


df = pd.get_dummies(df,columns=categorical_cols,drop_first=True)
df = df.astype(float)

print(df.head())
print(df.shape)


# #splitting training and testing data
X = df.drop(columns=['final_score','match_id'])
Y = df['final_score']


gss = GroupShuffleSplit(test_size=0.2,random_state=42)
train_idx , test_idx = next(gss.split(X,Y,groups=df['match_id']))
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
Y_train = Y.iloc[train_idx]
Y_test = Y.iloc[test_idx]




from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(
    tree_method='hist',
    device='cuda',     # uses Colab GPU
    random_state=42
)

classifier = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

classifier.fit(X_train, Y_train)


importance = classifier.best_estimator_.feature_importances_

feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feat_imp.head(20))

#
# print("Best Parameters:", classifier.best_params_)
# print("Best R2 Score:", classifier.best_score_)
#
# predictions = classifier.best_estimator_.predict(X_test)
# print("Test R2:", r2_score(Y_test, predictions))
# print("Test MAE:", mean_absolute_error(Y_test, predictions))
# print("Test MSE:", mean_squared_error(Y_test, predictions))
