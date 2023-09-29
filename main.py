import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def progress(row):
    if pd.isnull(row['firstRound']):
        return 0
    elif row['firstRound'] == 'L':
        return 0.5
    elif row['semis'] == 'L':
        return 1.5
    elif row['finals'] == 'L':
        return 2.5
    else:
        return 3


def main():
    awards_players = pd.read_csv('./basketballPlayoffs/awards_players.csv')
    coaches = pd.read_csv('./basketballPlayoffs/coaches.csv')
    players = pd.read_csv('./basketballPlayoffs/players.csv')
    players_teams = pd.read_csv('./basketballPlayoffs/players_teams.csv')
    series_post = pd.read_csv('./basketballPlayoffs/series_post.csv')
    teams = pd.read_csv('./basketballPlayoffs/teams.csv')
    teams_post = pd.read_csv('./basketballPlayoffs/teams_post.csv')

    dataframes_dict = {
        'awards_players': awards_players,
        'coaches': coaches,
        'players': players,
        'players_teams': players_teams,
        'series_post': series_post,
        'teams': teams,
        'teams_post': teams_post
    }

    # Death Date is irrelevant for the way they play
    dataframes_dict['players'].drop('deathDate', inplace=True, axis='columns')

    # Removing NaN on pos
    players = players.dropna(subset=['pos'])

    # Removing players with 0000-00-00 as birthDates
    players = players[players['birthDate'] != '0000-00-00']

    # for name, table in dataframes_dict.items():
    #     if table.isnull().values.any():
    #        print(name)
    #        print(table[table.isnull().any(axis=1)])

    # Converting colleges to an index
    unique_colleges = set(list(players['college'].unique()) + list(players['collegeOther'].unique()))
    college_mapping = {}
    for index, college in enumerate(unique_colleges):
        college_mapping[college] = index
    players['college'] = players['college'].replace(college_mapping)
    players['collegeOther'] = players['collegeOther'].replace(college_mapping)

    # First Season and Last Season were always 0, so we decided to remove them
    players = players.drop(['firstseason', 'lastseason'], axis='columns')

    # Converting positions to an index
    unique_pos = set(players['pos'].unique())
    pos_mapping = {pos: ind for ind, pos in enumerate(unique_pos)}
    players['pos'] = players['pos'].replace(pos_mapping)

    # Day and month of birth is irrelevant in our view therefore we will save only the birthYear
    players['birthDate'] = pd.to_datetime(players['birthDate'])
    players['birthYear'] = players['birthDate'].dt.year
    players = players.drop('birthDate', axis='columns')
    teams.drop('divID', inplace=True, axis='columns')

    merged_df = players.merge(awards_players, left_on='bioID', right_on='playerID', how='left')
    # Group by playerId and calculate the number of awards for each player
    awards_count = merged_df.groupby('bioID')['award'].count().reset_index()
    # Rename the 'award' column to 'numAwards' for clarity
    awards_count.rename(columns={'award': 'numAwards'}, inplace=True)
    # Merge the awards_count DataFrame back to the players DataFrame
    players = players.merge(awards_count, left_on='bioID', right_on='bioID', how='left')
    # Fill NaN values in numAwards column with 0 (players with no awards)
    players['numAwards'].fillna(0, inplace=True)

    players_teams['EFF'] = (players_teams['points'] + players_teams['rebounds'] + players_teams['assists'] +
                            players_teams['steals'] + players_teams['blocks'] - (
                                    players_teams['fgAttempted'] - players_teams['fgMade']) - (
                                    players_teams['ftAttempted'] - players_teams['ftMade']) - np.where(
                players_teams['GP'] == 0, 0, players_teams['turnovers'])) / players_teams['GP']

    players_teams = pd.merge(players_teams, players, left_on='playerID', right_on='bioID',
                             how='inner').drop_duplicates()

    # creating new features to help predict the EFF for the next year:

    # Create lag features for EFF - It's how you include the EFF of the previous years to predict this year
    lag_years = 3
    for year in range(1, lag_years + 1):
        players_teams[f'EFF_Lag_{year}'] = players_teams.groupby('playerID')['EFF'].shift(year)
    players_teams[[f'EFF_Lag_{year}' for year in range(1, lag_years + 1)]] = players_teams[
        [f'EFF_Lag_{year}' for year in range(1, lag_years + 1)]].fillna(0)

    # Calculate Field Goal Percentage (FG%), Free Throw Percentage (FT%), and Points Per Game (PPG)
    players_teams['FG_Percentage'] = (players_teams['fgMade'] / players_teams['fgAttempted']) * 100
    players_teams['FT_Percentage'] = (players_teams['ftMade'] / players_teams['ftAttempted']) * 100
    players_teams['PPG'] = players_teams['points'] / players_teams['GP']
    # merge players and awards_players by bioID and playerID

    players_teams = pd.merge(players_teams, players, left_on='playerID', right_on='bioID', how='inner')

    series_post.drop(['lgIDLoser', 'lgIDWinner'], inplace=True, axis='columns')
    unique_teams = set(list(series_post['tmIDLoser'].unique()) + list(series_post['tmIDWinner'].unique()))
    team_mapping = {}
    for index, team in enumerate(unique_teams):
        team_mapping[team] = index
    series_post['tmIDLoser'] = series_post['tmIDLoser'].replace(team_mapping)
    series_post['tmIDWinner'] = series_post['tmIDWinner'].replace(team_mapping)

    label_encoder = LabelEncoder()
    series_post['series'] = label_encoder.fit_transform(series_post['series'])
    round_mapping = {'FR': 1, 'CF': 2, 'F': 3}
    series_post['round'] = series_post['round'].replace(round_mapping)
    # Remove rows with missing values in FGP, FTP, and PPG
    players_teams.dropna(subset=['FG_Percentage', 'FT_Percentage', 'PPG'], inplace=True)

    teams['progress'] = teams.apply(lambda row: progress(row), axis=1)

    teams = teams.drop(columns=['firstRound', 'semis', 'finals', 'lgID', 'seeded', 'arena', 'attend', 'min'])

    teams['confWLDifference'] = teams['confW'] - teams['confL']
    teams['awayWLDifference'] = teams['awayW'] - teams['awayL']
    teams['homeWLDifference'] = teams['homeW'] - teams['homeL']
    teams['gamesWLDifference'] = teams['won'] - teams['lost']

    teams = teams.drop(columns=['confW', 'confL', 'awayW', 'awayL', 'homeW', 'homeL', 'won', 'lost'])

    teams['offensive_performance'] = ((teams['o_pts'] / (teams['o_fga'] + 0.44 * teams['o_fta'])) + (
                (teams['o_fgm'] + teams['o_3pm']) / (teams['o_fga'] + teams['o_3pa'])) + (
                                                  teams['o_asts'] / (teams['o_to'] + 1)) + teams['o_oreb'])
    teams['defensive_performance'] = (((teams['d_fgm'] + teams['d_3pm']) / (teams['d_fga'] + 0.44 * teams['d_fta'])) + (
                (teams['d_fgm'] + teams['d_3pm']) / (teams['d_fga'] + teams['d_3pa'])) + teams['d_dreb'] + teams[
                                          'd_stl'] + teams['d_blk'] - teams['d_pts'])

    teams = teams.drop(columns=['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb',
                                'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts',
                                'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb',
                                'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'tmORB',
                                'tmDRB', 'tmTRB', 'opptmORB', 'opptmDRB', 'opptmTRB'])
    print(teams.columns)

    print(teams)

    # Create shifted features for FGP, FTP, and PPG
    shifted_years = 3  # Number of previous years to consider
    for feat in ['FG_Percentage', 'FT_Percentage', 'PPG']:
        for year in range(1, shifted_years + 1):
            players_teams[f'{feat}_Lag_{year}'] = players_teams.groupby('playerID')[feat].shift(year)

    # Fill NaN values in the newly created columns with 0
    lagged_features = [f'{feat}_Lag_{year}' for feat in ['FG_Percentage', 'FT_Percentage', 'PPG'] for year in
                       range(1, shifted_years + 1)]
    players_teams[lagged_features] = players_teams[lagged_features].fillna(0)

    # Select relevant features including EFF and the lagged features
    eff_columns = [f'EFF_Lag_{year}' for year in range(1, lag_years + 1)]
    features = lagged_features + eff_columns

    # Define your target variable
    target = 'EFF'

    # Create a feature matrix (X) and target vector (y)
    X = players_teams[features]
    y = players_teams[target]

    # Models for predicting next years individual players performance

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Choosing a model - RandomForestRegressor as an example (LinearRegression, GradientBoostingRegressor are other models)
    from sklearn.ensemble import RandomForestRegressor

    randomForestRegressorModel = RandomForestRegressor(n_estimators=100)

    randomForestRegressorModel.fit(X_train, y_train)

    y_pred = randomForestRegressorModel.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Mean Absolute Error (MAE): {mae:.2f}')  # Means that on average the prediction is mae units off
    print(f'Mean Squared Error (MSE): {mse:.2f}')  #
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R²): {r2:.2f}')

    # Feature Importance - understanding which features are important:

    # Access feature importances
    feature_importances = randomForestRegressorModel.feature_importances_

    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Print or visualize the feature importances
    print(importance_df)

    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each player
    most_recent_years = players_teams.groupby('playerID')['year'].max().reset_index()

    # Create the 'future_player_data' DataFrame with columns matching your original data
    future_player_data = pd.DataFrame(columns=players_teams.columns)

    # Iterate through each player to get the most recent data
    for _, player_row in most_recent_years.iterrows():
        player_id = player_row['playerID']
        most_recent_year = player_row['year']

        # Extract the row corresponding to the most recent year for the player
        most_recent_data = players_teams[
            (players_teams['playerID'] == player_id) & (players_teams['year'] == most_recent_year)]

        # Calculate lagged features for this row (as you did in your previous code)
        for feat in ['FG_Percentage', 'FT_Percentage', 'PPG']:
            for year in range(1, shifted_years + 1):
                most_recent_data[f'{feat}_Lag_{year}'] = most_recent_data[feat].shift(year)

        # Fill NaN values in the newly created columns with 0
        lagged_features = [f'{feat}_Lag_{year}' for feat in ['FG_Percentage', 'FT_Percentage', 'PPG'] for year in
                           range(1, shifted_years + 1)]
        most_recent_data[lagged_features] = most_recent_data[lagged_features].fillna(0)

        # Append this row to the 'future_player_data' DataFrame
        future_player_data = pd.concat([future_player_data, most_recent_data])

    # Select the relevant features for predicting EFF
    future_features = lagged_features + eff_columns

    # Use the trained model to predict EFF for the next year
    future_predictions = randomForestRegressorModel.predict(future_player_data[future_features])

    # Add the predicted EFF values to the 'future_player_data' DataFrame
    future_player_data['Predicted_EFF_Next_Year'] = future_predictions

    print(future_player_data.head())

    coaches['WLDifference'] = coaches['won'] - coaches['lost']
    coaches = coaches.drop(columns=['won', 'lost', 'lgID'])
    print(players_teams)


main()
