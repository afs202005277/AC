import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import models


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


def find_and_move_max_year_records(players_teams):
    max_year = players_teams['year'].max()
    max_year_records = players_teams[players_teams['year'] == max_year]
    max_year_df = pd.DataFrame(max_year_records)
    players_teams = players_teams[players_teams['year'] != max_year]
    max_year_df.reset_index(drop=True, inplace=True)
    players_teams.reset_index(drop=True, inplace=True)

    return max_year_df, players_teams

def initial_data_load():
    print("Initial Data Load")
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

    return dataframes_dict

def player_data_cleanup(players):
    print("Player Data Cleanup")
    # Death Date is irrelevant for the way they play
    players.drop('deathDate', inplace=True, axis='columns')

    # Removing NaN on pos
    players = players.dropna(subset=['pos'])

    # Removing players with 0000-00-00 as birthDates
    players = players[players['birthDate'] != '0000-00-00']

    college_map = college_mapping(set(list(players['college'].unique()) + list(players['collegeOther'].unique())))

    players['college'] = players['college'].replace(college_map)
    players['collegeOther'] = players['collegeOther'].replace(college_map)

    # First Season and Last Season were always 0, so we decided to remove them
    players = players.drop(['firstseason', 'lastseason'], axis='columns')

    pos_mapping = position_mapping(set(players['pos'].unique()))

    # Converting positions to an index
    players['pos'] = players['pos'].replace(pos_mapping)

    # Day and month of birth is irrelevant in our view therefore we will save only the birthYear
    players['birthDate'] = pd.to_datetime(players['birthDate'])
    players['birthYear'] = players['birthDate'].dt.year
    players = players.drop('birthDate', axis='columns')

    return players

def college_mapping(unique_colleges):
    college_mapping = {}
    for index, college in enumerate(unique_colleges):
        college_mapping[college] = index

    return college_mapping

def team_mapping(unique_teams):
    team_mapping = {}
    for index, team in enumerate(unique_teams):
        team_mapping[team] = index

    return team_mapping

def position_mapping(unique_pos):
    pos_mapping = {pos: ind for ind, pos in enumerate(unique_pos)}
    return pos_mapping

def merge_players_awards(players, awards_players):
    print("Merge Players Awards")
    merged_df = players.merge(awards_players, left_on='bioID', right_on='playerID', how='left')
    # Group by playerId and calculate the number of awards for each player
    awards_count = merged_df.groupby('bioID')['award'].count().reset_index()
    # Rename the 'award' column to 'numAwards' for clarity
    awards_count.rename(columns={'award': 'numAwards'}, inplace=True)
    # Merge the awards_count DataFrame back to the players DataFrame
    players = players.merge(awards_count, left_on='bioID', right_on='bioID', how='left')
    # Fill NaN values in numAwards column with 0 (players with no awards)
    players['numAwards'].fillna(0, inplace=True)

    return players

def feature_creation_players_teams(players_teams):
    print("Feature Creation - Players Teams")
    # Creating Eficiency column of the players
    players_teams['EFF'] = (players_teams['points'] + players_teams['rebounds'] + players_teams['assists'] +
                            players_teams['steals'] + players_teams['blocks'] - (
                                    players_teams['fgAttempted'] - players_teams['fgMade']) - (
                                    players_teams['ftAttempted'] - players_teams['ftMade']) - np.where(
                players_teams['GP'] == 0, 0, players_teams['turnovers'])) / players_teams['GP']

    # Creating Defense Score column of the players
    players_teams['DPR'] = 100 - (100 * (
            players_teams['dRebounds'] + players_teams['steals'] + players_teams['blocks'] - players_teams['PF'] -
            players_teams['turnovers'] - players_teams['points'])) / players_teams['GP']

    # Calculate Field Goal Percentage (FG%), Free Throw Percentage (FT%), and Points Per Game (PPG)
    players_teams['FG_Percentage'] = (players_teams['fgMade'] / players_teams['fgAttempted']) * 100
    players_teams['FT_Percentage'] = (players_teams['ftMade'] / players_teams['ftAttempted']) * 100
    players_teams['PPG'] = players_teams['points'] / players_teams['GP']

    # Remove rows with missing values in FGP, FTP, and PPG
    players_teams.dropna(subset=['FG_Percentage', 'FT_Percentage', 'PPG'], inplace=True)

    return players_teams

def merge_players_teams(players_teams, players):
    print("Merge Player Teams")
    players_teams = pd.merge(players_teams, players, left_on='playerID', right_on='bioID',
                             how='inner').drop_duplicates()

    return players_teams

def series_post_data_cleanup(series_post, team_mapping):
    print("Series Post Data Cleanup")
    series_post.drop(['lgIDLoser', 'lgIDWinner'], inplace=True, axis='columns')
    series_post['tmIDLoser'] = series_post['tmIDLoser'].replace(team_mapping)
    series_post['tmIDWinner'] = series_post['tmIDWinner'].replace(team_mapping)

    label_encoder = LabelEncoder()
    series_post['series'] = label_encoder.fit_transform(series_post['series'])
    round_mapping = {'FR': 1, 'CF': 2, 'F': 3}
    series_post['round'] = series_post['round'].replace(round_mapping)

    return series_post

def create_lagged_features_players(players_teams, features_to_be_lagged, lag_years):
    print("Create Lagged Features Players")
    # Create lagged features
    for feat in features_to_be_lagged:
        for year in range(1, lag_years + 1):
            players_teams[f'{feat}_Lag_{year}'] = players_teams.groupby('playerID')[feat].shift(year)

    # Fill NaN values in the newly created columns with 0
    lagged_features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                       range(1, lag_years + 1)]
    players_teams[lagged_features] = players_teams[lagged_features].fillna(0)

    return players_teams

def models_train_and_test_players(players_teams, features, target):
    print("Models Running - Players")
    last_year_records, players_train_teams = find_and_move_max_year_records(players_teams)

    x_train = players_train_teams[features]
    y_train = players_train_teams[target]
    x_test = last_year_records[features]
    y_test = last_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target)

    # Feature Importance - understanding which features are important:

    # Access feature importances
    feature_importances = trained_models['Random Forest Regressor'].feature_importances_

    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Print or visualize the feature importances
    print(importance_df)

    return trained_models

def models_predict_future_players(players_teams, features, features_to_be_lagged, model, lag_years):
    print("Models Predicting - Players")
    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each player
    most_recent_years = players_teams.groupby('playerID')['year'].max().reset_index()

    # Create an empty list to store new rows
    new_rows = []

    # Iterate over unique player IDs and add new rows to the list
    for _, row in most_recent_years.iterrows():
        tmp_data = players_teams.loc[(players_teams['playerID'] == row['playerID']) & (players_teams['year'] == row['year'])]
        team_id = tmp_data.at[tmp_data.index[0], 'tmID']
        player_id = row['playerID']
        max_year = row['year']
        next_year = max_year + 1

        # Create a new row with the next year and player ID, leaving other columns blank
        new_row = [player_id, next_year, np.nan, team_id] + [np.nan] * (players_teams.shape[1] - 4)

        # Append the new row to the list
        new_rows.append(new_row)

    # Create the 'future_player_data' DataFrame with columns matching your original data
    future_player_data = pd.DataFrame(new_rows, columns=players_teams.columns)

    # Concatenate the original DataFrame and the new rows DataFrame
    future_player_data = pd.concat([players_teams, future_player_data], ignore_index=True)

    # Reset the index of the modified DataFrame
    future_player_data.reset_index(drop=True, inplace=True)

    future_player_data = create_lagged_features_players(future_player_data, features_to_be_lagged, lag_years)

    # Use the trained model to predict EFF for the next year
    future_predictions = model.predict(future_player_data[features])

    future_player_data['Predicted_EFF'] = future_predictions

    return future_player_data

def feature_creation_coaches(coaches):
    coaches['WLRatio'] = coaches['won'] / (coaches['won'] + coaches['lost'])
    coaches['WLRatio_Post'] = coaches['post_wins'] / (coaches['post_wins'] + coaches['post_losses'])

    return coaches


def coaches_data_cleanup(coaches, team_map):
    coaches = coaches.drop(columns=['won', 'lost', 'post_wins', 'post_losses', 'lgID'])
    coaches['tmID'] = coaches['tmID'].replace(team_map)

    return coaches


def feature_creation_team_score(teams, players_teams, teams_map=None):
    # Creating Team Score (Predicted and Real)

    team_eff_stats = players_teams.groupby(['year', 'tmID'])['Predicted_EFF'].agg(['sum', 'count']).reset_index()

    # Rename the columns to 'EFF_Sum' and 'EFF_Count'
    team_eff_stats.rename(columns={'sum': 'EFF_Sum', 'count': 'EFF_Count'}, inplace=True)

    if teams_map:
        team_eff_stats['tmID'] = team_eff_stats['tmID'].replace(teams_map)

    # Merge 'team_eff_stats' with 'teams' based on 'year' and 'tmID'
    teams = pd.merge(teams, team_eff_stats, on=['year', 'tmID'], how='left')

    # Calculate the 'TeamScore' by dividing 'EFF_Sum' by 'EFF_Count'
    teams['PredictedTeamScore'] = teams['EFF_Sum'] / teams['EFF_Count']

    # Drop the 'EFF_Sum' and 'EFF_Count' columns if not needed
    teams.drop(['EFF_Sum', 'EFF_Count'], axis=1, inplace=True)

    team_eff_stats = players_teams.groupby(['year', 'tmID'])['EFF'].agg(['sum', 'count']).reset_index()

    # Rename the columns to 'EFF_Sum' and 'EFF_Count'
    team_eff_stats.rename(columns={'sum': 'EFF_Sum', 'count': 'EFF_Count'}, inplace=True)

    # Merge 'team_eff_stats' with 'teams' based on 'year' and 'tmID'
    teams = pd.merge(teams, team_eff_stats, on=['year', 'tmID'], how='left')

    # Calculate the 'TeamScore' by dividing 'EFF_Sum' by 'EFF_Count'
    teams['RealTeamScore'] = teams['EFF_Sum'] / teams['EFF_Count']

    # Drop the 'EFF_Sum' and 'EFF_Count' columns if not needed
    teams.drop(['EFF_Sum', 'EFF_Count'], axis=1, inplace=True)

    return teams


def feature_creation_teams(teams, players_teams):
    print("Feature Creation - Teams")
    teams['progress'] = teams.apply(lambda row: progress(row), axis=1)

    teams['confWLRatio'] = teams['confW'] / (teams['confW'] + teams['confL'])
    teams['awayWLRatio'] = teams['awayW'] / (teams['awayW'] + teams['awayL'])
    teams['homeWLRatio'] = teams['homeW'] / (teams['homeW'] + teams['homeL'])
    teams['gamesWLRatio'] = teams['won'] / (teams['won'] + teams['lost'])

    teams['offensive_performance'] = ((teams['o_pts'] / (teams['o_fga'] + 0.44 * teams['o_fta'])) + (
            (teams['o_fgm'] + teams['o_3pm']) / (teams['o_fga'] + teams['o_3pa'])) + (
                                              teams['o_asts'] / (teams['o_to'] + 1)) + teams['o_oreb']) / teams['GP']
    teams['defensive_performance'] = (((teams['d_fgm'] + teams['d_3pm']) / (teams['d_fga'] + 0.44 * teams['d_fta'])) + (
            (teams['d_fgm'] + teams['d_3pm']) / (teams['d_fga'] + teams['d_3pa'])) + teams['d_dreb'] + teams[
                                          'd_stl'] + teams['d_blk'] - teams['d_pts']) / teams['GP']

    teams = feature_creation_team_score(teams, players_teams)

    # Create the new column 'WLRatioScaled' by multiplying 'WLRatio' by 10
    teams['coachScore'] = teams['WLRatio'] * 10
    teams.drop(['WLRatio'], axis=1, inplace=True)

    return teams

def merge_coaches(teams, coaches):
    teams = pd.merge(teams, coaches[['year', 'tmID', 'WLRatio']], on=['year', 'tmID'], how='left')

    return teams


def teams_data_cleanup(teams, team_map):
    print("Teams Data Cleanup")
    teams = teams.drop(columns=['divID', 'firstRound', 'semis', 'finals', 'lgID', 'seeded', 'arena', 'attend',
                                'min', 'confW', 'confL', 'awayW', 'awayL', 'homeW', 'homeL', 'won', 'lost', 'o_fgm',
                                'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb',
                                'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts',
                                'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb',
                                'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'tmORB',
                                'tmDRB', 'tmTRB', 'opptmORB', 'opptmDRB', 'opptmTRB'])
    teams['tmID'] = teams['tmID'].replace(team_map)

    playoff_mapping = {'Y': 1, 'N': 0}
    teams['playoff'] = teams['playoff'].replace(playoff_mapping)

    return teams

def create_lagged_features_teams(teams, features_to_be_lagged, lag_years):
    print("Create Lagged Features Teams")
    # Create lagged features
    for feat in features_to_be_lagged:
        for year in range(1, lag_years + 1):
            teams[f'{feat}_Lag_{year}'] = teams.groupby('franchID')[feat].shift(year)

    # Fill NaN values in the newly created columns with 0
    lagged_features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                       range(1, lag_years + 1)]
    teams[lagged_features] = teams[lagged_features].fillna(-1)

    return teams

def models_train_and_test_teams(teams, features, target):
    print("Models Running - Teams")
    this_year_records, train_teams = find_and_move_max_year_records(teams)

    x_train = train_teams[features]
    y_train = train_teams[target]
    x_test = this_year_records[features]
    y_test = this_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target)

    # Feature Importance - understanding which features are important:

    # Access feature importances
    feature_importances = trained_models['Random Forest Regressor'].feature_importances_

    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Print or visualize the feature importances
    print(importance_df)

    return trained_models

def models_predict_future_teams(teams, players_teams, teams_map, features, features_to_be_lagged, model, lag_years):
    print("Models Predicting - Teams")
    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each team
    most_recent_years = teams.groupby('franchID')['year'].max().reset_index()

    # Create an empty list to store new rows
    new_rows = []

    # Iterate over unique team IDs and add new rows to the list
    for _, row in most_recent_years.iterrows():
        tmp_data = teams.loc[(teams['franchID'] == row['franchID']) & (teams['year'] == row['year'])]
        team_id = tmp_data.at[tmp_data.index[0], 'tmID']
        franch_id = row['franchID']
        conf_id = tmp_data.at[tmp_data.index[0], 'confID']
        name = tmp_data.at[tmp_data.index[0], 'name']
        max_year = row['year']
        next_year = max_year + 1

        # Create a new row with the next year, team ID, franch id, conf id and name, leaving other columns blank
        new_row = [next_year, team_id, franch_id, conf_id] + [np.nan]*2 + [name] + [np.nan] * (teams.shape[1] - 7)

        # Append the new row to the list
        new_rows.append(new_row)

    # Create the 'future_team_data' DataFrame with columns matching your original data
    future_team_data = pd.DataFrame(new_rows, columns=teams.columns)

    # Concatenate the original DataFrame and the new rows DataFrame
    future_team_data = pd.concat([teams, future_team_data], ignore_index=True)

    # Reset the index of the modified DataFrame
    future_team_data.reset_index(drop=True, inplace=True)

    future_team_data = feature_creation_team_score(future_team_data, players_teams, teams_map)

    future_team_data = create_lagged_features_teams(future_team_data, features_to_be_lagged, lag_years)

    # Use the trained model to predict EFF for the next year
    max_year_data = future_team_data[future_team_data['year'] == future_team_data['year'].max()]
    future_predictions = model.predict(max_year_data[features])

    max_year_data['Predicted_Playoff'] = future_predictions

    future_team_data = pd.merge(future_team_data, max_year_data, on=['year', 'franchID', 'tmID', 'name'], how='left')

    return future_team_data

def main():

    # Load data from CSVs
    dataframes_dict = initial_data_load()

    # Cleanup data on players dataframe
    dataframes_dict['players'] = player_data_cleanup(dataframes_dict['players'])

    # Add number of awards to players dataframe
    dataframes_dict['players'] = merge_players_awards(dataframes_dict['players'], dataframes_dict['awards_players'])

    # Add EFF, DPR, FG%, FT%, PPG features to players_teams dataframe
    dataframes_dict['players_teams'] = feature_creation_players_teams(dataframes_dict['players_teams'])

    # Merge players_teams with players
    dataframes_dict['players_teams'] = merge_players_teams(dataframes_dict['players_teams'], dataframes_dict['players'])

    # creating new features to help predict the EFF for the next year:

    # Mapping teams to indexes
    team_map = team_mapping(set(list(dataframes_dict['series_post']['tmIDLoser'].unique()) + list(dataframes_dict['series_post']['tmIDWinner'].unique())))

    # Clean up data on series_post dataframe - USELESS????
    dataframes_dict['series_post'] = series_post_data_cleanup(dataframes_dict['series_post'], team_map)

    # USELESS ???
    """
    teams_post['tmID'] = teams_post['tmID'].replace(team_mapping)
    teams_post['WinRate'] = teams_post['W'] / (teams_post['W'] + teams_post['L'])
    teams_post.drop(['lgID', 'W', 'L'], inplace=True, axis='columns')
    """

    lag_years_players = 3
    features_to_be_lagged = ['FG_Percentage', 'FT_Percentage', 'PPG', 'EFF']
    dataframes_dict['players_teams'] = create_lagged_features_players(dataframes_dict['players_teams'], features_to_be_lagged, lag_years_players)

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_players + 1)] + ['year']
    target = 'EFF'
    #trained_models_players = models_train_and_test_players(dataframes_dict['players_teams'], features, target)
    trained_models_players = {}
    trained_models_players['Random Forest Regressor'] = joblib.load('models/RandomForestRegressorPlayers.joblib')

    dataframes_dict['players_teams'] = models_predict_future_players(dataframes_dict['players_teams'], features,
                                                       features_to_be_lagged,
                                                       trained_models_players['Random Forest Regressor'],
                                                       lag_years_players)


    print("PLAYERS MODELS DONE")

    dataframes_dict['coaches'] = feature_creation_coaches(dataframes_dict['coaches'])

    dataframes_dict['coaches'] = coaches_data_cleanup(dataframes_dict['coaches'], team_map)

    # DEPOIS DESTE MERGE APARECEM 2 EQUIPAS CUJO TEAM ID NÃO É UM NUMERO - TIVE THE USAR franchID PARA FAZER GROUP BY
    dataframes_dict['teams'] = merge_coaches(dataframes_dict['teams'], dataframes_dict['coaches'])

    dataframes_dict['teams'] = feature_creation_teams(dataframes_dict['teams'], dataframes_dict['players_teams'])

    dataframes_dict['teams'] = teams_data_cleanup(dataframes_dict['teams'], team_map)


    lag_years_teams = 3
    features_to_be_lagged = ['PredictedTeamScore', 'defensive_performance', 'offensive_performance', 'gamesWLRatio', 'homeWLRatio', 'awayWLRatio', 'confWLRatio', 'progress', 'playoff']
    dataframes_dict['teams'] = create_lagged_features_teams(dataframes_dict['teams'],
                                                                      features_to_be_lagged, lag_years_teams)

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_players + 1)] + ['year', 'PredictedTeamScore']
    target = 'playoff'
    #trained_models_teams = models_train_and_test_teams(dataframes_dict['teams'], features, target)
    trained_models_teams = {}
    trained_models_teams['Random Forest Regressor'] = joblib.load('models/RandomForestRegressorTeams.joblib')

    dataframes_dict['teams'] = models_predict_future_teams(dataframes_dict['teams'], dataframes_dict['players_teams'],
                                                           team_map, features, features_to_be_lagged,
                                                           trained_models_teams['Random Forest Regressor'],
                                                           lag_years_teams)


    print(dataframes_dict['teams'].sort_values(by='Predicted_Playoff', ascending=False).head(len(team_map)))


main()
