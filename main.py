import time

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import models

MODEL_PLAYERS_EFF = "Random Forest Regressor_RobustScaler"
"""
Model name for predicting player Efficiency (EFF).
"""
MODEL_PLAYERS_DPR = "Random Forest Regressor_RobustScaler"
"""
Model name for predicting player Defensive Performance Rating (DPR).
"""
MODEL_TEAMS = "Lasso Regression_None"
"""
Model name for predicting team qualification to the playoffs.
"""
MODEL_GAMES_SIM = "Random Forest Regressor_RobustScaler"
"""
Model name for simulating game outcomes.
"""


def progress(row):
    """
        Calculate the playoff progress based on the provided row.

        This function assigns a progress value based on the team's performance in the playoffs.
        The progress is calculated as follows:
        - 0 for teams not in the playoffs
        - 0.5 for teams eliminated in the first round
        - 1.5 for teams reaching the conference semifinals
        - 2.5 for teams reaching the finals
        - 3 for teams winning the championship

        Parameters:
        row (pandas.Series): A row of data containing playoff information.

        Returns:
        float: The calculated playoff progress value.
        """
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
    """
        Separates records corresponding to the maximum year in the dataset and moves them to a separate DataFrame.

        Parameters:
        - players_teams (pd.DataFrame): The input DataFrame containing player-team records.

        Returns:
        - max_year_df (pd.DataFrame): DataFrame containing records for the maximum year.
        - players_teams (pd.DataFrame): DataFrame with records excluding those for the maximum year.
        """
    players_teams = players_teams.copy()
    max_year = players_teams['year'].max()
    max_year_records = players_teams[players_teams['year'] == max_year]
    max_year_df = pd.DataFrame(max_year_records)
    players_teams = players_teams[players_teams['year'] != max_year]
    max_year_df.reset_index(drop=True, inplace=True)
    players_teams.reset_index(drop=True, inplace=True)

    return max_year_df, players_teams


def initial_data_load():
    """
        Loads initial data from CSV files into separate DataFrames and returns a dictionary of DataFrames.

        Returns:
        - dataframes_dict (dict): A dictionary containing DataFrames for different datasets.
        """
    print("Initial Data Load")
    awards_players = pd.read_csv('./basketballPlayoffs/awards_players.csv')
    coaches = pd.read_csv('./basketballPlayoffs/coaches.csv')
    players = pd.read_csv('./basketballPlayoffs/players.csv')
    players_teams = pd.read_csv('./basketballPlayoffs/players_teams.csv')
    series_post = pd.read_csv('./basketballPlayoffs/series_post.csv')
    teams = pd.read_csv('./basketballPlayoffs/teams.csv')
    # teams = teams[teams['year'] != 10]
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
    """
        Cleans up player data by performing various transformations and filtering irrelevant information.

        Parameters:
        - players (pd.DataFrame): DataFrame containing player data.

        Returns:
        - players (pd.DataFrame): Cleaned DataFrame with specified modifications.
        """
    print("Player Data Cleanup")
    # Death Date is irrelevant for the way they play
    players.drop('deathDate', inplace=True, axis='columns')

    # Removing NaN on position
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
    """
        Creates a mapping dictionary for unique college names to numerical indices.

        Parameters:
        - unique_colleges (set): Set containing unique college names.

        Returns:
        - college_mapping_dict (dict): Mapping dictionary with college names as keys and corresponding indices as values.
        """
    college_mapping_dict = {}
    for index, college in enumerate(unique_colleges):
        college_mapping_dict[college] = index

    return college_mapping_dict


def team_mapping(unique_teams):
    """
        Creates a mapping dictionary for unique team names to numerical indices.

        Parameters:
        - unique_teams (set): Set containing unique team names.

        Returns:
        - team_mapping_dict (dict): Mapping dictionary with team names as keys and corresponding indices as values.
        """
    team_mapping_dict = {}
    for index, team in enumerate(unique_teams):
        team_mapping_dict[team] = index

    return team_mapping_dict


def position_mapping(unique_pos):
    """
        Creates a mapping dictionary for unique player positions to numerical indices.

        Parameters:
        - unique_pos (set): Set containing unique player positions.

        Returns:
        - pos_mapping (dict): Mapping dictionary with player positions as keys and corresponding indices as values.
        """
    pos_mapping = {pos: ind for ind, pos in enumerate(unique_pos)}
    return pos_mapping


def merge_players_awards(players, awards_players):
    """
        Merges player data with awards data and calculates the number of awards for each player.

        Parameters:
        - players (pd.DataFrame): DataFrame containing player data.
        - awards_players (pd.DataFrame): DataFrame containing awards data for players.

        Returns:
        - players (pd.DataFrame): DataFrame with merged data and a new column 'numAwards'.
        """
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


def feature_creation_players_teams(players_teams, predicting=False):
    """
        Creates additional features for player-team data, including Efficiency, Defense Score, FG Percentage,
        FT Percentage, and Points Per Game.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - predicting (bool): Flag indicating whether the function is used for prediction (default is False).

        Returns:
        - players_teams (pd.DataFrame): DataFrame with added features.
        """
    print("Feature Creation - Players Teams")
    # Creating Efficiency column of the players
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
    if not predicting:
        players_teams.dropna(subset=['FG_Percentage', 'FT_Percentage', 'PPG'], inplace=True)

    return players_teams


def merge_players_teams(players_teams, players):
    """
        Merges player-team data with player data based on the 'playerID' and 'bioID' columns.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - players (pd.DataFrame): DataFrame containing player data.

        Returns:
        - players_teams (pd.DataFrame): DataFrame with merged data.
        """
    print("Merge Player Teams")
    players_teams = pd.merge(players_teams, players, left_on='playerID', right_on='bioID',
                             how='outer').drop_duplicates()

    return players_teams


def series_post_data_cleanup(series_post, team_mapping_dict):
    """
        Cleans up series post data by performing various transformations and encoding.

        Parameters:
        - series_post (pd.DataFrame): DataFrame containing series post data.
        - team_mapping_dict (dict): Mapping dictionary for team names to numerical indices.

        Returns:
        - series_post (pd.DataFrame): Cleaned DataFrame with specified modifications.
        """
    print("Series Post Data Cleanup")
    series_post.drop(['lgIDLoser', 'lgIDWinner'], inplace=True, axis='columns')

    label_encoder = LabelEncoder()
    series_post['series'] = label_encoder.fit_transform(series_post['series'])

    series_post['tmIDWinner'] = series_post['tmIDWinner'].replace(team_mapping_dict)
    series_post['tmIDLoser'] = series_post['tmIDLoser'].replace(team_mapping_dict)

    total_games = series_post['W'] + series_post['L']
    series_post['W'] = series_post['W'] / total_games
    series_post['L'] = series_post['L'] / total_games

    return series_post


def create_lagged_features_players(players_teams, features_to_be_lagged, lag_years):
    """
        Creates lagged features for specified columns in player-team data.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - features_to_be_lagged (list): List of column names for which lagged features are to be created.
        - lag_years (int): Number of years to lag the features.

        Returns:
        - players_teams (pd.DataFrame): DataFrame with lagged features.
        """
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


def models_train_and_test_players_dpr(players_teams, features, target):
    """
        Trains and tests machine learning models on player-team data for predicting Defense Score (DPR).

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - features (list): List of feature columns used for training the models.
        - target (str): Target column to be predicted.

        Returns:
        - trained_models (dict): Dictionary containing trained machine learning models.
        """
    print("Models Running - Players")
    players_teams = players_teams.copy()
    players_teams = players_teams.dropna()
    last_year_records, players_train_teams = find_and_move_max_year_records(players_teams)

    x_train = players_train_teams[features]
    y_train = players_train_teams[target]
    x_test = last_year_records[features]
    y_test = last_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target, "Players")

    features.remove('year')

    return trained_models


def models_train_and_test_players(players_teams, features, target):
    """
        Trains and tests machine learning models on player-team data.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - features (list): List of feature columns used for training the models.
        - target (str): Target column to be predicted.

        Returns:
        - trained_models (dict): Dictionary containing trained machine learning models.
        """
    print("Models Running - Players")
    last_year_records, players_train_teams = find_and_move_max_year_records(players_teams)

    x_train = players_train_teams[features]
    y_train = players_train_teams[target]
    x_test = last_year_records[features]
    y_test = last_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target, "Players")

    features.remove('year')

    return trained_models


def models_predict_future_players_dpr(players_teams, features, features_to_be_lagged, model, lag_years, predict_column):
    """
        Predicts Defense Score (DPR) for future years using a trained machine learning model.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - features (list): List of feature columns used for prediction.
        - features_to_be_lagged (list): List of column names for which lagged features are to be created.
        - model: Trained machine learning model.
        - lag_years (int): Number of years to lag the features for prediction.
        - predict_column (str): Column name to store the predicted values.

        Returns:
        - future_player_data (pd.DataFrame): DataFrame with predicted Defense Score for future years.
        """
    print("Models Predicting - Players")
    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each player
    most_recent_years = players_teams.groupby('playerID')['year'].max().reset_index()

    # Concatenate the original DataFrame and the new rows DataFrame
    future_player_data = players_teams.copy()

    future_player_data = create_lagged_features_players(future_player_data, features_to_be_lagged, lag_years)

    # Use the trained model to predict DPR for the next year
    future_predictions = model.predict(scale_data(RobustScaler(), future_player_data[features]))

    future_player_data[predict_column] = future_predictions

    return future_player_data


def scale_data(scaler, x_train):
    """
        Scales the input data using the specified scaler.

        Parameters:
        - scaler: Scaler object for data scaling.
        - x_train (pd.DataFrame): Input DataFrame to be scaled.

        Returns:
        - x_train_scaled (pd.DataFrame): Scaled DataFrame.
        """
    columns = x_train.columns
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(data=x_train, columns=columns)
    return x_train


def models_predict_future_players(players_teams, features, features_to_be_lagged, model, lag_years, predict_column):
    """
        Predicts future values for a specified column using a trained machine learning model.

        Parameters:
        - players_teams (pd.DataFrame): DataFrame containing player-team data.
        - features (list): List of feature columns used for prediction.
        - features_to_be_lagged (list): List of column names for which lagged features are to be created.
        - model: Trained machine learning model.
        - lag_years (int): Number of years to lag the features for prediction.
        - predict_column (str): Column name to store the predicted values.

        Returns:
        - future_player_data (pd.DataFrame): DataFrame with predicted values for future years.
        """
    print("Models Predicting - Players")
    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each player
    most_recent_years = players_teams.groupby('playerID')['year'].max().reset_index()

    # Create an empty list to store new rows
    new_rows = []

    # Iterate over unique player IDs and add new rows to the list
    for _, row in most_recent_years.iterrows():
        tmp_data = players_teams.loc[
            (players_teams['playerID'] == row['playerID']) & (players_teams['year'] == row['year'])]
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
    future_predictions = model.predict(scale_data(RobustScaler(), future_player_data[features]))

    future_player_data[predict_column] = future_predictions

    return future_player_data


def feature_creation_coaches(coaches):
    """
        Creates additional features for coach data, including Win-Loss Ratio (WLRatio) and Postseason Win-Loss Ratio (WLRatio_Post).

        Parameters:
        - coaches (pd.DataFrame): DataFrame containing coach data.

        Returns:
        - coaches (pd.DataFrame): DataFrame with added features.
        """
    coaches['WLRatio'] = coaches['won'] / (coaches['won'] + coaches['lost'])
    coaches['WLRatio_Post'] = coaches['post_wins'] / (coaches['post_wins'] + coaches['post_losses'])

    return coaches


def coaches_data_cleanup(coaches):
    """
        Cleans up coach data by sorting it based on 'stint' in descending order and keeping only the first entry for each year and team.

        Parameters:
        - coaches (pd.DataFrame): DataFrame containing coach data.

        Returns:
        - coaches (pd.DataFrame): Cleaned DataFrame with specified modifications.
        """
    coaches = coaches.sort_values(by='stint', ascending=False)
    coaches = coaches.groupby(['year', 'tmID']).first().reset_index()
    return coaches


def feature_creation_team_score(teams, players_teams, teams_map=None):
    """
        Creates Team Score features (Predicted and Real) and adds them to the 'teams' DataFrame.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - players_teams (pd.DataFrame): DataFrame containing player-team data with predicted and real Efficiency (EFF).
        - teams_map (dict): Mapping dictionary for team names to numerical indices (default is None).

        Returns:
        - teams (pd.DataFrame): DataFrame with added Team Score features.
        """
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

    if teams_map:
        team_eff_stats['tmID'] = team_eff_stats['tmID'].replace(teams_map)

    # Merge 'team_eff_stats' with 'teams' based on 'year' and 'tmID'
    teams = pd.merge(teams, team_eff_stats, on=['year', 'tmID'], how='left')

    # Calculate the 'TeamScore' by dividing 'EFF_Sum' by 'EFF_Count'
    teams['RealTeamScore'] = teams['EFF_Sum'] / teams['EFF_Count']

    # Drop the 'EFF_Sum' and 'EFF_Count' columns if not needed
    teams.drop(['EFF_Sum', 'EFF_Count'], axis=1, inplace=True)

    return teams


def feature_creation_teams(teams, players_teams):
    """
        Creates additional features for team data and adds them to the 'teams' DataFrame.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - players_teams (pd.DataFrame): DataFrame containing player-team data.

        Returns:
        - teams (pd.DataFrame): DataFrame with added features.
        """
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
    """
        Merges coach data with team data based on the 'year' and 'tmID' columns.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - coaches (pd.DataFrame): DataFrame containing coach data.

        Returns:
        - teams (pd.DataFrame): Merged DataFrame with coach data.
        """
    teams = pd.merge(teams, coaches[['year', 'tmID', 'WLRatio']], on=['year', 'tmID'], how='left')

    return teams


def teams_data_cleanup(teams, team_map):
    """
        Cleans up team data by replacing team names with numerical indices and mapping playoff values to binary.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - team_map (dict): Mapping dictionary for team names to numerical indices.

        Returns:
        - teams (pd.DataFrame): Cleaned DataFrame with specified modifications.
        """
    print("Teams Data Cleanup")
    teams['tmID'] = teams['tmID'].replace(team_map)

    playoff_mapping = {'Y': 1, 'N': 0}
    teams['playoff'] = teams['playoff'].replace(playoff_mapping)

    return teams


def create_lagged_features_teams(teams, features_to_be_lagged, lag_years):
    """
        Creates lagged features for specified columns in the 'teams' DataFrame.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - features_to_be_lagged (list): List of column names for which lagged features are to be created.
        - lag_years (int): Number of years to lag the features.

        Returns:
        - teams (pd.DataFrame): DataFrame with added lagged features.
        """
    print("Create Lagged Features Teams")
    # Create lagged features
    for feat in features_to_be_lagged:
        for year in range(1, lag_years + 1):
            teams[f'{feat}_Lag_{year}'] = teams.groupby('tmID')[feat].shift(year)

    # Fill NaN values in the newly created columns with 0
    lagged_features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                       range(1, lag_years + 1)]
    teams[lagged_features] = teams[lagged_features].fillna(-1)
    teams[[i for i in lagged_features if "playoff" in i]] = teams[
        [i for i in lagged_features if "playoff" in i]].replace(-1, 0.5)

    return teams


def models_train_and_test_teams(teams, features, target):
    """
        Trains and tests machine learning models for team data.

        Parameters:
        - teams (pd.DataFrame): DataFrame containing team data.
        - features (list): List of feature column names.
        - target (str): Target column name.

        Returns:
        - trained_models: Result of running machine learning models.
        """
    print("Models Running - Teams")
    this_year_records, train_teams = find_and_move_max_year_records(teams)
    x_train = train_teams[features]
    y_train = train_teams[target]
    x_test = this_year_records[features]
    y_test = this_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target, "Teams")

    features.remove('year')

    return trained_models


def models_predict_future_teams(teams, players_teams, teams_map, features, features_to_be_lagged, model, lag_years):
    """
       Predicts future team performance using machine learning models.

       Parameters:
       - teams (pd.DataFrame): DataFrame containing team data.
       - players_teams (pd.DataFrame): DataFrame containing player-team data.
       - teams_map (dict): Mapping dictionary for team names to numerical indices.
       - features (list): List of feature column names.
       - features_to_be_lagged (list): List of column names for which lagged features are to be created.
       - model: Trained machine learning model.
       - lag_years (int): Number of years to lag the features.

       Returns:
       - future_team_data (pd.DataFrame): DataFrame with predicted future team performance.
       """
    print("Models Predicting - Teams")
    # Creating dataframe with the next years predicted EFF for each player

    # Find the most recent year for each team
    most_recent_years = teams.groupby('tmID')['year'].max().reset_index()

    # Create an empty list to store new rows
    new_rows = []

    # Iterate over unique team IDs and add new rows to the list
    for _, row in most_recent_years.iterrows():
        tmp_data = teams.loc[(teams['tmID'] == row['tmID']) & (teams['year'] == row['year'])]
        team_id = tmp_data.at[tmp_data.index[0], 'tmID']
        franch_id = row['tmID']
        conf_id = tmp_data.at[tmp_data.index[0], 'confID']
        name = tmp_data.at[tmp_data.index[0], 'name']
        max_year = row['year']
        next_year = max_year + 1

        # Create a new row with the next year, team ID, franch id, conf id and name, leaving other columns blank
        new_row = [next_year, team_id, franch_id, conf_id] + [np.nan] * 2 + [name] + [np.nan] * (teams.shape[1] - 7)

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

    future_predictions = model.predict(scale_data(None, max_year_data[features]))

    max_year_data['Predicted_Playoff'] = future_predictions

    future_team_data = pd.merge(future_team_data, max_year_data, on=['year', 'tmID', 'tmID', 'name'], how='left')

    return future_team_data


def turn_series_post_into_no_bias(series_post):
    """
        Converts the 'series_post' DataFrame into a format without bias, where team1 and team2 are randomly assigned.

        Parameters:
        - series_post (pd.DataFrame): DataFrame containing playoff series data.

        Returns:
        - series_post (pd.DataFrame): Transformed DataFrame with team1, team2, team1wins, and team2wins columns.
        """
    # Create new columns for team1 and team2
    series_post['team1'] = np.where(np.random.rand(len(series_post)) < 0.5, series_post['tmIDWinner'],
                                    series_post['tmIDLoser'])
    series_post['team2'] = np.where(series_post['team1'] == series_post['tmIDWinner'], series_post['tmIDLoser'],
                                    series_post['tmIDWinner'])

    # Adjust W and L columns to represent team1wins and team2wins
    series_post['team1wins'] = np.where(series_post['team1'] == series_post['tmIDWinner'], series_post['W'],
                                        series_post['L'])
    series_post['team2wins'] = np.where(series_post['team2'] == series_post['tmIDWinner'], series_post['W'],
                                        series_post['L'])

    # Drop unnecessary columns if needed
    series_post.drop(columns=['W', 'L', 'tmIDWinner', 'tmIDLoser'], inplace=True)

    return series_post


def merge_teams(series_post, teams):
    """
        Merges the 'series_post' DataFrame with the 'teams' DataFrame for team1 and team2.

        Parameters:
        - series_post (pd.DataFrame): DataFrame containing playoff series data.
        - teams (pd.DataFrame): DataFrame containing team data.

        Returns:
        - series_post (pd.DataFrame): Merged DataFrame with team1 and team2 data.
        """
    # Merge for team1 using suffix '_team1'
    merged_team1 = pd.merge(series_post, teams, left_on=['year', 'team1'], right_on=['year', 'tmID'], how='inner')

    # Merge for team2 using suffix '_team2'
    merged_team2 = pd.merge(series_post, teams, left_on=['year', 'team2'], right_on=['year', 'tmID'], how='inner')

    series_post = pd.merge(merged_team1, merged_team2, on=['year', 'team1', 'team2'], how='inner',
                           suffixes=('_team1', '_team2'))

    return series_post


def models_train_and_test_games(series_post, features, target):
    """
        Trains and tests machine learning models for simulating game outcomes.

        Parameters:
        - series_post (pd.DataFrame): DataFrame containing playoff series data.
        - features (list): List of feature column names.
        - target (str): Target column name.

        Returns:
        - trained_models: Trained machine learning models.
        """
    print("Models Running - Games Simulation")
    this_year_records, train_teams = find_and_move_max_year_records(series_post)
    x_train = train_teams[features]
    y_train = train_teams[target]
    x_test = this_year_records[features]
    y_test = this_year_records[target]
    trained_models = models.run_all(x_train, y_train, x_test, y_test, 3, 7, target, "GamesSimulator")

    features.remove('year')

    return trained_models


def simulate_games(teams, model, features):
    """
        Trains and tests machine learning models for simulating game outcomes.
        Alternative approach to predict the playoffs' qualification.

        Parameters:
        - series_post (pd.DataFrame): DataFrame containing playoff series data.
        - features (list): List of feature column names.
        - target (str): Target column name.

        Returns:
        - trained_models: Trained machine learning models.
        """
    columns = features.copy()
    for feature in features:
        features[features.index(feature)] = feature[:-5] + 'x'

    features = list(set(features))

    res_df = pd.DataFrame(columns=['year', 'team', 'teamWins'])

    for most_recent_year in range(1, teams['year'].max() + 1):

        filtered_teams = teams.loc[teams['year'] == most_recent_year, ['tmID'] + features]

        # Keep unique rows based on 'tmID' column
        teams_clean = filtered_teams.drop_duplicates(subset='tmID')

        # Create an empty list to store individual DataFrames
        prepared_data = []

        # Iterate over each team1 in teams_clean
        for idx, team1_row in teams_clean.iterrows():
            team1_id = team1_row['tmID']

            # Iterate over all other teams except team1
            for _, team2_row in teams_clean[teams_clean['tmID'] != team1_id].iterrows():
                team2_id = team2_row['tmID']

                # Create a dictionary to store the row data
                row_data = {}

                row_data['team1'] = team1_id
                row_data['team2'] = team2_id

                for col in columns:
                    if col[-1] == '1':
                        row_data[col] = team1_row[col[:-5] + 'x']
                    if col[-1] == '2':
                        row_data[col] = team2_row[col[:-5] + 'x']

                # Convert the row data to a DataFrame and append it to the list
                prepared_data.append(pd.DataFrame([row_data]))

        # Concatenate all individual DataFrames into a single DataFrame
        prepared_db = pd.concat(prepared_data, ignore_index=True)
        prepared_db.dropna(inplace=True)

        y_pred = model.predict(prepared_db[columns])
        prepared_db['team1Wins'] = y_pred
        prepared_db['team2Wins'] = 1 - prepared_db['team1Wins']

        team1_wins_sum = prepared_db.groupby('team1')['team1Wins'].sum().reset_index()
        team2_wins_sum = prepared_db.groupby('team2')['team2Wins'].sum().reset_index()

        merged_df = pd.merge(team1_wins_sum, team2_wins_sum, how='inner', left_on='team1', right_on='team2')
        merged_df['teamWins'] = merged_df['team1Wins'] + merged_df['team2Wins']

        total_wins = merged_df[['team1', 'teamWins']]
        total_wins.columns = ['team', 'teamWins']
        total_wins = total_wins.copy()
        total_wins.loc[:, 'year'] = most_recent_year

        res_df = pd.concat([res_df, total_wins])

    return res_df


def data_profiling():
    """
        Generate data profiling reports for each DataFrame in the project.
        Creates files with the performed analysis
        Returns: None
        """
    from ydata_profiling import ProfileReport
    import os

    dataframes_dict = initial_data_load()

    if not os.path.exists("data_reports"):
        os.makedirs("data_reports")

    for name, df in dataframes_dict.items():
        profile = ProfileReport(df, title="Profiling Report")
        profile.to_file(os.path.join("data_reports", f"{name}.html"))


def eleventh_year_data_load():
    """
        Load data for predicting the eleventh year.

        Returns:
        dict: A dictionary containing DataFrames for coaches, players_teams, and teams.
        """
    print("Predict Data Load")
    players_teams = pd.read_csv('./Season 11/players_teams.csv')
    coaches = pd.read_csv('./Season 11/coaches.csv')
    teams = pd.read_csv('./Season 11/teams.csv')

    dataframes_dict = {
        'coaches': coaches,
        'players_teams': players_teams,
        'teams': teams,
    }

    return dataframes_dict


def merge_dataframes(df1, df2):
    """
        Merge two dictionaries of DataFrames.

        Args:
        df1 (dict): The first dictionary of DataFrames.
        df2 (dict): The second dictionary of DataFrames.

        Returns:
        dict: Merged dictionary containing DataFrames from both input dictionaries.
        """
    for k in df1.keys():
        if k in df2:
            df1[k] = pd.concat([df1[k], df2[k]])

    for k in df2.keys():
        if k not in df1:
            df1[k] = df2[k]

    return df1


def predict_11th_year():
    """
        Predict player and team performance for the 11th year of NBA data.

        This function loads, processes, and predicts player and team metrics for the 11th year of NBA data.
        It includes data loading, feature engineering, and using pre-trained machine learning models for predictions.

        Returns:
        None
        """
    # Load data from CSVs
    dataframes_dict = initial_data_load()
    predict_year_dict = eleventh_year_data_load()

    complete_dataframe_dict = merge_dataframes(dataframes_dict, predict_year_dict)
    dataframes_dict = complete_dataframe_dict

    # Cleanup data on players dataframe
    dataframes_dict['players'] = player_data_cleanup(dataframes_dict['players'])

    # Add number of awards to players dataframe
    dataframes_dict['players'] = merge_players_awards(dataframes_dict['players'], dataframes_dict['awards_players'])

    # Add EFF, DPR, FG%, FT%, PPG features to players_teams dataframe
    dataframes_dict['players_teams'] = feature_creation_players_teams(dataframes_dict['players_teams'], predicting=True)

    # Merge players_teams with players
    dataframes_dict['players_teams'] = merge_players_teams(dataframes_dict['players_teams'], dataframes_dict['players'])

    # creating new features to help predict the EFF for the next year:

    lag_years_players = 7
    features_to_be_lagged = ['FG_Percentage', 'FT_Percentage', 'PPG', 'EFF', 'DPR']
    dataframes_dict['players_teams'] = create_lagged_features_players(dataframes_dict['players_teams'],
                                                                      features_to_be_lagged, lag_years_players)

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_players + 1)]
    name_file = MODEL_PLAYERS_EFF.replace(' ', '').split('_')
    name_file.insert(1, 'Players')
    name_file.append('EFF')
    name_file = '_'.join(name_file)
    eff_model = joblib.load('models/' + name_file + '.joblib')

    # transforming player_teams to just year 11
    player_teams = dataframes_dict['players_teams']
    results = eff_model.predict(player_teams[features])
    player_teams['Predicted_EFF'] = results

    # dpr

    name_file = MODEL_PLAYERS_DPR.replace(' ', '').split('_')
    name_file.insert(1, 'Players')
    name_file.append('EFF')
    name_file = '_'.join(name_file)
    dpr_model = joblib.load('models/' + name_file + '.joblib')
    results = dpr_model.predict(player_teams[features])
    player_teams['Predicted_DPR'] = results

    dataframes_dict['players_teams'] = player_teams

    # teams

    # Mapping teams to indexes
    team_map = team_mapping(list(dataframes_dict['players_teams']['tmID'].unique()))

    dataframes_dict['coaches'] = feature_creation_coaches(dataframes_dict['coaches'])
    dataframes_dict['coaches'] = coaches_data_cleanup(dataframes_dict['coaches'])

    dataframes_dict['teams'] = merge_coaches(dataframes_dict['teams'], dataframes_dict['coaches'])
    dataframes_dict['teams'] = feature_creation_teams(dataframes_dict['teams'], dataframes_dict['players_teams'])
    dataframes_dict['teams'] = teams_data_cleanup(dataframes_dict['teams'], team_map)

    lag_years_teams = 3
    features_to_be_lagged = ['RealTeamScore', 'defensive_performance', 'offensive_performance', 'gamesWLRatio',
                             'homeWLRatio', 'awayWLRatio', 'confWLRatio', 'progress', 'playoff']
    dataframes_dict['teams'] = create_lagged_features_teams(dataframes_dict['teams'],
                                                            features_to_be_lagged, lag_years_teams)

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_teams + 1)] + ['PredictedTeamScore']

    name_file = MODEL_TEAMS.replace(' ', '').split('_')
    name_file.insert(1, 'Teams')
    name_file.append('playoff')
    name_file = '_'.join(name_file)
    playoff_model = joblib.load('models/' + name_file + '.joblib')
    results = playoff_model.predict(dataframes_dict['teams'][features])
    dataframes_dict['teams']['Predicted_Playoff'] = results

    dataframes_dict['teams'] = dataframes_dict['teams'].sort_values(by='Predicted_Playoff', ascending=False)
    reverse_team_map = {value: key for key, value in team_map.items()}
    dataframes_dict['teams']['tmID'] = dataframes_dict['teams']['tmID'].replace(reverse_team_map)
    print(dataframes_dict['teams'][dataframes_dict['teams']['year'] == 11].head(len(team_map)))


def main():
    """
        Main function for NBA playoff prediction.

        This function orchestrates the entire workflow, including data loading, cleaning, feature engineering,
        training and testing of machine learning models for player and team predictions, playoff predictions,
        game simulation, and evaluation of the model's performance.

        Returns:
        None
        """
    start = time.time()
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
    team_map = team_mapping(list(dataframes_dict['teams']['tmID'].unique()))

    lag_years_players = 7
    features_to_be_lagged = ['FG_Percentage', 'FT_Percentage', 'PPG', 'EFF', 'DPR']
    dataframes_dict['players_teams'] = create_lagged_features_players(dataframes_dict['players_teams'],
                                                                      features_to_be_lagged, lag_years_players)

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_players + 1)] + ['year']
    target = 'EFF'
    trained_models_players = models_train_and_test_players(dataframes_dict['players_teams'], features, target)
    eff_model = trained_models_players[MODEL_PLAYERS_EFF]
    dataframes_dict['players_teams'] = models_predict_future_players(dataframes_dict['players_teams'], features,
                                                                     features_to_be_lagged,
                                                                     eff_model,
                                                                     lag_years_players, 'Predicted_EFF')
    features.append('Predicted_EFF')
    features.append('year')
    target = 'DPR'

    trained_models_players = models_train_and_test_players_dpr(dataframes_dict['players_teams'], features, target)
    dpr_model = trained_models_players[MODEL_PLAYERS_DPR]

    dataframes_dict['players_teams'] = models_predict_future_players_dpr(dataframes_dict['players_teams'], features,
                                                                         features_to_be_lagged,
                                                                         dpr_model,
                                                                         lag_years_players, 'Predicted_DPR')
    print("PLAYERS MODELS DONE")

    dataframes_dict['coaches'] = feature_creation_coaches(dataframes_dict['coaches'])
    dataframes_dict['coaches'] = coaches_data_cleanup(dataframes_dict['coaches'])

    dataframes_dict['teams'] = merge_coaches(dataframes_dict['teams'], dataframes_dict['coaches'])
    dataframes_dict['teams'] = feature_creation_teams(dataframes_dict['teams'], dataframes_dict['players_teams'])
    dataframes_dict['teams'] = teams_data_cleanup(dataframes_dict['teams'], team_map)

    lag_years_teams = 3
    features_to_be_lagged = ['RealTeamScore', 'defensive_performance', 'offensive_performance', 'gamesWLRatio',
                             'homeWLRatio', 'awayWLRatio', 'confWLRatio', 'progress', 'playoff']
    dataframes_dict['teams'] = create_lagged_features_teams(dataframes_dict['teams'],
                                                            features_to_be_lagged, lag_years_teams)

    # data preparation for game simulation

    dataframes_dict['series_post'] = series_post_data_cleanup(dataframes_dict['series_post'], team_map)

    dataframes_dict['series_post'] = turn_series_post_into_no_bias(dataframes_dict['series_post'])

    dataframes_dict['series_post'] = merge_teams(dataframes_dict['series_post'], dataframes_dict['teams'])

    # Select relevant features including lagged features
    features_games = [f'{feat}_Lag_{year}_{team}' for feat in features_to_be_lagged for year in
                      range(1, lag_years_teams + 1) for team in ['team1', 'team2']] + ['year'] + [
                         'PredictedTeamScore' + team for team in ['_team1', '_team2']]
    target = 'team1wins_team1'
    trained_models_for_games = models_train_and_test_games(dataframes_dict['series_post'], features_games, target)

    print("TEAMS PLAYOFF PREDICT")

    # Select relevant features including lagged features
    features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                range(1, lag_years_teams + 1)] + ['year', 'PredictedTeamScore']
    target = 'playoff'
    trained_models_teams = models_train_and_test_teams(dataframes_dict['teams'], features, target)

    dataframes_dict['teams'] = models_predict_future_teams(dataframes_dict['teams'], dataframes_dict['players_teams'],
                                                           team_map, features, features_to_be_lagged,
                                                           trained_models_teams[MODEL_TEAMS],
                                                           lag_years_teams)

    dataframes_dict['teams'] = dataframes_dict['teams'].sort_values(by='Predicted_Playoff', ascending=False)
    print(dataframes_dict['teams'].head(len(team_map)))

    print("PREDICT NUMBER OF GAMES WON BY EACH TEAM")

    games_to_be_won = simulate_games(dataframes_dict['teams'],
                                     trained_models_for_games[MODEL_GAMES_SIM],
                                     features_games)

    print(games_to_be_won.sort_values(by=['year', 'teamWins'], ascending=False).head(len(team_map)))

    games_to_be_won = games_to_be_won.merge(dataframes_dict['teams'][['year', 'tmID', 'playoff_x']],
                                            left_on=['team', 'year'],
                                            right_on=['tmID', 'year'], how='left')

    games_to_be_won.dropna(inplace=True)

    # Rank teams within each year based on 'teamWins' in descending order
    games_to_be_won['rank'] = games_to_be_won.groupby('year')['teamWins'].rank(method='min', ascending=False)

    # Assign 'Y' to top 8 teams and 'N' to the rest
    games_to_be_won['playoffs'] = games_to_be_won['rank'].apply(lambda x: 'Y' if x <= 8 else 'N')

    # Drop the intermediate 'rank' column if needed
    games_to_be_won.drop(columns=['rank'], inplace=True)

    metrics_data = games_to_be_won.copy()

    metrics_data["playoff_x"][metrics_data["playoff_x"] == 1] = 'Y'
    metrics_data["playoff_x"][metrics_data["playoff_x"] == 0] = 'N'

    metrics_data = metrics_data[metrics_data["year"] == 10]

    y_test = metrics_data['playoff_x']
    y_pred = metrics_data['playoffs']

    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    print("Game Simulation (Metrics):\n")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100.0}%")
    precision = precision_score(y_test, y_pred, pos_label='Y')
    print(f"Precision: {precision * 100.0}%")
    recall = recall_score(y_test, y_pred, pos_label='Y')
    print(f"Recall: {recall * 100.0}%")
    f1 = f1_score(y_test, y_pred, pos_label='Y')
    print(f"F1 Score: {f1 * 100.0}%")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(conf_matrix)
    end = time.time()
    print("Elapsed time: " + str(end - start))


main()
