import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

    
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

    # print(players.describe())

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

    # First Season and Last Season were always 0 so we decided to remove them
    players = players.drop(['firstseason', 'lastseason'], axis='columns')

    # Converting positions to an index
    unique_pos = set(players['pos'].unique())
    pos_mapping = {pos: ind for ind, pos in enumerate(unique_pos)}

    players['pos'] = players['pos'].replace(pos_mapping)

    # Day and month of birth is irrelevant in our view therefore we will save only the birthYear
    players['birthDate'] = pd.to_datetime(players['birthDate'])
    players['birthYear'] = players['birthDate'].dt.year
    players = players.drop('birthDate', axis='columns')
    print(players.loc[:, players.columns != 'bioID'].corr())
    
    #remove divID that is NaN in all objects
    print(teams['divID'].isna().sum())
    teams.drop('divID', inplace = True, axis = 'columns')

    #merge players and awards_players by bioID and playerID
    players = pd.merge(players,awards_players, left_on='bioID', right_on='playerID', how ='inner')
    del awards_players
    players_teams['EFF'] = (players_teams['points'] + players_teams['rebounds'] + players_teams['assists']+ players_teams['steals']+ players_teams['blocks']- (players_teams['fgAttempted'] - players_teams['fgMade'])- (players_teams['ftAttempted'] - players_teams['ftMade'])- np.where(players_teams['GP'] == 0, 0, players_teams['turnovers'])) / players_teams['GP']
    
    players_teams = pd.merge(players_teams,players, left_on='playerID', right_on='bioID', how = 'inner')

    series_post.drop(['lgIDLoser', 'lgIDWinner'], inplace=True, axis='columns')

    unique_teams = set(list(series_post['tmIDLoser'].unique()) + list(series_post['tmIDWinner'].unique()))
    team_mapping = {}
    for index, team in enumerate(unique_teams):
        team_mapping[team] = index

    series_post['tmIDLoser'] = series_post['tmIDLoser'].replace(team_mapping)
    series_post['tmIDWinner'] = series_post['tmIDWinner'].replace(team_mapping)

    label_encoder = LabelEncoder()
    series_post['series'] = label_encoder.fit_transform(series_post['series'])
    round_mapping = {'FR': 1, 'CF':2, 'F':3}
    series_post['round'] = series_post['round'].replace(round_mapping)
    print()

main()
