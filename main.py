import pandas as pd


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
    print(unique_colleges)
    college_mapping = {}
    for index, college in enumerate(unique_colleges):
        college_mapping[college] = index

    players['college'] = players['college'].replace(college_mapping)
    players['collegeOther'] = players['collegeOther'].replace(college_mapping)

    print(players)

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

    print(players.describe())
    print(players.loc[:, players.columns != 'bioID'].corr())



main()
