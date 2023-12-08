import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
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


def coaches_analysis(dataframes_dict):
    # Check for null values
    print(dataframes_dict['coaches'].isnull().sum())

    # Statistics of numerical values
    print(dataframes_dict['coaches'].describe())

    # Information about columns of this table
    print(dataframes_dict['coaches'].info())

    # Displays the number of different coaches a team had during the 10 years
    unique_coaches_per_team = dataframes_dict['coaches'].drop_duplicates(subset=['coachID', 'tmID'])
    total_coaches_per_team = unique_coaches_per_team.groupby('tmID')['coachID'].count().reset_index()
    unique_teams = dataframes_dict['teams'][['tmID', 'name']].drop_duplicates()
    total_coaches_per_team = total_coaches_per_team.merge(unique_teams, on='tmID', how='left')

    plt.figure(figsize=(12, 6))
    bars = plt.bar(total_coaches_per_team['name'], total_coaches_per_team['coachID'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Team Name')
    plt.ylabel('Number of Unique Coaches (Last 10 Years)')
    plt.title('Number of Different Coaches per Team in the Last 10 Years')
    plt.tight_layout()
    plt.show()

    # Displays the top 20 coaches with the highest Win-Lost Ratio
    dataframes_dict['coaches']['WLRatio'] = dataframes_dict['coaches']['won'] / (
            dataframes_dict['coaches']['won'] + dataframes_dict['coaches']['lost'])

    best_coaches = dataframes_dict['coaches'].sort_values(by='WLRatio', ascending=False).head(35)

    plt.figure(figsize=(10, 6))
    plt.barh(best_coaches['coachID'], best_coaches['WLRatio'], color='skyblue')
    plt.xlabel('Win-Loss Ratio')
    plt.ylabel('Coach ID')
    plt.title('Top 20 Coaches by Win-Loss Ratio')
    plt.gca().invert_yaxis()
    plt.show()


def awards_analysis(dataframes_dict):
    # Check for null values
    print(dataframes_dict['awards_players'].isnull().sum())

    # No need to describe() - just one numerical value

    # Information about columns of this table
    print(dataframes_dict['awards_players'].info())

    # Number of awards of each type
    plt.figure(figsize=(12, 6))
    sns.countplot(x='award', data=dataframes_dict['awards_players'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Displays the top 10 players with the most awards won
    award_counts = dataframes_dict['awards_players'].groupby('playerID')['award'].count().reset_index()

    award_counts = award_counts.sort_values(by='award', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(award_counts['playerID'], award_counts['award'], color='skyblue')
    plt.xlabel('Player ID')
    plt.ylabel('Number of awards')
    plt.title('Top 10 Players with more awards')
    plt.gca().invert_yaxis()
    plt.show()


def players_teams_analysis(dataframes_dict):
    # Check for null values
    print(dataframes_dict['players_teams'].isnull().sum())

    # Statistics of numerical values
    print(dataframes_dict['players_teams'].describe())

    # Information about columns of this table
    print(dataframes_dict['players_teams'].info())

    # Correlation matrix of players_teams
    numerical_data = dataframes_dict['players_teams'].select_dtypes(include='int64')
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()


def players_analysis(dataframes_dict):
    # Check for null values
    print(dataframes_dict['players'].isnull().sum())

    # Statistics of numerical values
    print(dataframes_dict['players'].describe())

    # Information about columns of this table
    print(dataframes_dict['players'].info())

    # See that deathDate is almost the same to every player, so it doesn't add anything
    print((dataframes_dict['players']['deathDate'] != '0000-00-00').sum())

    # Correlation matrix of players
    numerical_data = dataframes_dict['players'].select_dtypes(include=['int64', 'float64'])
    correlation_matrix_players = numerical_data.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix_players, annot=True, cmap='coolwarm')
    plt.show()


def teams_analysis(dataframes_dict):
    # Check for null values
    print(dataframes_dict['teams'].isnull().sum())

    # Statistics of numerical values
    print(dataframes_dict['teams'].describe())

    # Information about columns of this table
    print(dataframes_dict['teams'].info())

    # Displays variation of classification of teams during the years
    print(dataframes_dict['teams'].columns)

    # Displays number of presences in playoffs of each team
    playoffs = dataframes_dict['teams'][dataframes_dict['teams']['playoff'] == 'Y']
    times_in_playoff = playoffs['tmID'].value_counts().reset_index()
    times_in_playoff.columns = ['Team', 'Playoff Appearances']
    print(times_in_playoff)


def teams_post_analysis(dataframes_dict):
    # Calculate WLRatio
    dataframes_dict['teams_post']['WLRatio'] = dataframes_dict['teams_post']['W'] / (
            dataframes_dict['teams_post']['W'] + dataframes_dict['teams_post']['L'])

    # Group by 'tmID' and 'year' and calculate the mean WLRatio for each team for each year
    team_year_wl_ratio = dataframes_dict['teams_post'].groupby(['tmID', 'year'])['WLRatio'].mean().reset_index()

    # Try to make a plot with the rank of each team per year (lineplot)

    # Rename columns for clarity
    team_year_wl_ratio.columns = ['Team', 'Year', 'WLRatio']

    team_year_wl_ratio = team_year_wl_ratio[team_year_wl_ratio['WLRatio'] != 0]

    # Display the table
    print(team_year_wl_ratio)


def series_post_analysis(dataframes_dict):
    playoff_finals = dataframes_dict['series_post'][dataframes_dict['series_post']['round'] == 'F']

    # Count the number of playoff wins for each team
    playoff_wins = playoff_finals['tmIDWinner'].value_counts().reset_index()
    playoff_wins.columns = ['TeamID', 'PlayoffWins']

    print(playoff_wins)


def draw_plots_model_scaler():
    df = pd.read_csv('resultsTeams_playoff.csv')
    df['Scaler'].fillna('None', inplace=True)

    models = df['Model'].unique()

    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c']

    plt.ylim(0, 1)

    for model in models:
        model_data = df[df['Model'] == model]

        plt.figure(figsize=(10, 8))
        for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
            metric_data = model_data[metric]
            plt.plot(model_data['Scaler'].astype(str), metric_data, label=metric, linestyle=line_styles[i],
                     color=colors[i])

        plt.xlabel('Scaler')
        plt.ylabel('Score')
        plt.title(f'{model} Model Performance')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        plt.savefig(f'plots/{model}_performance_plot.png')
        plt.show()


def main():
    draw_plots_model_scaler()
    dataframes_dict = load_data()
    coaches_analysis(dataframes_dict)
    awards_analysis(dataframes_dict)
    players_teams_analysis(dataframes_dict)
    players_analysis(dataframes_dict)
    teams_analysis(dataframes_dict)
    teams_post_analysis(dataframes_dict)
    series_post_analysis(dataframes_dict)


main()
