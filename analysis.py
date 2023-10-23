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
    #Check for null values
    print(dataframes_dict['coaches'].isnull().sum())
    
    #Statistics of numerical values
    print(dataframes_dict['coaches'].describe())
    
    #Information about columns of this table
    print(dataframes_dict['coaches'].info())
    
    #Displays the number of different coaches a team had during the 10 years
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
    
    #Displays the top 20 coaches with the highest Win-Lost Ratio
    dataframes_dict['coaches']['WLRatio'] = dataframes_dict['coaches']['won'] / (dataframes_dict['coaches']['won'] + dataframes_dict['coaches']['lost'])

    best_coaches = dataframes_dict['coaches'].sort_values(by='WLRatio', ascending=False).head(35)
    
    plt.figure(figsize=(10, 6))
    plt.barh(best_coaches['coachID'], best_coaches['WLRatio'], color='skyblue')
    plt.xlabel('Win-Loss Ratio')
    plt.ylabel('Coach ID')
    plt.title('Top 20 Coaches by Win-Loss Ratio')
    plt.gca().invert_yaxis()  
    plt.show()
    
def awards_analysis(dataframes_dict):
    #Check for null values
    print(dataframes_dict['awards_players'].isnull().sum())
    
    #No need to describe() - just one numerical value
    
    #Information about columns of this table
    print(dataframes_dict['awards_players'].info())
    
    #Number of awards of each type
    plt.figure(figsize=(12, 6))
    sns.countplot(x='award', data=dataframes_dict['awards_players'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    #Displays the top 10 players with the most awards won
    award_counts = dataframes_dict['awards_players'].groupby('playerID')['award'].count().reset_index()
    
    award_counts = award_counts.sort_values(by='award', ascending = False).head(10)
        
    plt.figure(figsize=(10, 6))
    plt.barh(award_counts['playerID'], award_counts['award'], color='skyblue')
    plt.xlabel('Player ID')
    plt.ylabel('Number of awards')
    plt.title('Top 10 Players with more awards')  
    plt.gca().invert_yaxis()  
    plt.show()
    
def players_teams_analysis(dataframes_dict):
    #Check for null values
    print(dataframes_dict['players_teams'].isnull().sum())
    
    #Statistics of numerical values
    print(dataframes_dict['players_teams'].describe())
    
    #Information about columns of this table
    print(dataframes_dict['players_teams'].info())
    
    #Correlation matrix of players_teams
    numerical_data = dataframes_dict['players_teams'].select_dtypes(include='int64')
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(15,12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()
    
def players_analysis(dataframes_dict):
    #Check for null values
    print(dataframes_dict['players'].isnull().sum())
    
    #Statistics of numerical values
    print(dataframes_dict['players'].describe())
    
    #Information about columns of this table
    print(dataframes_dict['players'].info())    
    
    #See that deathDate is almost the same to every player, so it doesn't add anything
    print((dataframes_dict['players']['deathDate'] != '0000-00-00').sum())
    
    #Correlation matrix of players
    numerical_data = dataframes_dict['players'].select_dtypes(include=['int64','float64'])
    correlation_matrix_players = numerical_data.corr()
    plt.figure(figsize=(15,12))
    sns.heatmap(correlation_matrix_players, annot=True, cmap='coolwarm')
    plt.show()
    
def teams_analysis(dataframes_dict):
    #Check for null values
    print(dataframes_dict['teams'].isnull().sum())
    
    #Statistics of numerical values
    print(dataframes_dict['teams'].describe())
    
    #Information about columns of this table
    print(dataframes_dict['teams'].info()) 
    
    #Displays variation of classification of teams during the years
    print(dataframes_dict['teams'].columns)
    
    #sns.set(style="darkgrid")
    #plt.figure(figsize=(12, 6))
    #sns.lineplot(data=dataframes_dict['teams'], x='year', y='rank', hue='tmID', marker="o", markersize=8, dashes=False)
    #plt.xlabel('Year')
    #plt.ylabel('Rank')
    #plt.title('Rank per Year')
    #plt.legend(title='Teams', loc='upper left', bbox_to_anchor=(1, 1))
    #plt.show()
        
def main():
    dataframes_dict = load_data()
    coaches_analysis(dataframes_dict) 
    awards_analysis(dataframes_dict)
    players_teams_analysis(dataframes_dict)
    players_analysis(dataframes_dict)
    teams_analysis(dataframes_dict)
    
main()    
    
    
    
    
    
    
    


    