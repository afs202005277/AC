**Domain Description: Predictive Analysis for Basketball Playoff Qualification**

*Introduction:*

This project delves into the domain of basketball tournaments, specifically focusing on the prediction of playoff qualification for teams. Basketball tournaments are traditionally divided into two phases: the regular season, where teams compete to accumulate the highest number of wins, and the playoffs, featuring knockout matches for the championship. The objective of this project is to utilize a decade's worth of comprehensive data, encompassing players, teams, coaches, games, and various performance metrics, to forecast which teams will qualify for the playoffs in the forthcoming season.

*Dataset Overview:*

The dataset at hand spans ten years, providing a rich and extensive repository of information relevant to basketball tournaments. It includes detailed player statistics and respective awards, team performance metrics, coaching data, and game-specific information. Furthermore, the dataset also provides data of the team's performance both in the season and in the post-season.
Moreover, the dataset is not provided as a typical one table dataset. Instead, we were provided with data that was not previously cleaned and it was scattered through seven distinct tables.

*Dataset Preparation Steps:*

In every Machine Learning project, the dataset preparation is a crucial step in order to allow the constructed model to make accurate and meaningful predictions. Bearing this in mind, we did the following transformations to the data that was provided:

- Players:
  - Removal of the death date of players since it is not relevant to how well the team performs (**é preciso ver se há equipas a jogar com players q estão mortos na altura dos jogos**)
  - Removal of the players that do not have a position assigned to them or whose birth date is all zeros, since these are clearly errors in the data collection
  - Conversion of college's names and the positions of the players into numerical indexes, allowing the calculation of the correlation matrix in order to find out the existence of redundant attributes
  - Removal of the "firstseason" and "lastseason" columns, since all the instances have '0' in both the columns therefore these attributes do not add additional information to the players
  - Replacement of the birth date with the birth year of the player to make the comparisons easier without losing relevant information.
  - Creation of a new column 'EFF' that is a measure of a player performace which is very useful for our final goal.
  - Merge dataframe players with players_teams (where playerID = bioID) 

- Teams:
  - Removal of the divID attribute that is NAN in all rows of the table, therefore useless
  - 
*Project Objective:*

The primary goal of this project is to leverage the available data to develop predictive models that can accurately identify which teams will secure playoff berths in the upcoming season. This task holds significant importance for basketball enthusiasts, team management, sponsors, and the broader sports industry.

*Key Factors Influencing Playoff Qualification:*

Several key factors may influence a team's qualification for the playoffs. These factors could include individual player statistics, such as points scored, rebounds, and assists, as well as team-level performance indicators like win-loss records and scoring differentials. **completar com os cálculos**
