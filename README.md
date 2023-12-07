## Algoritmo em Players

- Calcular scores por ano dos jogadores (done)
- Concat player_teams com player onde bioId = playerId (done)
- Treinar ML em players_teams com o objetivo de prever of scores previamente acrescentados (passo 1)
- Calcular para o 10o ano os scores

## Algoritmo em Teams

- Concat players (players_teams+players) com Teams
- Criar tabela com series_post + Teams que indica o numero de jogos que uma equipa ganhou a outra. pode ate ser um racio = W/(W+L)
- Fazer algoritmo que analisa cada ano tendo em conta os players, as equipas e o coach
- Calcula score 0-1 que determina equipa vencedora
- Listar scores
- Top 8 vai para playoffs

