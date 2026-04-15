import requests
from datetime import datetime
import time
import pandas as pd
import numpy as np
import boto3
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_team_info(team_id, url, retries=5, backoff=4):
    team_url = f"{url}v1/roster/{team_id}/current"

    for attempt in range(retries):
        response = requests.get(team_url)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = int(response.headers.get("Retry-After", backoff))
            print(f"Rate limited. Waiting {wait} seconds for team {team_id}...")
            time.sleep(wait)
        else:
            print(f"Bad connection: {response.status_code} @ {team_id}")
            break

    return None

def get_player_info(skater_id, url, retries=5, backoff=4):
    player_url = f"{url}v1/player/{skater_id}/landing"

    
    for attempt in range(retries):
        response = requests.get(player_url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = int(response.headers.get("Retry-After", backoff))
            print(f"Rate limited. Waiting {wait} seconds for player {skater_id}...")
            time.sleep(wait)
        else:
            print(f"Bad connection: {response.status_code} @ {skater_id}")
            break

    return None

def get_team_standing_info(url, retries=5, backoff=2):
    for attempt in range(retries):
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = int(response.headers.get("Retry-After", backoff))
            print(f"Rate limited. Waiting {wait} seconds for standings...")
            time.sleep(wait)
        else:
            print(f"Bad connection: {response.status_code}")
            break

    return None

def mmss_to_seconds(time: str):
    time_split = time.split(":")
    return int(time_split[0]) * 60 + int(time_split[1])

base_url = "https://api-web.nhle.com/"
team_standings = requests.get("https://api-web.nhle.com/v1/standings/now")
if(team_standings.status_code != 200):
    print('url failed')
else:
    teams = [names['teamAbbrev']['default'] for names in team_standings.json()["standings"]]

rosters = []
for team in teams:
    try:
        rosters.append(get_team_info(team, base_url))
    except:
        print(f'team not found!')

players = {}
for roster in rosters:
    skaters = roster.get('forwards', []) + roster.get('defensemen', [])
    try:
        for player in skaters:
            players.update({f"{player['firstName']['default']} {player['lastName']['default']}": player['id']})
    except:
        print(f'roster not found!')

x = 0
career_stat = []
player_info = []
current_stat = []
current_year = f"{int(datetime.now().year)- 1}-{datetime.now().year}"

for player in players.items():
    player_data = get_player_info(str(player[1]), base_url)
    season_totals = player_data.get('seasonTotals', [])
    first_name = player_data.get('firstName', {}).get('default', None)
    last_name = player_data.get('lastName', {}).get('default', None)
    badges = player_data.get('badges', None)
    team_logo = player_data.get('teamLogo', None)
    sweater_number = player_data.get('sweaterNumber', None)
    position = player_data.get('position', None)
    headshot = player_data.get('headshot', None)
    hero_image = player_data.get('heroImage', None)
    height_in = player_data.get('heightInInches', None)
    height_cm = player_data.get('heightInCentimeters', None)
    weight_lb = player_data.get('weightInPounds', None)
    weight_kg = player_data.get('weightInKilograms', None)
    birth_date = player_data.get('birthDate', None)
    birth_city = player_data.get('birthCity', {}).get('default', None)
    birth_state = player_data.get('birthStateProvince', {}).get('default', None)
    birth_country = player_data.get('birthCountry', None)
    shoots_catches = player_data.get('shootsCatches', None)
    player_personal_data = {
        "playerId": player[1],
        "first_name": first_name,
        "last_name": last_name,
        "badges": badges,
        "team_logo": team_logo,
        "sweater_number": sweater_number,
        "position": position,
        "headshot": headshot,
        "hero_image": hero_image,
        "height_in": height_in,
        "height_cm": height_cm,
        "weight_lb": weight_lb,
        "weight_kg": weight_kg,
        "birth_date": birth_date,
        "birth_city": birth_city,
        "birth_state": birth_state,
        "birth_country": birth_country,
        "shoots_catches": shoots_catches
    }
    player_info.append(player_personal_data)

    for season in season_totals:
        
        year = str(season.get('season'))[:4] + "-" + str(season.get('season'))[4:]
        team = season.get('teamName', {}).get('default', 'Unknown')
        league = season.get('leagueAbbrev', 'Unknown')
        gp = season.get('gamesPlayed', 0)
        goals = season.get('goals', 0)
        assists = season.get('assists', 0)
        points = season.get('points', 0)
        pim = season.get('pim', 0)
        plus_minus = season.get('plusMinus', None)
        avg_toi = season.get('avgToi', "00:00")
        faceoff_pct = season.get('faceoffWinningPctg', None)
        gwg = season.get('gameWinningGoals', 0)
        otg = season.get('otGoals', 0)
        pp_goals = season.get('powerPlayGoals', 0)
        pp_points = season.get('powerPlayPoints', 0)
        sh_goals = season.get('shorthandedGoals', 0)
        sh_points = season.get('shorthandedPoints', 0)
        shots = season.get('shots', 0)
        stat_line = {
            "playerId": player[1],
            "position": position,
            "season": year,
            "team": team,
            "league": league,
            "gamesPlayed": gp,
            "goals": goals,
            "assists": assists,
            "points": points,
            "pim": pim,
            "plusMinus": plus_minus,
            "avgToi": avg_toi,
            "faceoffPct": faceoff_pct,
            "gameWinningGoals": gwg,
            "otGoals": otg,
            "powerPlayGoals": pp_goals,
            "powerPlayPoints": pp_points,
            "shGoals": sh_goals,
            "shPoints": sh_points,
            "shots": shots
        }
        career_stat.append(stat_line)

        if year == current_year and league == 'NHL':
            current_stat.append(stat_line)
    if x % 10 == 0:
        print(f"{x} out of {len(players.items())}")
    x += 1




career_df = pd.DataFrame(career_stat)
career_df.to_csv("files/skater_career_stats.csv", index=False)

current_df = pd.DataFrame(current_stat)
current_df['avgToi'] = current_df['avgToi'].apply(mmss_to_seconds)
current_df.to_csv("files/skater_current_stats.csv", index=False)

personal_df = pd.DataFrame(player_info)

merge_df = pd.merge(personal_df, current_df, on='playerId', how='left')

group_cols = [
    "playerId", "first_name", "last_name", "season"
]

agg_dict = {
    "badges": "first",
    "team_logo": "first",
    "sweater_number": "first",
    "position_x": "first",
    "headshot": "first",
    "hero_image": "first",
    "height_in": "first",
    "height_cm": "first",
    "weight_lb": "first",
    "weight_kg": "first",
    "birth_date": "first",
    "birth_city": "first",
    "birth_state": "first",
    "birth_country": "first",
    "shoots_catches": "first",
    "position_y": "first",
    "league": "first",
    "team": lambda x: " / ".join(x.unique()),
    "gamesPlayed": "sum",
    "goals": "sum",
    "assists": "sum",
    "points": "sum",
    "pim": "sum",
    "plusMinus": "sum",
    "avgToi": "sum",
    "gameWinningGoals": "sum",
    "otGoals": "sum",
    "powerPlayGoals": "sum",
    "powerPlayPoints": "sum",
    "shGoals": "sum",
    "shPoints": "sum",
    "shots": "sum",
}

merge_df = merge_df.groupby(group_cols, as_index=False).agg(agg_dict)

total_toi = merge_df["avgToi"] * merge_df["gamesPlayed"]

merge_df["goals_per_60"] = merge_df["goals"] / total_toi * 3600
merge_df["assists_per_60"] = merge_df["assists"] / total_toi * 3600
merge_df["shots_per_60"] = merge_df["shots"] / total_toi * 3600
merge_df["points_per_60"] = merge_df["points"] / total_toi * 3600
merge_df = merge_df[merge_df["position_x"].isin(["C", "L", "R"])]
merge_df = merge_df.reset_index(drop=True)

merge_df.to_csv("files/forwards_data.csv", index=False)
personal_df.to_csv("files/personal_data.csv", index=False)



current_year = f"{int(datetime.now().year)- 1}-{datetime.now().year}"
standing_url = f"https://api.nhle.com/stats/rest/en/team/summary?sort=shotsForPerGame&cayenneExp=seasonId={current_year[:4]+current_year[5:]}%20and%20gameTypeId=2"



standings_data = get_team_standing_info(standing_url)['data']

team_info = []

for team in standings_data:
    team_data = {
        "teamId": team.get("teamId"),
        "team_name": team.get("teamFullName"),
        "season": team.get("seasonId"),
        "games_played": team.get("gamesPlayed"),
        "wins": team.get("wins"),
        "losses": team.get("losses"),
        "ot_losses": team.get("otLosses"),
        "points": team.get("points"),
        "point_pct": team.get("pointPct"),
        "goals_for": team.get("goalsFor"),
        "goals_against": team.get("goalsAgainst"),
        "goals_for_per_game": team.get("goalsForPerGame"),
        "goals_against_per_game": team.get("goalsAgainstPerGame"),
        "shots_for_per_game": team.get("shotsForPerGame"),
        "shots_against_per_game": team.get("shotsAgainstPerGame"),
        "faceoff_win_pct": team.get("faceoffWinPct"),
        "power_play_pct": team.get("powerPlayPct"),
        "penalty_kill_pct": team.get("penaltyKillPct"),
        "regulation_wins": team.get("winsInRegulation"),
        "shootout_wins": team.get("winsInShootout"),
        "row": team.get("regulationAndOtWins"),
        "shutouts": team.get("teamShutouts"),
    }

    team_info.append(team_data)
teams_df = pd.DataFrame(team_info)
teams_df.to_csv('files/team_standings.csv', index=False)
