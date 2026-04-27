import torch
import pandas as pd
import glob
import os

def getDF():
    # Point to folder
    folder_path = "data/tennis_atp/matches/"

    # all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "atp_matches_2*.csv"))

    csv_files_futures = glob.glob(os.path.join(folder_path, "atp_matches_futures_2*.csv"))

    csv_files_chall = glob.glob(os.path.join(folder_path, "atp_matches_qual_chall_2*.csv"))


    # Read and concatenate them all into one DataFrame
    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    df_futures = pd.concat((pd.read_csv(file) for file in csv_files_futures), ignore_index=True)

    df_chall = pd.concat((pd.read_csv(file) for file in csv_files_chall), ignore_index=True)

    df["match_type"] = "main"
    df_futures["match_type"] = "futures"
    df_chall["match_type"] = "challenger"
    df = pd.concat([df, df_futures, df_chall], ignore_index=True)
    df = pd.get_dummies(df, columns=["match_type"], prefix="match_type")
    return df

def getPlayerStatLastYrAvg(df, player_name, match_date, stat):

    player_df = df[(df["winner_name"] == player_name) | (df["loser_name"] == player_name)]
    player_df = player_df[(player_df["tourney_date"] < match_date) &
                           (player_df["tourney_date"] > match_date - 10000)]
    winner_col = None
    loser_col = None

    if "winner_" + stat in df.columns:
        winner_col = "winner_" + stat
    elif "w_" + stat in df.columns:
        winner_col = "w_" + stat

    if "loser_" + stat in df.columns:
        loser_col = "loser_" + stat
    elif "l_" + stat in df.columns:
        loser_col = "l_" + stat

    if winner_col is None or loser_col is None:
        return None
    
    winner_stats = player_df[player_df["winner_name"] == player_name][winner_col]
    loser_stats = player_df[player_df["loser_name"] == player_name][loser_col]

    stats = pd.concat([winner_stats, loser_stats]).dropna()
    return stats.mean()


