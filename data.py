import torch
import pandas as pd
import glob
import os

def getDF():
    # Point to your folder
    folder_path = "data/tennis_atp/matches/"

    # Grab all CSV files in the folder
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
    df = pd.concat([df, df_chall], ignore_index=True)
    df = pd.get_dummies(df, columns=["match_type"], prefix="match_type")
    return df


## Makes a df where each row is a single player, marked by win/loss
def playerDf(df):

    winDf = df[["tourney_id","tourney_name","surface","draw_size","tourney_level","tourney_date",
    "match_num","winner_id","winner_seed","winner_entry","winner_name","winner_hand","winner_ht",
    "winner_ioc","winner_age","score","best_of","round","minutes","w_ace","w_df","w_svpt","w_1stIn"
    ,"w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced"]]
    loseDf = df[["tourney_id","tourney_name","surface","draw_size","tourney_level","tourney_date",
    "match_num","loser_id","loser_seed","loser_entry","loser_name","loser_hand","loser_ht",
    "loser_ioc","loser_age","score","best_of","round","minutes","l_ace","l_df","l_svpt","l_1stIn"
    ,"l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced"]]

    winDf.columns = ["tourney_id","tourney_name","surface","draw_size","tourney_level","tourney_date",
    "match_num","player_id","player_seed","player_entry","player_name","player_hand","player_ht",
    "player_ioc","player_age","score","best_of","round","minutes","ace","df","svpt","1stIn"
    ,"1stWon","2ndWon","SvGms","bpSaved","bpFaced"]

    loseDf.columns = ["tourney_id","tourney_name","surface","draw_size","tourney_level","tourney_date",
    "match_num","player_id","player_seed","player_entry","player_name","player_hand","player_ht",
    "player_ioc","player_age","score","best_of","round","minutes","ace","df","svpt","1stIn"
    ,"1stWon","2ndWon","SvGms","bpSaved","bpFaced"]

    winDf["won"] = 1
    loseDf["won"] = 0

    return pd.concat([winDf, loseDf], ignore_index=True)

#gets the average of a numerical stat for a player in the last year before a match date. Used for feature engineering.

def getPlayerStatLastYrAvg(df, player_name, match_date, stat):

    match_date = pd.to_datetime(match_date, format="%Y%m%d")
    frame = df[df["player_name"] == player_name]
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"], format="%Y%m%d")
    frame = frame[frame["tourney_date"] < match_date]
    frame = frame[frame["tourney_date"] >= (match_date - pd.DateOffset(years=1))]

    return frame[stat].mean()

#functions just for exploration
def numNAByRow(df):
    df["num_na"] = df.isna().sum(axis=1)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

    return df[["tourney_date", "num_na"]]

def numNAByCol(df):
    return df.isna().sum()