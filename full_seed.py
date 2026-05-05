import pandas as pd
import glob
import os


def getDF():
    folder_path = "data/tennis_atp/matches/"
    csv_files       = glob.glob(os.path.join(folder_path, "atp_matches_2*.csv"))
    csv_files_fut   = glob.glob(os.path.join(folder_path, "atp_matches_futures_2*.csv"))
    csv_files_chall = glob.glob(os.path.join(folder_path, "atp_matches_qual_chall_2*.csv"))

    df       = pd.concat((pd.read_csv(f) for f in csv_files),       ignore_index=True)
    df_fut   = pd.concat((pd.read_csv(f) for f in csv_files_fut),   ignore_index=True)
    df_chall = pd.concat((pd.read_csv(f) for f in csv_files_chall), ignore_index=True)

    return pd.concat([df, df_fut, df_chall], ignore_index=True)


if __name__ == "__main__":
    print("Loading data...")
    df = getDF()
    print(f"Loaded {len(df)} total matches")

    seeded = df.copy()
    seeded["winner_seed"] = pd.to_numeric(seeded["winner_seed"], errors="coerce")
    seeded["loser_seed"]  = pd.to_numeric(seeded["loser_seed"],  errors="coerce")
    seeded = seeded.dropna(subset=["winner_seed", "loser_seed"])

    print(f"{len(seeded)} matches had both players seeded\n")

    higher_seed_won = (seeded["winner_seed"] < seeded["loser_seed"]).sum()
    lower_seed_won  = (seeded["winner_seed"] > seeded["loser_seed"]).sum()

    total = len(seeded)
    print(f"Higher seed won: {higher_seed_won} ({100 * higher_seed_won / total:.1f}%)")
    print(f"Lower seed won: {lower_seed_won}  ({100 * lower_seed_won  / total:.1f}%)")