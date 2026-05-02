import pandas as pd
import numpy as np
import glob
import os

# Load data (copied from data.py with minor tweaks)

def getDF():
    folder_path = "data/tennis_atp/matches/"
    csv_files       = glob.glob(os.path.join(folder_path, "atp_matches_2*.csv"))
    csv_files_fut   = glob.glob(os.path.join(folder_path, "atp_matches_futures_2*.csv"))
    csv_files_chall = glob.glob(os.path.join(folder_path, "atp_matches_qual_chall_2*.csv"))
 
    df       = pd.concat((pd.read_csv(f) for f in csv_files),       ignore_index=True)
    df_fut   = pd.concat((pd.read_csv(f) for f in csv_files_fut),   ignore_index=True)
    df_chall = pd.concat((pd.read_csv(f) for f in csv_files_chall), ignore_index=True)
 
    df["match_type"]       = "main"
    df_fut["match_type"]   = "futures"
    df_chall["match_type"] = "challenger"
 
    df = pd.concat([df, df_fut, df_chall], ignore_index=True)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df

# helper funcs

def prev_year(date):
    return date - 10000


def rolling_serve_stats(df, player_name, before_date):
    mask = (
        ((df["winner_name"] == player_name) | (df["loser_name"] == player_name)) &
        (df["tourney_date"] < before_date) &
        (df["tourney_date"] >= prev_year(before_date))
    )
    player_df = df[mask]

    if player_df.empty:
        return {s: np.nan for s in ["ace_vs_df", "first_in", "first_won", "second_won", "bp_saved_pct"]}

    w = player_df[player_df["winner_name"] == player_name]
    l = player_df[player_df["loser_name"]  == player_name]

    def concat_stat(wcol, lcol):
        return pd.concat([w[wcol].dropna(), l[lcol].dropna()])

    ace  = concat_stat("w_ace",    "l_ace")
    df_  = concat_stat("w_df",     "l_df")
    svpt = concat_stat("w_svpt",   "l_svpt")
    fin  = concat_stat("w_1stIn",  "l_1stIn")
    fwon = concat_stat("w_1stWon", "l_1stWon")
    swon = concat_stat("w_2ndWon", "l_2ndWon")
    bps  = concat_stat("w_bpSaved","l_bpSaved")
    bpf  = concat_stat("w_bpFaced","l_bpFaced")

    ace_vs_df   = (ace.sum()  / df_.sum())  if df_.sum()  > 0 else np.nan
    first_in    = (fin.sum()  / svpt.sum()) if svpt.sum() > 0 else np.nan
    first_won   = (fwon.sum() / fin.sum())  if fin.sum()  > 0 else np.nan

    s_attempts  = svpt.sum() - fin.sum()
    second_won  = (swon.sum() / s_attempts) if s_attempts > 0 else np.nan
    bp_saved    = (bps.sum()  / bpf.sum())  if bpf.sum()  > 0 else np.nan

    return {
        "ace_vs_df":    ace_vs_df,
        "first_in":     first_in,
        "first_won":    first_won,
        "second_won":   second_won,
        "bp_saved_pct": bp_saved,
    }


def surface_win_pct(df, player_name, surface, before_date):
    mask = (
        ((df["winner_name"] == player_name) | (df["loser_name"] == player_name)) &
        (df["surface"] == surface) &
        (df["tourney_date"] < before_date) &
        (df["tourney_date"] >= prev_year(before_date))
    )
    player_df = df[mask]
    if player_df.empty:
        return np.nan
    wins = (player_df["winner_name"] == player_name).sum()
    return wins / len(player_df)


# main build loop

def build(df):
    rows = []
    total = len(df)

    for i, match in df.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{total} rows processed...")

        date = match["tourney_date"]
        p1   = match["winner_name"]
        p2   = match["loser_name"]
        surf = match.get("surface", None)

        s1 = rolling_serve_stats(df, p1, date)
        s2 = rolling_serve_stats(df, p2, date)

        sp1 = surface_win_pct(df, p1, surf, date) if surf else np.nan
        sp2 = surface_win_pct(df, p2, surf, date) if surf else np.nan

        row = {
            # Match metadata
            "tourney_id":   match.get("tourney_id"),
            "tourney_date": date,
            "surface":      surf,
            "round":        match.get("round"),
            "match_type":   match.get("match_type"),

            # Label: 1 = p1 (winner) won, flipped 50% of the time below
            "label": 1,

            # Simple diffs (winner - loser)
            "rank_diff":     (match.get("winner_rank")        or 0) - (match.get("loser_rank")        or 0),
            "rank_pts_diff": (match.get("winner_rank_points") or 0) - (match.get("loser_rank_points") or 0),
            "age_diff":      (match.get("winner_age")         or 0) - (match.get("loser_age")         or 0),
            "ht_diff":       (match.get("winner_ht")          or 0) - (match.get("loser_ht")          or 0),

            # Surface win pct diff
            "surface_per_diff": (sp1 or 0) - (sp2 or 0),

            # Serve stat diffs (rolling 12 months)
            "ace_vs_df_diff":    (s1["ace_vs_df"]    or 0) - (s2["ace_vs_df"]    or 0),
            "first_in_diff":     (s1["first_in"]     or 0) - (s2["first_in"]     or 0),
            "first_won_diff":    (s1["first_won"]    or 0) - (s2["first_won"]    or 0),
            "second_won_diff":   (s1["second_won"]   or 0) - (s2["second_won"]   or 0),
            "bp_saved_pct_diff": (s1["bp_saved_pct"] or 0) - (s2["bp_saved_pct"] or 0),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    # Randomly flip ~50% so the model doesn't learn "row order = winner"
    diff_cols = [c for c in result.columns if c.endswith("_diff")]
    flip = np.random.rand(len(result)) > 0.5
    result.loc[flip, diff_cols] *= -1
    result.loc[flip, "label"] = 0

    return result



if __name__ == "__main__":
    df = getDF()
    # df = df.head(500)
    print(f"Loaded {len(df)} matches. Building features...")

    features = build(df)

    out_path = "match_features.csv"
    features.to_csv(out_path, index=False)
    print(f"\nDone! Saved {len(features)} rows to '{out_path}'.")