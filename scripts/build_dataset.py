import pandas as pd
import numpy as np
import glob
import os
import time

# --- Stat Key ---
# ace        : aces served
# df         : double faults
# svpt       : total serve points played
# 1stIn      : first serves in
# 1stWon     : points won on first serve
# 2ndWon     : points won on second serve
# SvGms      : service games played
# bpSaved    : break points saved (as server)
# bpFaced    : break points faced (as server)
# opp_bpf    : break points the opponent faced (= break point opportunities for this player as returner)
# opp_bps    : break points the opponent saved (as server)

# Derived features (all computed over rolling 6-month window):
# ace_vs_df        : aces / double faults
# first_in         : 1stIn / svpt  (first serve percentage)
# first_won        : 1stWon / 1stIn
# second_won       : 2ndWon / (svpt - 1stIn)
# bp_saved_pct     : bpSaved / bpFaced
# bp_converted_pct : (opp_bpf - opp_bps) / opp_bpf  (break points converted as returner)
# win_pct          : matches won / matches played
# surface_win_pct  : matches won / matches played on the current surface
# games_played     : total matches played in the window (activity/injury proxy)


def loadCSVs():
    """Combine all match CSVs into a single DataFrame, sorted by date, drop walkovers and retirements"""
    
    # Glob CSVs together
    folder_path = "data/tennis_atp/matches/"
    csv_files = glob.glob(os.path.join(folder_path, "atp_matches_2*.csv"))

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    score = df["score"].str.lower().str.strip()

    # Drop walkovers and retirements
    df = df[~score.str.contains("W/O|RET")].reset_index(drop=True)

    return df.reset_index(drop=True)


def one_year_ago(date_int):
    """Converts YYYYMMDD int to datetime, subtracts 12 months, and converts back to YYYYMMDD int"""
    d = pd.to_datetime(str(date_int), format="%Y%m%d")
    return int((d - pd.DateOffset(months=12)).strftime("%Y%m%d"))


def nan_diff(a, b):
    """Subtracts b from a, but returns NaN if either a or b is NaN (to avoid misleading values when one stat is missing)"""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return a - b


def build_player_index(df):
    """Precalculate player indexed list of matches to speed up rolling stat calculation"""

    records = df.to_dict("records")
    index = {}
    for match in records:
        index.setdefault(match["winner_name"], []).append(match)
        index.setdefault(match["loser_name"],  []).append(match)
    return index


def rolling_stats(player_index, player_name, surface, before_date):
    """Calculates rolling stats for a player up to a given date, optionally filtered by surface"""

    cutoff = one_year_ago(before_date)
    all_matches = player_index.get(player_name, [])
    window = [m for m in all_matches if cutoff <= m["tourney_date"] < before_date]

    if not window:
        return {s: np.nan for s in [
            "ace_vs_df", "first_in", "first_won", "second_won",
            "bp_saved_pct", "bp_converted_pct", "win_pct", "surface_win_pct", "games_played",
        ]}

    # Collect wins and losses seperately 
    w = [m for m in window if m["winner_name"] == player_name]
    l = [m for m in window if m["loser_name"]  == player_name]

    def stat(matches, col):
        arr = np.array([m[col] for m in matches], dtype=float)
        return arr[~np.isnan(arr)]

    def concat_stat(wcol, lcol):
        return np.concatenate([stat(w, wcol), stat(l, lcol)])

    # Concatenate stats from both wins and losses
    ace  = concat_stat("w_ace",     "l_ace")
    df_  = concat_stat("w_df",      "l_df")
    svpt = concat_stat("w_svpt",    "l_svpt")
    fin  = concat_stat("w_1stIn",   "l_1stIn")
    fwon = concat_stat("w_1stWon",  "l_1stWon")
    swon = concat_stat("w_2ndWon",  "l_2ndWon")
    bps  = concat_stat("w_bpSaved", "l_bpSaved")
    bpf  = concat_stat("w_bpFaced", "l_bpFaced")
    opp_bpf = np.concatenate([stat(w, "l_bpFaced"), stat(l, "w_bpFaced")])
    opp_bps = np.concatenate([stat(w, "l_bpSaved"), stat(l, "w_bpSaved")])

    # Calculate derived stats, handling division by zero and NaNs
    s_attempts = svpt.sum() - fin.sum()

    surface_matches = [m for m in window if m.get("surface") == surface] if surface else []
    n_surface = len(surface_matches)
    surface_win_pct = sum(1 for m in surface_matches if m["winner_name"] == player_name) / n_surface if n_surface > 0 else np.nan

    return {
        "ace_vs_df":         (ace.sum()  / df_.sum())   if df_.sum()  > 0 else np.nan,
        "first_in":          (fin.sum()  / svpt.sum())  if svpt.sum() > 0 else np.nan,
        "first_won":         (fwon.sum() / fin.sum())   if fin.sum()  > 0 else np.nan,
        "second_won":        (swon.sum() / s_attempts)  if s_attempts > 0 else np.nan,
        "bp_saved_pct":      (bps.sum()  / bpf.sum())   if bpf.sum()  > 0 else np.nan,
        "bp_converted_pct":  ((opp_bpf.sum() - opp_bps.sum()) / opp_bpf.sum()) if opp_bpf.sum() > 0 else np.nan,
        "win_pct":           len(w) / len(window),
        "surface_win_pct":   surface_win_pct,
        "games_played":      len(window),
    }


def build(df):
    rows = []
    total = len(df)
    player_index = build_player_index(df)
    print("Player matches indexed.")

    for i, match in df.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{total} rows processed...")

        # Get date, player, surface info
        date = match["tourney_date"]
        p1   = match["winner_name"]
        p2   = match["loser_name"]
        surf = match.get("surface", None)

        # Calculate 6-month rolling stats for both players up to the match date
        s1 = rolling_stats(player_index, p1, surf, date)
        s2 = rolling_stats(player_index, p2, surf, date)

        p1_rank     = match["winner_rank"]
        p2_rank     = match["loser_rank"]
        p1_rpts     = match["winner_rank_points"]
        p2_rpts     = match["loser_rank_points"]
        p1_age      = match["winner_age"]
        p2_age      = match["loser_age"]
        p1_height   = match["winner_ht"]
        p2_height   = match["loser_ht"]

        row = {
            "tourney_id":              match["tourney_id"],
            "tourney_date":            date,
            "round":                   match["round"],
            "p1_name":                 p1,
            "p2_name":                 p2,
            "p1_won":                  1,
            "surface_hard":            1 if surf == "Hard"  else 0,
            "surface_clay":            1 if surf == "Clay"  else 0,
            "surface_grass":           1 if surf == "Grass" else 0,
            "p1_rank":                 p1_rank,
            "p2_rank":                 p2_rank,
            "p1_rank_points":          p1_rpts,
            "p2_rank_points":          p2_rpts,
            "p1_height":               p1_height,
            "p2_height":               p2_height,
            "age_diff":                nan_diff(p1_age,  p2_age),
            "surface_win_pct_diff":    nan_diff(s1["surface_win_pct"],   s2["surface_win_pct"]),
            "ace_vs_df_diff":          nan_diff(s1["ace_vs_df"],         s2["ace_vs_df"]),
            "first_in_diff":           nan_diff(s1["first_in"],          s2["first_in"]),
            "first_won_diff":          nan_diff(s1["first_won"],         s2["first_won"]),
            "second_won_diff":         nan_diff(s1["second_won"],        s2["second_won"]),
            "bp_saved_pct_diff":       nan_diff(s1["bp_saved_pct"],      s2["bp_saved_pct"]),
            "bp_converted_pct_diff":   nan_diff(s1["bp_converted_pct"],  s2["bp_converted_pct"]),
            "win_pct_diff":            nan_diff(s1["win_pct"],           s2["win_pct"]),
            "games_played_diff":       nan_diff(s1["games_played"],      s2["games_played"]),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    # Flip all match rows for data augmentation
    diff_cols = [c for c in result.columns if c.endswith("_diff")]
    p1_cols   = [c for c in result.columns if c.startswith("p1_") and c not in ("p1_name", "p1_won")]
    p2_cols   = [c for c in result.columns if c.startswith("p2_") and c != "p2_name"]
    feature_cols = p1_cols + p2_cols + diff_cols

    flipped = result.copy()
    flipped["p1_won"] = 0
    flipped[diff_cols] *= -1
    flipped[p1_cols + p2_cols] = result[p2_cols + p1_cols].values
    flipped[["p1_name", "p2_name"]] = flipped[["p2_name", "p1_name"]].values

    combined = pd.concat([result, flipped], ignore_index=True)
    return combined.dropna(subset=feature_cols)


if __name__ == "__main__":
    df = loadCSVs()
    print(f"Loaded {len(df)} matches. Building features...")

    start = time.time()
    features = build(df)
    elapsed = time.time() - start

    out_path = "data/cleaned/some_diffs.csv"
    features.to_csv(out_path, index=False)
    print(f"\nDone! Saved {len(features)} rows to '{out_path}' in {elapsed:.3f}s.")
