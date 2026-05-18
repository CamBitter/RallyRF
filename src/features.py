"""
Feature reference
-----------------
  surface_hard             1 if match played on hard court
  surface_clay             1 if match played on clay
  surface_grass            1 if match played on grass

Raw player attributes (directly from Sackmann):
  p1/p2_rank               ATP ranking (lower = better)
  p1/p2_rank_points        ATP ranking points
  p1/p2_age                Age at match date
  p1/p2_height             Height in cm

Historical stats per player (rolled up from prior matches):
  p1/p2_surface_win_pct    Win % on this surface
  p1/p2_ace_vs_df          Ace-to-double-fault ratio [derived]
  p1/p2_first_in           First-serve in %
  p1/p2_first_won          First-serve win %
  p1/p2_second_won         Second-serve win %
  p1/p2_bp_saved_pct       Break points saved %
  p1/p2_bp_converted_pct   Break points converted %
  p1/p2_win_pct            Overall win %
  p1/p2_games_played       Total games played

Diff features (p1 - p2):
  rank_diff                Rank difference; from Sackmann
  rank_pts_diff            Rank points difference; from Sackmann
  age_diff                 Age difference in years; from Sackmann
  height_diff              Height difference in cm; from Sackmann
  surface_win_pct_diff     Difference in win % on this surface
  ace_vs_df_diff           Difference in ace-to-double-fault ratio [derived]
  first_in_diff            Difference in first-serve in %
  first_won_diff           Difference in first-serve win %
  second_won_diff          Difference in second-serve win %
  bp_saved_pct_diff        Difference in break points saved %
  bp_converted_pct_diff    Difference in break points converted %
  win_pct_diff             Difference in overall win %
  games_played_diff        Difference in total games played
"""

FEATURE_SETS = {
    # all stats as diffs
    "all_diffs.csv": [
        "surface_hard",
        "surface_clay",
        "surface_grass",
        "rank_diff",
        "rank_pts_diff",
        "age_diff",
        "height_diff",
        "surface_win_pct_diff",
        "ace_vs_df_diff",
        "first_in_diff",
        "first_won_diff",
        "second_won_diff",
        "bp_saved_pct_diff",
        "bp_converted_pct_diff",
        "win_pct_diff",
        "games_played_diff",
    ],

    # rank/height/rank_pts raw per player
    "some_diffs.csv": [
        "surface_hard",
        "surface_clay",
        "surface_grass",
        "p1_rank",
        "p2_rank",
        "p1_rank_points",
        "p2_rank_points",
        "p1_height",
        "p2_height",
        "age_diff",
        "surface_win_pct_diff",
        "ace_vs_df_diff",
        "first_in_diff",
        "first_won_diff",
        "second_won_diff",
        "bp_saved_pct_diff",
        "bp_converted_pct_diff",
        "win_pct_diff",
        "games_played_diff",
    ],

    # all stats as raw per-player values, no diffs
    "no_diffs.csv": [
        "surface_hard",
        "surface_clay",
        "surface_grass",
        "p1_rank",
        "p2_rank",
        "p1_rank_points",
        "p2_rank_points",
        "p1_age",
        "p2_age",
        "p1_height",
        "p2_height",
        "p1_surface_win_pct",
        "p2_surface_win_pct",
        "p1_ace_vs_df",
        "p2_ace_vs_df",
        "p1_first_in",
        "p2_first_in",
        "p1_first_won",
        "p2_first_won",
        "p1_second_won",
        "p2_second_won",
        "p1_bp_saved_pct",
        "p2_bp_saved_pct",
        "p1_bp_converted_pct",
        "p2_bp_converted_pct",
        "p1_win_pct",
        "p2_win_pct",
        "p1_games_played",
        "p2_games_played",
    ],
}

def get_features(csv_path):
    filename = csv_path.split("/")[-1]
    if filename not in FEATURE_SETS:
        raise KeyError(f"No feature set defined for '{filename}'. Add it to features.py.")
    return FEATURE_SETS[filename]
