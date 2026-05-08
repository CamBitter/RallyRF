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
