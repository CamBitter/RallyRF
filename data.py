import pandas as pd
import glob
import os

# Point to your folder
folder_path = "data/tennis_atp/"

# Grab all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "atp_matches_2*.csv"))

# Read and concatenate them all into one DataFrame
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

print(df.shape)
print(df.head())
