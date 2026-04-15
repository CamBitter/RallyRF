
import pandas as pd

list_of_dfs = []

for i in range(25): # generate years 2000 through 2024
    year = 2000 + i

    # get atp match data
    urlATP = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    try:
        yearly_df = pd.read_csv(urlATP)
        list_of_dfs.append(yearly_df)
    except Exception as e:
        print(f"Could not load data for year {year}: {e}")

    # get atp future match data
    urlFuture = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"
    try:
        yearly_df = pd.read_csv(urlFuture)
        list_of_dfs.append(yearly_df)
    except Exception as e:
        print(f"Could not load data for year {year}: {e}")

    # get atp qual chall match data
    urlQual = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{year}.csv"
    try:
        yearly_df = pd.read_csv(urlQual)
        list_of_dfs.append(yearly_df)
    except Exception as e:
        print(f"Could not load data for year {year}: {e}")

    

fullData = pd.concat(list_of_dfs, ignore_index=True)

# fullData.head()
print(fullData.size)