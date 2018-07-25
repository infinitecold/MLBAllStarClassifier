import pandas as pd

people_df = pd.read_csv('../../data/raw/People.csv', usecols=['playerID', 'bbrefID'])
batting_df = pd.read_csv('../../data/bbref/Batting_raw.csv')
fielding_df = pd.read_csv('../../data/bbref/Fielding_raw.csv')
appearances_df = pd.read_csv('../../data/bbref/Appearances_raw.csv')

# convert each playerID in context of bbref to playerID in context of our data
def bbrefID_to_playerID(df):
    df.rename({'playerID': 'bbrefID'}, axis=1, inplace=True)
    df = pd.merge(df, people_df, how='left', on='bbrefID')
    df['playerID'] = df['playerID'].fillna(df['bbrefID'])
    df.drop('bbrefID', axis=1, inplace=True)
    return df

batting_df = bbrefID_to_playerID(batting_df)
# reorganize columns
cols = batting_df.columns.tolist()
batting_df = batting_df[cols[-1:] + cols[:-1]]

fielding_df = bbrefID_to_playerID(fielding_df)
# reorganize columns
cols = fielding_df.columns.tolist()
fielding_df = fielding_df[cols[-1:] + cols[:-1]]

appearances_df = bbrefID_to_playerID(appearances_df)
# reorganize columns
cols = appearances_df.columns.tolist()
appearances_df = appearances_df[cols[:3] + cols[-1:] + cols[3:-1]]

# fill lgID column of appearances table with lgID column of batting table
appearances_df.drop('lgID', axis=1, inplace=True)
deduped_batting_df = batting_df.loc[batting_df.groupby(['playerID', 'yearID'], as_index=False, sort=False)['G'].idxmax()]
deduped_batting_df = deduped_batting_df[['playerID', 'yearID', 'lgID']]
appearances_df = pd.merge(appearances_df, deduped_batting_df, how='left', on=['playerID', 'yearID'])
appearances_df['lgID'].fillna('AL', inplace=True)  # fill remaining missing data with AL (since probably AL pitchers)
# reorganize columns
cols = appearances_df.columns.tolist()
appearances_df = appearances_df[cols[:2] + cols[-1:] + cols[2:-1]]

batting_df.to_csv('../../data/raw/Batting2018.csv', index=False)
fielding_df.to_csv('../../data/raw/Fielding2018.csv', index=False)
appearances_df.to_csv('../../data/raw/Appearances2018.csv', index=False)
