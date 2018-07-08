import pandas as pd

# parameters
first_allstar_year = 1933
current_year = 2018

# read in data from CSVs
stats_df = pd.read_csv('data/preproc/stats.csv')
allstar_df = pd.read_csv('data/raw/AllstarFull.csv')

with open('data/preproc/team_sizes.csv', 'w') as output_file:
    output_file.write('yearID,AL,NL\n')
    for year in range(first_allstar_year, current_year):
        AL = NL = 0
        annual_allstars = stats_df[(stats_df['yearID'] == year) & (stats_df['all_star?'] == 1)]
        for _, allstar in annual_allstars.iterrows():
            league = allstar_df[(allstar_df['playerID'] == allstar['playerID']) & (allstar_df['yearID'] == year)]['lgID'].iloc[0]
            if league == 'AL':
                AL += 1
            elif league == 'NL':
                NL += 1
            else:
                raise ValueError('Unknown league found for %s (%d)' % (allstar['playerID'], allstar['yearID']))
        output_file.write('%d,%d,%d\n' % (year, AL, NL))
