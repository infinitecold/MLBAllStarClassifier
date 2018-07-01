import matplotlib.pyplot as plt
import pandas as pd

# parameters
x_stat = 'yearID'
x_stat_name = 'Year'
y_stat = 'AVG'
y_stat_name = 'Season Batting Average'
games_threshold = 100

# read in data from CSV
stats_df = pd.read_csv('preprocessed/stats.csv')

# initialize figure and subplot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# batting average vs year (more than 100 G)
non_all_stars = stats_df[(stats_df['all_star?'] == 0) & (stats_df['G'] > games_threshold)]
all_stars = stats_df[(stats_df['all_star?'] == 1) & (stats_df['G'] > games_threshold)]
ax.scatter(non_all_stars[x_stat], non_all_stars[y_stat], s=1, c='r', label='Non All-Stars')
ax.scatter(all_stars[x_stat], all_stars[y_stat], s=1, c='b', label='All-Stars')
ax.set_xlabel(x_stat_name)
ax.set_ylabel(y_stat_name)
ax.set_title('%s vs. %s' % (x_stat_name, y_stat_name))
ax.legend(loc='lower right', scatterpoints=1)

plt.show()
