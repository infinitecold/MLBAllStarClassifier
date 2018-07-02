import matplotlib.pyplot as plt
import pandas as pd

# parameters
x_stat = 'H'
x_stat_name = 'Hits'
y_stat = 'AVG'
y_stat_name = 'Batting Average'
games_threshold = 20

# read in data from CSV
stats_df = pd.read_csv('preprocessed/stats.csv')

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# initialize figure and subplot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# batting average vs year
stats_df = stats_df[stats_df['G'] > games_threshold]
non_all_stars = stats_df[stats_df['all_star?'] == 0]
all_stars = stats_df[stats_df['all_star?'] == 1]
ax.scatter(non_all_stars[x_stat], non_all_stars[y_stat], s=1, c='r', label='Non All-Stars')
ax.scatter(all_stars[x_stat], all_stars[y_stat], s=1, c='b', label='All-Stars')
ax.set_xlabel(x_stat_name)
ax.set_ylabel(y_stat_name)
ax.set_title('Season %s vs. %s' % (x_stat_name, y_stat_name))
ax.legend(loc='best', scatterpoints=1)

plt.show()
