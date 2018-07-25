# URLs:
# https://www.baseball-reference.com/leagues/MLB/2018-standard-batting.shtml
# https://www.baseball-reference.com/leagues/MLB/2018-standard-fielding.shtml
# https://www.baseball-reference.com/leagues/MLB/2018-appearances-fielding.shtml

import urllib2

url = 'https://www.baseball-reference.com/leagues/MLB/2018-standard-batting.shtml'
output_file_path = '../../data/bbref/Batting_raw_071718.xml'

request = urllib2.urlopen(url)
raw_html = request.read()

# get html starting from second table
raw_html = raw_html.split('<table ')[2]

# get contents between table body tags
raw_table = raw_html[raw_html.find('<tbody>'):raw_html.find('</tbody>')]

# get actual table rows
raw_table_rows = raw_table.split('\n')
table_rows = [row.replace('&nbsp;', ' ') for row in raw_table_rows if row.startswith('<tr')]

# output to file in xml format
with open(output_file_path, 'w') as f:
    f.write('<?xml version="1.0"?>\n<data>\n%s\n</data>\n' % '\n'.join(table_rows))
