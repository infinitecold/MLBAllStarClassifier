import pandas as pd
import xml.etree.ElementTree as ET

stat_type = 'Batting'
input_file_path = '%s_raw_071718.xml' % stat_type
output_file_path = '../data/raw/%s2018.csv' % stat_type

if stat_type == 'Batting':
    raw_columns = ['playerID', 'age', 'teamID', 'lgID', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS',
                   'BB', 'SO', 'BA', 'OBP', 'slugging', 'OPS', 'OPS+', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
                   'Pos Summary']
    final_columns = 'playerID,yearID,stint,teamID,lgID,G,AB,R,H,2B,3B,HR,RBI,SB,CS,BB,SO,IBB,HBP,SH,SF,GIDP'.split(',')
elif stat_type == 'Fielding':
    raw_columns = ['playerID', 'age', 'teamID', 'lgID', 'G', 'GS', 'CG', 'InnOuts', 'Ch', 'PO', 'A', 'E', 'DP', 'Fld%',
                   'Rtot', 'Rtot/yr', 'Rdrs', 'Rdrs/yr', 'RF/9', 'RF/G', 'Pos Summary']
    final_columns = 'playerID,yearID,stint,teamID,lgID,POS,G,GS,InnOuts,PO,A,E,DP,PB,WP,SB,CS,ZR'.split(',')
elif stat_type == 'Appearances':
    raw_columns = ['playerID', 'age', 'teamID', 'yrs', 'G_all', 'GS', 'G_batting', 'G_defense', 'G_p', 'G_c', 'G_1b',
                   'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_of', 'G_dh', 'G_ph', 'G_pr']
    final_columns = 'yearID,teamID,lgID,playerID,G_all,GS,G_batting,G_defense,G_p,G_c,G_1b,G_2b,G_3b,G_ss,G_lf,G_cf,G_rf,G_of,G_dh,G_ph,G_pr'.split(',')
else:
    raise ValueError('stat_type is invalid: must be \'Batting\', \'Fielding\', or \'Appearances\'')

tree = ET.parse(input_file_path)
root = tree.getroot()

data = []
for row in root:
    row_data = []
    if row.get('class') == 'league_average_table':  # skip final league average row
        continue
    for stat in row.iter('td'):
        if stat.get('data-stat') == 'player':
            row_data.append(str(stat.get('data-append-csv')))
        elif stat.get('data-stat') == 'team_ID':
            if stat_type == 'Batting' and stat.text == 'TOT':
                break
            if stat.find('a') is not None:  # if hyperlinked team
                row_data.append(str(stat.find('a').text))
            else:  # if non-hyperlinked (TOT, 2TM, 3TM, etc.)
                row_data.append(str(stat.text))
        else:
            row_data.append(str(stat.text))
    if len(raw_columns) == len(row_data):
        data.append(tuple(row_data))

df = pd.DataFrame(data, columns=raw_columns)

# fill missing columns
df['yearID'] = 2018

if stat_type == 'Batting':
    df['stint'] = ''
elif stat_type == 'Fielding':
    df['stint'] = ''
    df['POS'] = df['Pos Summary'].apply(lambda x: str(x).split('-')[0])
    df['PB'] = ''
    df['WP'] = ''
    df['SB'] = ''
    df['CS'] = ''
    df['ZR'] = ''
elif stat_type == 'Appearances':
    df['lgID'] = ''

df[final_columns].to_csv(output_file_path, index=False)
