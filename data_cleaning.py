"""
This module is for your data cleaning.
It should be repeatable.

## PRECLEANING
There should be a separate script recording how you transformed the json api calls into a dataframe and csv.

## SUPPORT FUNCTIONS
There can be an unlimited amount of support functions.
Each support function should have an informative name and return the partially cleaned bit of the dataset.
"""
import pandas as pd

def get_years(data):
    a = data.loc[((data['Year'] <= 2017.0) & (data['Year'] > 1997.0 ))]
    return a

def get_columns(data):
    b = data[['Player','Pos','FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
       '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
       'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'G']]
    return b

def drop_na_reset_index(data):
    c = data.dropna()
    d = c.reset_index(drop = True)
    return d

def make_new_columns(data):
    data['ORB/G'] = round(data['ORB']/data['G'],2)
    data['DRB/G'] = round(data['DRB']/data['G'],2)
    data['TRB/G'] = round(data['TRB']/data['G'],2)
    data['AST/G'] = round(data['AST']/data['G'],2)
    data['STL/G'] = round(data['STL']/data['G'],2)
    data['BLK/G'] = round(data['BLK']/data['G'],2)
    data['TOV/G'] = round(data['TOV']/data['G'],2)
    data['PF/G'] = round(data['PF']/data['G'],2)
    data['PTS/G'] = round(data['PTS']/data['G'],2)
    return data

def final_columns(data):
    e = data[['Player', 'Pos', 'FG%', '3P%','2P%', 'FT%','ORB/G', 'DRB/G', 'TRB/G', 'AST/G',
       'STL/G', 'BLK/G', 'TOV/G', 'PF/G', 'PTS/G']]
    return e

def group_by_means(data):
    f = data.groupby('Player')[['FG%', '3P%','2P%', 'FT%','ORB/G', 'DRB/G', 'TRB/G', 'AST/G',
       'STL/G', 'BLK/G', 'TOV/G', 'PF/G', 'PTS/G']].mean()
    h = f.round(3)
    return h

def make_unique(data):
    data['POS'] = data['Pos'].unique()
    return data
    
def del_duplicate_positions(data):
    data['POS'] = data['POS'].map(lambda x: x[0])
    return data


# for new csv
def drop_columns(data):
    i = data.drop(['collage', 'born', 'birth_city', 'birth_state'], axis=1)
    return i

def left_merge(a, b):
    j = pd.merge(a, b, on = 'Player', how = 'left')
    return j
    

    
# full clean 
def full_clean():
    """
    This is the one function called that will run all the support functions.
    Assumption: Your data will be saved in a data folder and named "dirty_data.csv"

    :return: cleaned dataset to be passed to hypothesis testing and visualization modules.
    """
    dirty_data = pd.read_csv("./nba-players-stats/Seasons_Stats.csv")

    cleaning_data1 = get_years(dirty_data)
    cleaning_data2 = get_columns(cleaning_data1)
    cleaning_data3 = drop_na_reset_index(cleaning_data2)
    cleaning_data4 = make_new_columns(cleaning_data3)
    cleaning_data5 = final_columns(cleaning_data4)
    cleaning_data6 = group_by_means(cleaning_data5)
    cleaning_data7 = make_unique(cleaning_data6)
    cleaning_data8 = del_duplicate_positions(cleaning_data7)
    
    names_data = pd.read_csv("./nba-players-stats/Players.csv", index_col = 0)
    
    cleaning_data9 = drop_columns(names_data)
    cleaned_data = left_merge(cleaning_data8, cleaning_data9)
    
    cleaned_data.to_csv('./cleaned_for_testing.csv')
    
    return cleaned_data