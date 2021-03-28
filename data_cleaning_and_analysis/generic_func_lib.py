# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:33:21 2021

@author: bcyk5
"""
import pandas as pd
def load_all_dfs(out_dir) -> list:
    """ loads all the dfs in the out dir into a list.
    
        If there are other csvs in the folder then this function
        will think they are dataframes

    Parameters
    ----------
    out_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    df_list: list
        A list of each pd.DataFame object saved as a .csv in the directory

    """
    import os

    
    df_list = []
    
    for file in os.listdir(out_dir):
        if file.endswith('.csv'):
            df_list.append(pd.read_csv(out_dir + "\\" + file))
    
    return df_list

def cat_dfs(df_list: list) -> pd.DataFrame:
    """ cats all df in df list. will probably want to add some 
        more logic in.
        
    Parameters
    ----------
    df_list : list
        a list containing pd.DataFrame objects to be combined. These will
        usually (but no always) have the same columns. 
    
    Returns
    -------
    df: pd.DataFrame
        The data frame resulting from concatinating all the pd.DataFrames
        in df_list
        
        
    """
    
    
    df = pd.DataFrame()
    
    for frame in df_list:
        df = pd.concat([df, frame])
    
    return df

def save_pickles(fListIn,NListIn,outLoc):
        ###pickles the outfile and saves it to a dir
    import pickle
    
    #wrkDir = os.getcwd()
    #'relative/path/to/file/you/want'
    #os.chdir('')
    fileNames = NListIn
    vars2Save = fListIn
    for i,fileName in enumerate(fileNames):
        fileName = fileName + '.pckl'
        path = outLoc + '/' + fileName
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(vars2Save[i], f)
        f.close()
        
        
        
def load_one_pickled(fileName, path):
    import pickle
    fileName = path + "\\" +  fileName
    with open(fileName,'rb') as f:
        outName = pickle.load(f)
    
    return outName


def back_up_in_file_tree(path: str, num: int) -> str:
    """ want a generic function which returns the path of the folder 
        above the input path (or num levels above)
    

    Parameters
    ----------
    path : str
        The file path to be modified.
    num : int
        number of times to back up (or levels to go up in the file tree).

    Returns
    -------
    path: str
        the path after the backup operation preformed

    """
    import os
    
    for _ in range(num):
        path = os.path.dirname(path)
    
    
    return path
