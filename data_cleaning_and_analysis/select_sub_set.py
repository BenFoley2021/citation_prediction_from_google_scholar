# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:50:57 2021

want to be able to select a subset of all the data, so can have more managle sized data
for experimenting. This will be done so the sampling is still as dense as possible (not randomly selecting)

first pass, use key words to select. If key words are in title, abstract, or 

example use general_multi_proc
df['titleID_list'] = general_multi_proc(str_col_to_list_par, df['titleID'], " ")

@author: Ben
"""
from generic_func_lib import *
import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import os

from one_hot_encode_parallelizeing import general_multi_proc


def are_keywords_there(df, keywords, locations):
    """ Checks if keywords are found in the df at locations.
    """
    def check_words(x):
        for place in locations:
            for word in keyowrds:
                if word in x[place].lower():
                    return True


    df = df[df.apply(check_words) == True]
    #more logic needed?
    return df


if __name__ == "__main__":
    
    #### do not delete!!!!!!!!!!!!!!
    # out_dir = "cleaned_data"
    
    # df_list = load_all_dfs(out_dir)
    
    # df = cat_dfs(df_list)
    #### do not delete!!!!!!!!!!!!!!
    
    file_name = "df_for_results__27-03-2021 16_20_32.csv"
    cwd = os.getcwd()
    path = cwd + "\\cleaned_data\\"
    df = pd.read_csv(path + file_name)
    
    
    keywords = ['battery', 'batteries', 'li-ion']
    
    
    
    