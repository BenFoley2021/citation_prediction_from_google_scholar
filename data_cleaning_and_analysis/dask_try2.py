# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:51:04 2021
https://gdcoder.com/speed-up-pandas-apply-function-using-dask-or-swifter-tutorial/

other options https://stackoverflow.com/questions/31361721/python-dask-dataframe-support-for-trivially-parallelizable-row-apply

@author: Ben
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
import multiprocessing
import os
import time
import numba
from dask.distributed import Client, LocalCluster

def test_fun(x,y):
    return x - y

test_fun_n = numba.jit(test_fun)

if __name__ == "__main__":
    
    cluster = LocalCluster()
    client = Client(cluster)
    cluster.scale(4)
    
    file_name = "df_for_results__27-03-2021 16_20_32.csv"
    cwd = os.getcwd()
    path = cwd + "\\cleaned_data\\"
    
    df = pd.read_csv(path + file_name)
    
    n_part = 4
    
    startTime = time.time()
    d_data = dd.from_pandas(df, \
        npartitions = n_part).map_partitions(lambda df: \
    df.apply(lambda row: test_fun_n(row.cited_num, row.cites_per_year), axis = 1)).compute(scheduler = 'threads')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))                          
                                              
    startTime = time.time()
    df_temp = df.apply(lambda row: test_fun_n(row.cited_num, row.cites_per_year), axis = 1)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime)) 