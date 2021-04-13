# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:24:20 2021

@author: Ben
"""
import pandas as pd
from functools import partial
import os
# A normal function
def f(a, b, c, x):
    return 1000*a + 100*b + 10*c + x
  
def custom1(x, y):
    return x + y
  
def custom2(x, y):
    return x.cited_num + x.cites_per_year + y

def test_df(col_name, y, df):
    df[col_name] = df['cited_num'].apply(lambda x: x+ y)
    return df


# A partial function that calls f with
# a as 3, b as 1 and c as 4.
g = partial(f, 3, 1, 4)
  
# Calling g()
print(g(5))

file_name = "df_for_results__27-03-2021 19_27_54.csv"

df = pd.read_csv(file_name)
df = df[["cited_num", "cites_per_year"]]
fp = partial(custom1, 2)

fp2 = partial(test_df, "test_3", 3)

df['test'] = df['cited_num'].apply(lambda x: fp(x))

df = test_df("test_2", 3, df)

df = fp2(df)

#df['test3'] = df.apply(lambda x: fp2, axis = 1)