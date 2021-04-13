# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:03:01 2021

@author: Ben
"""

from dask.distributed import Client, LocalCluster
from dask import delayed
import time

def inc(x):
    sleep(1)
    return x + 1

def add(x, y):
    sleep(1)
    return x + y

if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    cluster.scale(3)

# This runs immediately, all it does is build a graph

    x = delayed(inc)(1)
    y = delayed(inc)(2)
    z = delayed(add)(x, y)
    #z.compute()
    #z.visualize()
    
    
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    
    results = []

    for x in data:
        y = delayed(inc)(x)
        results.append(y)
        
    total = delayed(sum)(results)
    print(total)