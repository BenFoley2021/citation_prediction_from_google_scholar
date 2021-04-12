# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:03:01 2021

@author: Ben
"""

from dask.distributed import Client, LocalCluster
from dask import delayed

if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)

    %%time
# This runs immediately, all it does is build a graph

    x = delayed(inc)(1)
    y = delayed(inc)(2)
    z = delayed(add)(x, y)