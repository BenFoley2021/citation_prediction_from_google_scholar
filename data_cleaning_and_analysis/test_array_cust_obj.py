# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:38:22 2021

@author: Ben
"""


import numpy as np

y = np.array([1,2,3,4,5,5,6])

pred = np.array([0,1,2,3,4,5,6])

test = abs(y-pred)/(np.minimum(pred,y)+1)