# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:38:00 2021
	Unnamed: 0	ids	title	jref	cited	scrapTitle	year	doi
0	0	0704.0135	A Single Trapped Ion as a Time-Dependent Harmonic Oscillator	Phys. Rev. A 76, 052105 (2007)	1.8571428571428572	"data-clk-atid=""0YWF6r0pXJ0J"">Single trapped ion as a time-dependent harmonic oscillator"	2007	10.1103/PhysRevA.76.052105

@author: Ben Foley
"""


import pandas as pd
import pickle
import seaborn as sns
#from rawDataToBagOfWords_oneHotEncode_3 import *

import os
from generic_func_lib import *
#from makingPrelimAnalFigs import *

os.getcwd()


def putResInDf(df,resDict):
    
    df['pred'] = df['index1'].apply(lambda x: resDict[x][0])
    df['actual'] = df['index1'].apply(lambda x: resDict[x][1])
    df['res'] = df['actual'] - df['pred']

    return df


def setUp(df_to_read, res_dict_to_read):

    #### importing and cleaning
    df = pd.read_csv(df_to_read)
    
    resDict = pickle.load(open(res_dict_to_read, "rb"))
    

    
    # setting index as a column
    df['index1'] = df.index
    df['index1'] = df['index1'].astype(str)
    ### keeping only rows that were in the test set
    df2 = df[df['index1'].apply(lambda x: str(x) in resDict) == True] #this seems really slow
    
    
    listIds = df2['index1'].to_list()
    
    # df['year'] = df['year'].astype(int)
    # df['cited'] = df['cited'].astype(int)
    
    #converting to cited per ( and making sure they still line up with whats in resDict)
    #df['cited'] = df['cited'] / (int(2021) - df['year'])
    
    #df2.to_csv('resDf_2-26')
    
    ## making df for further analysis with important cols
    df2 = df2[['index1','titleID','year','Journal','Authors','cites_per_year']]
    
    ### placing y_pred and y_actual in df2
    df2 = putResInDf(df2,resDict)
    
    ### quick manual check to make sure y_actual and cited from the original df match
    #testVar = df[df['ids'] == '0704.0163']
    
    return df2

#################
#making hists of the residuals
# 1D: want average res vs y_actual
# need to rebin df for this

# 2D want hist of res as f(y_actual). try and use exisintg functs in prelimAnalFigs

def getRelativeRes(x):
    """ want to normalize the residuals by the min of the predicted and actual
        This will show which predictions are off by a large amount. For example,
        if the actual was 20 and we predict 40 thats not too bad (off by 2x). but if the actual 
        was 20 and we predict 0.5 thats terrible (off by 40x)
    
        this metric (or similar) is a candiate for a custom loss function in future work
    
        The current problem is how to make this handle when the min is 0. Easiest to just 
        let the model predict on the raw data, and put +1 in the denominator 
    """
    
    return (x.pred - x.actual)/min(x.pred,x.actual)
    
def basicHist(df2,col,xLow,xHigh,bins):
    resDist = sns.distplot(x = df2[col],bins = bins, kde = False)
    #resDist.set(yscale="log")
    resDist.set(xlim = (xLow,xHigh))
    resDist.set_title(label = 'residual histogram')        
    resDist.set(ylabel='count')
    resDist.set(xlabel='residual')


if __name__ == "__main__":
    
    res_dict_to_read = load_one_pickled("resDict.pckl", "model_related_outputs")
    
    df_list = load_all_dfs("cleaned_data")
    df_to_read = cat_dfs(df_list)
    
    df2 = setUp(df_to_read, res_dict_to_read)
    df2['relativeRes'] = df2.apply(getRelativeRes, axis =1)

#df2 = pd.read_csv('resDF_2-26.csv')

if __name__ != "__main__":
    df2['normRes'] = df2['res']/df2['actual']
    
    df2['relativeRes'] = df2.apply(getRelativeRes, axis =1)
    
    ######### basic residual histogram. this isn't useful because the counts need to 
    # normalized by 
    
    
    
    #make2DHist(df2,'res','actual')
    
    #2Dhist(df2)
    xLow = -20
    xHigh = 20
    bins = 10000
    basicHist(df2,'res',-20,20,1000)
    
    xLow = -10
    xHigh = 10
    bins = 5000
    basicHist(df2,'relativeRes',xLow,xHigh,bins)
