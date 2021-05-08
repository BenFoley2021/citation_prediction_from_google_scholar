# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:38:00 2021

need 2D hist of relative error as f(y)
do this today
@author: Ben Foley
"""


import pandas as pd
import pickle
import seaborn as sns
#from rawDataToBagOfWords_oneHotEncode_3 import *
import numpy as np
import os
from generic_func_lib import *
import matplotlib.pyplot as plt
#from makingPrelimAnalFigs import *

os.getcwd()

def putResInDf(df,resDict):
    
    df['pred'] = df['index1'].apply(lambda x: resDict[x][0])
    df['actual'] = df['index1'].apply(lambda x: resDict[x][1][0])
    df['res'] = df['actual'] - df['pred']

    return df


def setUp(df_to_read, resDict):

    #### importing and cleaning
    df = pd.read_csv(df_to_read)
    
    #resDict = pickle.load(open(res_dict_to_read, "rb"))

    # setting index as a column
    df['index1'] = df.index
    #df['index1'] = df['index1'].astype(str)
    ### keeping only rows that were in the test set
    df2 = df[df['index1'].apply(lambda x: x in resDict) == True] #this seems really slow
    
    listIds = df2['index1'].to_list()
    
    # df['year'] = df['year'].astype(int)
    # df['cited'] = df['cited'].astype(int)
    
    #converting to cited per (and making sure they still line up with whats in resDict)
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
    
    return (x.pred - x.actual)/(min(x.pred, x.actual) +1 )
    
def basicHist(df2,col,xLow,xHigh,bins):
    resDist = sns.distplot(x = df2[col],bins = bins, kde = False)
    #resDist.set(yscale="log")
    resDist.set(xlim = (xLow,xHigh))
    resDist.set_title(label = 'residual histogram')        
    resDist.set(ylabel='count')
    resDist.set(xlabel='residual')

def make_shuffle_rel_res(df):
    """ shuffles the actual cites_per_year, then calculates the rel res as
        was done for the preds
    """
    def custom_1(x):
        return (x.actual_shuffle - x.actual)/(min(x.actual_shuffle, x.actual) +1 )
    
    import numpy.random
    df['actual_shuffle'] = numpy.random.permutation(df['actual'].values)

    df['shuffle_rel_res'] = df.apply(custom_1, axis =1)
    
    return df

def compare_res_rel(df):
    """ plots histograms of the relative residuals compared to the 

    """
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    
    plt.figure()
    sns.distplot(x = df['relativeRes'], kde = False, label =  'relative residuals')
    sns.distplot(x = df['shuffle_rel_res'], kde = False, label =  'shuffled residuals')
    plt.legend()
    
    
def compare_res(df):
    
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    plt.figure()
    sns.distplot(x = df[df['actual'] < 20]['actual'], kde = False, label =  'actual')
    sns.distplot(x = df['pred'], kde = False, label =  'predicted')
    plt.legend()
    plt.xlim((0, 100))
    
def two_d_hist(df, x_lims, y_lims):
    # relativeRes
    # actual
    plt.figure()
    sns.displot(df, x="actual", y="relativeRes", kind="kde", levels = 5)
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    
def two_d_scatter(df, x_lims, y_lims, hue = None):
    df['year'] = df['year'].astype(int)
    #sns.color_palette("Spectral", as_cmap=True)
    plt.figure()
    sns.scatterplot(data = df, x='actual', y='relativeRes', hue = hue,\
                    alpha=0.1, palette = 'Spectral')
    
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    
    
def fraction_correct_and_order(res_dict_to_read):
    """ Given index, actual, and predicted, do
        calculate the % got correct for a quantile (eg top 10%)
        sort each these for actual and predicted
        
        make colormap where order (1-N) is color, 
    """
    dict_to_df = {}
    for key, val in res_dict_to_read.items():
        dict_to_df[int(key)] = (val[0], val[1])
        
    df = pd.DataFrame(dict_to_df).transpose().reset_index()
    df = df.rename(columns = {0: 'preds', 1: 'actual'})
    quantile = 0.1
    
    # need to get quantiles for actual and predicted
    df = df.sort_values(by = 'actual', ascending = False)
    df_top_actual = df.iloc[0:int(np.round(len(df)*quantile))]
    
    df = df.sort_values(by = 'preds', ascending = False)
    df_top_preds = df.iloc[0:int(np.round(len(df)*quantile))]
    
    fraction_correct = len(df_top_preds.merge(df_top_actual)) / len(df_top_actual)
    print(fraction_correct)
    
    return None
    
if __name__ == "__main__":
    
    res_dict_to_read = load_one_pickled("res_dct.pckl", "model_related_outputs")
    
    fraction_correct_and_order(res_dict_to_read)
    
    df_file_name = 'df_to_lead_test.csv'
    
    df2 = setUp(df_file_name, res_dict_to_read)
    #df2['relativeRes'] = df2.apply(getRelativeRes, axis =1)

    # df2 = make_shuffle_rel_res(df2)
    
    # compare_res_rel(df2)
    
    # compare_res(df2)
    
    # two_d_scatter(df2, (0, 100), (-50, 10), hue = 'year')

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
