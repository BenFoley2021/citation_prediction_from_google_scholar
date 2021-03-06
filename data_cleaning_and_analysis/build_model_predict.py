# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:20:04 2021
5/4 things to try: https://pythonhealthcare.org/2019/04/12/122-oversampling-to-correct-for-imbalanced-data-using-naive-sampling-or-smote/#:~:text=SMOTE%20with%20continuous%20variables,point%20in%20to%20the%20sample.

5/2
    to do: use k fold, switch o smote for resampling
    try using ranked order, write thing to plot the number captured as 
    f(fraction selected)


4/29
    having trouble getting reasonable fits with the perovskite data set
    betting a slightly bigger set to try

    the imbalance makes classification harder
    
    try oversampling, then smote
    https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

4/26:
    runs with the prelimiany perovskite data, ~20k articles
    xgb boost able to capture some variance when svd done on X, (at least it doesn't guess the
    average for everything')
    
    need to go back and get a real residual analysis working
    
    functions used
    #run_script() gets the data 
    run_pipe = run_xgb_pipe()

to do:
    make an estimator/transform which decides k for svd by going to at least 
    X explanined variance
        running this takes a really long time. just use grid search
    
    
    
    objective function id like
    obj = abs(y - pred)/min(y, pred) # need to test this to make the min works
    as intended
    
    resources 
    https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
    
    https://stats.stackexchange.com/questions/484340/the-hessian-in-xgboost-loss-function-doesnt-look-like-a-square-matrix
    
    k fold cross val with xgboost
    resuources 
        https://www.kaggle.com/prashant111/xgboost-k-fold-cv-feature-importance
        https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/
        
        
    general xgboost 
    https://projector-video-pdf-converter.datacamp.com/3679/chapter2.pdf
    https://github.com/tqchen/xgboost/tree/master/demo    
    
    example of hyper parameter tuning for predicting the number of comments
    as fb post gets https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
    
    
@author: Ben Foley
"""
from generic_func_lib import *
from one_hot_encode_parallelizeing import *

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

#from numpy import savez_compressed
#from numpy import load
import scipy.sparse
#from scipy.sparse import coo_matrix, lil_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pickle as pckl

from sklearn.pipeline import Pipeline
from sklearn import base
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import math

#import tensorflow as tf

def loadPickled2(path):
    """ loads the pickled files in a specfied directory

    ----------
    path : string
        the subfolder to load files from

    Returns
    -------
    varList : list
        DESCRIPTION: list storing each variable that was found as a .pckl file
    nameList : list
        list of strings, names of files corresponding to the vars in varlist

    """
    #import pickle as pckl
    import os

    
    var_name_dict = {}
    
    path = os.getcwd() + '/' + path +'/'
    
    wrkDir = os.getcwd()
    os.chdir(path)
    
    for file in os.listdir(path):
        print(file)
        if file.endswith('.pckl'):
            #nameList.append(file)
            with open(file,'rb') as f:
                var_name_dict[file] = pckl.load(f)
                #varList.append(pckl.load(f))
                f.close()
            
    os.chdir(wrkDir)
    return var_name_dict ####### returning a list of all the variables
    
def loadFiles2(path):
    #gets the npzs, this probably should be combined with loadpickled by making them polymorphic. do if have time later
    import os

    var_names_npz = {}
    #fileName = path + fileName + '.pckl'
    path = os.getcwd() + '/' + path +'/'

    wrkDir = os.getcwd()
    os.chdir(path)
    
    for file in os.listdir(path):
        #print(file)
        if file.endswith('.npz'):
            with open(file,'rb') as f:
                var_names_npz[file] = scipy.sparse.load_npz(f)
                #dummies.append(scipy.sparse.load_npz(f))
                f.close()

    os.chdir(wrkDir)
            
    return var_names_npz #### return list of variables loaded

    

    # print('changed to np arrays')
    
    
    #return X,y, inputs, names

def loadInputData2(dirToRead): ### another ad hoc function to load the data for when the files needed to be split up becuase i can't upload things larger than 25 MB to github
    """ loads data for ML
    X = sparse matrix
    y = array of float, the citations per year
    inputs = list of y and dictionaries which have all the data used to 
    construct X
    """    

    #dirToRead = 'one_hot_encoded_data'
    
    var_names_dict = loadPickled2(dirToRead)
    var_names_npz = loadFiles2(dirToRead)
    
    return var_names_dict, var_names_npz

def tryCrossVal(X,y,model): #### not used in current run
    kfold = KFold(n_splits=3)
    results = cross_val_score(model, X, y, cv=kfold, scoring = 'neg_root_mean_squared_error')
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean(), results.std()))

    return results
    
def manTTS(keys,X,y, ind): #### setting up the train test split
    #doing it this way so I can keep track of the ids for each paper (row)
    #split hardcoded to 80/20 rn

     # hard coded to have the ones from 2019
    
    #keys = list(dicMain.keys())
    
    Xm_train = X[0:ind,:]
    Xm_test = X[ind:-1,:]
    
    ym_train = y[0:ind]
    ym_test = y[ind:-1]
    
    keys_train = keys[0:ind]
    keys_test = keys[ind:-1]
    
    
    return Xm_train, Xm_test, ym_train, ym_test, keys_train, keys_test

def getListCitedBy(dicMain):
    #### this gets a list of the citation numbers from the main dict. 
    tempList = []
    for thing in dicMain:
        tempList.append((dicMain[thing][0]))
    
    return tempList

def putResInDf(preds,cited_test,keys_test):

    #### puts the residuals back into the original df so that residuals can be analyzed
    
    from rawDataToBagOfWords_oneHotEncode_2 import getYear

    def testIfInDic(thing):
        # if the ids for that row is the test_keys, return true, otherwise false
        try:
            keysDic[thing]
            return True
        except:
            return False
        
        
    def checkRow2(row):
        if int(keysDic[row['ids']][1]) == int(row['cited']): #### making sure that citations (target) from the data fed to the model is consistent with whats in the df. this is just a gravity check
            return int(row['cited']) - int(keysDic[row['ids']][0]) ### return the residual
        else:
            return np.nan #### if the data got mixed up somehow, return nan
            
        
            
    def ifInDic2(thing): #### checking to see if the ids for that row is in the dict
        try:
            keysDic[thing]
            return 'yes'
        except:
            return 'no'
            

    predActual = list(zip(preds, cited_test))  
    keysDic = dict(zip(keys_test, predActual))
    
    dfIn = pd.read_csv('processed_2-8.csv')  
    dfIn['ids'] = dfIn['ids'].astype('str')  
    #keysDic = dict(zip(keys_test, keys_test)) ####### making a dict out of the keys_test list
    # for faster lookup
    print('loaded df')
    dfIn = getYear(dfIn)
    
    
    # apply is way faster than iterrows    
    dfIn['temp'] = dfIn.apply(lambda x: ifInDic2(x.ids),axis = 1)
    
    dfIn = dfIn[dfIn['temp'] == 'yes'] ##### dropping all the rows that weren't in the test set

    print('dropped things not in keys_test')
    dfIn['res'] = 0
    
    ####### there are a few rows from the test set which don't match up with the data frame
    ####### this may be a formatting issue, or i loaded an older version of the df which is missing some data
    missingIds = []
    for i,row in dfIn.iterrows():
        try:
            keysDic[row['ids']]
        except:
            missingIds.append(row['ids'])
            
    ###### the line below calculates the residuals column, provided data is consistent
    dfIn['res'] = dfIn.apply(lambda x: checkRow2(x),axis = 1)
        
    print('added residuals')

    return dfIn, missingIds

def run_xgboost_k_fold(Xm_train,ym_train):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from xgboost import cv
    
    data_dmatrix = xgb.DMatrix(data=Xm_train,label=ym_train)
    
    params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3,' learning_rate': 0.1, \
                'max_depth': 5, 'alpha': 10}
    
    
    xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3, \
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
    
    return xgb_cv

def runXGboost(Xm_train,ym_train):# ,ym_train,ym_test,keys_train,keys_test,cited_train,cited_test):
    """ script to fit input data to XG boost model
    
        should reduce the scope of this function to just training the xgboost

    """
    import xgboost as xgb
    from typing import Tuple
    ###########################
    # example code for custom objective function
    def gradient_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        ''' plugged  d/d(predt) abs(y-predt)/(np.minimum(predt, y)+1) into wolfram'''
        y = dtrain.get_label() + 1
        #return abs(y-predt)/(np.minimum(predt, y)+1))
        grad = np.zeros(y.shape)
        for i, val in enumerate(y):
            if predt[i] > y[i]:
                grad[i] = 1/(y[i] + 1)
            else:
                grad[i] = -(y[i] + 1)/(predt[i]**2 + 1)
        #y[y < 0] = 10**10
        #grad = abs((((y-predt)**2)**0.5)/(np.minimum(predt, y) + 1))
        
        return grad
        #return (np.log1p(y) - np.log1p(predt)**2)/ (np.log1p(y) - 1)
    
    def hessian_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''

        return np.ones(predt.shape)
    
    def custom_obj(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        grad = gradient_c(predt, dtrain)
        hess = hessian_c(predt, dtrain)
        return grad, hess
    #####################################
    
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        y = dtrain.get_label()
        return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        return np.ones(predt.shape)
        #return ((-np.log1p(predt) + np.log1p(y) + 1) /
        #    np.power(predt + 1, 2))
    
    
    def squared_log(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        #predt[predt < -1] = -1 + 1e-6
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess
    
    ###########################
    
    data_dmatrix = xgb.DMatrix(data=Xm_train,label=ym_train) ### loading into xgboost format

    ########### running with k fold cross validation
    params = {'colsample_bytree': 0.3,'learning_rate': 0.1,
                    'max_depth': 20, 'alpha': 10}
    
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 20, 'alpha': 10}
    
    xg_reg = xgb.train(params=params, \
                       dtrain=data_dmatrix, \
                       num_boost_round=10, \
                       obj = custom_obj)
    print('trained')
    
    return xg_reg

def analyze_xgboost_model(xg_reg, Xm_test, cited_test, keys_test):
    xgb.plot_importance(xg_reg,max_num_features=3)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

    ########## getting prediction
    data_dmatrix_Xm_test = xgb.DMatrix(data=Xm_test)
    preds = xg_reg.predict(data_dmatrix_Xm_test)

    dfRes,missingIds = putResInDf(preds,cited_test,keys_test) #########putting the residuals back into the df for later analysis

def setUp4Keras(Xm_train, Xm_test, ym_train, ym_test):
    """ setting up inputs to used with keras
    scaling and experimenting with ways to put the sparse matrix in
    
    """
    #### min max scaler does not accept sparse input
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # Xm_train = scaler.fit_transform(Xm_train)
    # Xm_test = scaler.fit_transform(Xm_test)

    ##### creating td.sparse object

    #https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent

    coo = Xm_train.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, ym_train, coo.shape)

def trainKerasVal(Xm_train, Xm_test, ym_train, ym_test):
    #redoing trainKeras but with validation generator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    import tensorflow as tf
    from math import ceil
    
    def batch_generator(Xl, yl, batch_size): ####### should prob rewrite this
        number_of_batches = samplesPerEpoch/batchSize
        counter=0
        shuffle_index = np.arange(np.shape(yl)[0])
        np.random.shuffle(shuffle_index)
        Xl =  Xl[shuffle_index, :]
        yl =  yl[shuffle_index]
        while 1:
            index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = Xl[index_batch,:].todense()
            y_batch = yl[index_batch]
            counter += 1
            yield(np.array(X_batch),y_batch)
            if (counter < number_of_batches):
                np.random.shuffle(shuffle_index)
                counter=0
    
    
    model = Sequential()
    model.add(Dense(Xm_test.shape[1],activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(25,activation = 'relu'))
    model.add(Dropout(.3))
    model.add(Dense(10,activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    
    samplesPerEpoch=Xm_train.shape[0]
    batchSize = ceil(Xm_train.shape[0]/300)
    batchSizeV = ceil(Xm_test.shape[0]/300)
    
    # for log loss https://keras.io/api/losses/regression_losses/#mean_squared_logarithmic_error-function
    model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
    # model.fit_generator(generator = batch_generator, (Xm_train, ym_train, batch_size = batchSize),\
    #                     validation_data = batch_generator, (Xm_tets, ym_test, batch_size = batchSizeV), \
    #                     steps_per_epoch=1, epochs = 1)
    
    train_generator = batch_generator(Xm_train, ym_train, batch_size = batchSize)
    val_generator = batch_generator(Xm_test, ym_test, batch_size = batchSizeV)
    
    history = model.fit_generator(generator = train_generator, \
                        validation_data = val_generator, \
                        steps_per_epoch = 300, epochs = 5)
        
        
    return model, history

def trainKeras(Xm_train, Xm_test, ym_train, ym_test):
    """ training a keras model
    currently experimenting to get it to work with sparse inputs

    Parameters
    ----------
    Xm_train : TYPE
        DESCRIPTION.
    Xm_test : TYPE
        DESCRIPTION.
    ym_train : TYPE
        DESCRIPTION.
    ym_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    current problem:is out of order. Many sparse ops require sorted indices.
    Use `tf.sparse.reorder` to create a correctly ordered copy.
    https://stackoverflow.com/questions/61961042/indices201-0-8-is-out-of-order-many-sparse-ops-require-sorted-indices-use
    
    https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
    """    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    import tensorflow as tf
    from math import ceil
    
    def batch_generator(Xl, yl, batch_size): ####### should prob rewrite this
        number_of_batches = samplesPerEpoch/batchSize
        counter=0
        shuffle_index = np.arange(np.shape(yl)[0])
        np.random.shuffle(shuffle_index)
        Xl =  Xl[shuffle_index, :]
        yl =  yl[shuffle_index]
        while 1:
            index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = Xl[index_batch,:].todense()
            y_batch = yl[index_batch]
            counter += 1
            yield(np.array(X_batch),y_batch)
            if (counter < number_of_batches):
                np.random.shuffle(shuffle_index)
                counter=0
    
    
    model = Sequential()
    model.add(Dense(Xm_test.shape[1],activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(25,activation = 'relu'))
    model.add(Dropout(.3))
    model.add(Dense(10,activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    
    samplesPerEpoch=Xm_train.shape[0]
    batchSize = ceil(Xm_train.shape[0]/300)
    
    # for log loss https://keras.io/api/losses/regression_losses/#mean_squared_logarithmic_error-function
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit_generator(generator=batch_generator(Xm_train, ym_train, \
                    batch_size = batchSize), steps_per_epoch=300, epochs = 4)
    
    return model
        
def predictKeras(model,Xm):
    
    preds = []
    incSize = 5000
    numBatch = math.ceil(Xm_test.shape[0]/incSize)
    for ind in range(numBatch):
        if (ind + 1) * incSize > Xm_test.shape[0]:
            indEnd = Xm_test.shape[0]
        else:
            indEnd = (ind + 1) * incSize
        
        X_batch = Xm_test[(ind * incSize): indEnd, :].todense()
            
        preds.append(model.predict(X_batch))
    
    return preds

def saveResults(keys_test,preds,ym_test):
    #### basic file to save things, all it does is help me remeber
    #how to set up inputs for saveOutput2
    outLoc = 'fitting_3-3'
    from rawDataToBagOfWords_oneHotEncode_2 import saveOutput2

    fList = [keys_test,preds,ym_test]
    nList = ['keys_test','preds','ym_test']

    saveOutput2(fList,nList,outLoc)  ###### saving variables in the list with the desired names

def tempFlattenPreds(preds):
    # when prediction done in batches (memory issuses with large sparse)
    # need to flatten
    
    predsOut = []
    
    for ar in preds:
        for num in ar:
            predsOut.append(num[0])
            
    return predsOut

def packagePreds(preds,keys_test,ym_test,dicMain): ## obsoleted
    #puts the preds and actual into a dict with the id as the key
    
    def verifyOrder(resDict,dicMain):
        misMatchList = []
        for key in resDict:
            if dicMain[key][0] != resDict[key][1]:
                misMatchList.append(key)
        return misMatchList
            
    resDict = {} # residuals dictionary
    
    for i, key in enumerate(keys_test):
        try:
            resDict[key] = [preds[i],ym_test[i]]
        except:
            print(str(i) + ' ' + key)
        
    misMatchList = verifyOrder(resDict,dicMain)
        
    return resDict, misMatchList

def packagePreds_v2(preds, keys_test, ym_test):
    """ Puts the predictions in a dictionary with the index of the df as a key.
        The y_test is also included just as gravity check that nothing got mixed up.
        Residuals are processed this way to make it simple load them back into the 
        df for analysis.

    Parameters
    ----------
    preds : Array
        The prdictions from the model.
    keys_test : TYPE
        The index of that row (or data point) from the df.
    ym_test : TYPE
        The labels for the test set.

    Returns
    -------
    resDict : Dict
        Dictionary containing the predicted and actual value for each data point,
        where the key is the index of the df.

    """
    #puts the preds and actual into a dict with the id as the key
            
    resDict = {} # residuals dictionary
    
    for i, key in enumerate(keys_test):
        try:
            resDict[key] = [preds[i],ym_test[i]]
        except:
            print(str(i) + ' ' + str(key))
        
    return resDict
    
def shuffle(dicMain, X, y):
    from sklearn.utils import shuffle
    
    dicMainKeys = list(dicMain.keys())
    dicMainKeys = np.array(dicMainKeys)
    dicMainKeys, X, y = shuffle(dicMainKeys, X, y, random_state=0)
    
    return dicMainKeys, X, y

def run_for_custom_encoded():
    var_names_dict, var_names_npz = loadInputData2('one_hot_encoded_data')
    
    print('loaded vars')
    X = var_names_npz['bowVecSparseX.npz']
    y = var_names_dict['labelX.pckl']
    
    dicMain = var_names_dict['dicMain.pckl'] ###### grabing the main dictionary from the list of loaded variables
    
    dicMainKeys = list(dicMain.keys())
    #citedList = getListCitedBy(dicMain) i dont know what this was for
    
    Xm_train, Xm_test, ym_train, ym_test, keys_train, keys_test \
    = manTTS(dicMainKeys, X, y)
    
    xg_reg = runXGboost(X, y) # should really split model construction and training into seperate functions

    # getting accuracy from test set
    data_dmatrix = xgb.DMatrix(data=Xm_test)
    preds = xg_reg.predict(data_dmatrix)

    plt.hist(preds)
    # making a dictionary of the residuals. This is loaded by the residual analysis sccript, and the
    # placed into the data frame used for one hot encoding
    resDict, misMatchList = packagePreds(preds,keys_test,ym_test,dicMain)
    # saving resDict
    save_pickles([resDict], ["resDict"], "model_related_outputs")

def run_for_encoding_v2():
    
    var_names_dict, var_names_npz = loadInputData2('one_hot_encoded_data_v2')
    
    print('loaded vars')
    X = var_names_npz['bow_mat_X.npz']
    y = var_names_dict['labels.pckl']
    
    keys = var_names_dict['paper_ids.pckl']
    #need to get keys
    #citedList = getListCitedBy(dicMain) i dont know what this was for
    
    Xm_train, Xm_test, ym_train, ym_test, keys_train, keys_test \
    = manTTS(keys, X, y)
    
    xg_reg = runXGboost(X, y) # should really split model construction and training into seperate functions

    # getting accuracy from test set
    data_dmatrix = xgb.DMatrix(data=Xm_test)
    preds = xg_reg.predict(data_dmatrix)

    plt.hist(preds)
    # making a dictionary of the residuals. This is loaded by the residual analysis sccript, and the
    # placed into the data frame used for one hot encoding
    resDict = packagePreds_v2(preds,keys_test,ym_test)
    # saving resDict
    save_pickles([resDict], ["resDict"], "model_related_outputs")
    
    return resDict

class DimensionalityReducer(base.BaseEstimator, base.TransformerMixin):
    """ this if for svd / type estimator/transformers. find a value for k
        that explains at least 95% of variance, set that trained model as
        self. estimator
    """
    def __init__(self, estimator_class):
        self.estimator = estimator_class
    
    def fit(self, X):
        
        grid = np.logspace(0,3,20)
        grid = [math.ceil(num) for num in np.logspace(0,3,20) if math.ceil(num) > 300]
        
        for i, num in enumerate(grid):
            dim_reducer = self.estimator(num)
            dim_reducer.fit_transform(X)
            print('current explanied ratioence ratio is ' + sum(dim_reducer.explained_variance_ratio_))
            if sum(dim_reducer.explained_variance_ratio_) > 0.95:
                break
            if i == len(grid) - 1:
                print('DimensionalityReducer never got to 95% explained variance')
        
        
        self.estimator = dim_reducer # overwite the unfitted class with the fitted object of that class

        return self
    
    def transform(self, X):

        return self.estimator.transform(X)

def fetch_data():
    var_names_dict, var_names_npz = loadInputData2('one_hot_encoded_data_v2')
    print('loaded vars')
    X = var_names_npz['bow_mat_X.npz']
    y = var_names_dict['labels.pckl']
    keys = var_names_dict['paper_ids.pckl']
    idx = var_names_dict["year_idx.pckl"]
    return X, y, keys, idx

def set_classes(y: np.array, thresh1: float, thresh2: float):
    """ sets any y above a threshold to 1, anything below to 1
    """
    y_class = np.zeros(y.shape)
    for i, thing in enumerate(y):
        if thing > thresh2:
            y_class[i] = 1
        elif thing > thresh1:
            y_class[i] = 1
        else:
            y[i] = 0
    
    return y_class

def oversample(X_train, y_train, method = 'normal'):
    """ oversampling to deal with imbalanced classes
    """
    #normal, no smote
    if method == 'normal':
        from imblearn.over_sampling import RandomOverSampler
    
        oversample = RandomOverSampler(sampling_strategy='minority')
        
    else:
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
        
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    return X_over, y_over

def custom_metrics(preds):
    #want to know how many papers we weeded out 
    counter = 0
    for i, thing in enumerate(preds):
        if type(thing) == np.float64:
            if int(thing) == 1:
                counter += 1
        else:
            if thing[1] > thing[0]:
                counter += 1
    print('left with ' + str(counter/len(preds)) + " fraction of things")
    
def run_xgb_pipe_classify():
    """ script for fitting an xgboost model
        functions for the custom objective function are included
        dimensioality reduction done in here
    """
    
    from sklearn.model_selection import train_test_split #GridSearchCV, RandomizedSearchCV,
    import xgboost as xgb
    from sklearn.decomposition import TruncatedSVD
    from typing import Tuple
    import numpy as np
    from sklearn.metrics import classification_report,confusion_matrix
    ###########################
    # example code for custom objective function
    def gradient_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        ''' plugged  d/d(predt) abs(y-predt)/(np.minimum(predt, y)+1) into wolfram'''
        y = dtrain.get_label() + 1
        #return abs(y-predt)/(np.minimum(predt, y)+1))
        grad = np.zeros(y.shape)
        for i, val in enumerate(y):
            if predt[i] > y[i]:
                grad[i] = 1/(y[i] + 1)
            else:
                grad[i] = -(y[i] + 1)/(predt[i]**2 + 1)
        #y[y < 0] = 10**10
        #grad = abs((((y-predt)**2)**0.5)/(np.minimum(predt, y) + 1))
       
        return grad
        #return (np.log1p(y) - np.log1p(predt)**2)/ (np.log1p(y) - 1)
   
    def hessian_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''

        return np.ones(predt.shape)
   
    def custom_obj(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        grad = gradient_c(predt, dtrain)
        hess = hessian_c(predt, dtrain)
        return grad, hess
    
    def cust_eval(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        y = dtrain.get_label()
        #predt[predt < -1] = -1 + 1e-6
        elements = np.zeros(y.shape)
        for i, pred_val in enumerate(predt):
            elements[i] = abs(y[i] -pred_val)/min(pred_val, y[i])
        return 'cust_eval', elements.mean()
    
    ################### start of script
    from xgboost import XGBClassifier
    X, y, keys, idx = fetch_data()
    #y = ravel(y)
    #y = y.reshape(-1,1)
    # X_train, X_test, y_train, y_test = train_test_split(\
    #                 X, y, test_size=0.2, random_state=0)
        

    
    #y = y + 3

    X_train, X_test, y_train, y_test, keys_train, keys_test = manTTS(keys, X, y, idx)
    w = (y_train+1)**0.2
    y_train = set_classes(y_train, 10, 10)
    y_test = set_classes(y_test, 10, 10)
    #X_train, X_test, y_weight, y_weight_test, keys_train, keys_test = manTTS(keys, X, y_weight, idx)
    # train and fit the svd transformer
    svd = TruncatedSVD(6000)
    
    #w = y_weight# switched to oversampling
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    #X_train, y_train = oversample(X_train, y_train, 'smote')
    
    
    print('explained variance ratio is ' + str(sum(svd.explained_variance_ratio_)))
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    
    #params = {"objective":"reg:logistic"}
    # params = {
    # 'eta': 0.3, 
    # 'max_depth': 3,
    # 'num_class': 2}
    
    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=15,
                      verbose = 2, eval_set = eval_set, sample_weight=w)
                      # try aucpr
    data_dmatrix_X_test = xgb.DMatrix(data=X_test)
    preds = model.predict(X_test)
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    preds = model.predict(X_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    
    print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    print(classification_report(y_test,preds))
    custom_metrics(preds)
    res_analysis(preds, y_test, keys_test)

    return model

def run_xgb_pipe():
    """ script for fitting an xgboost model
        functions for the custom objective function are included
        dimensioality reduction done in here
    """
    
    from sklearn.model_selection import train_test_split #GridSearchCV, RandomizedSearchCV,
    import xgboost as xgb
    from sklearn.decomposition import TruncatedSVD
    from typing import Tuple
    import numpy as np
    from sklearn.metrics import classification_report,confusion_matrix
    ###########################
    # example code for custom objective function
    def gradient_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        ''' plugged  d/d(predt) abs(y-predt)/(np.minimum(predt, y)+1) into wolfram'''
        y = dtrain.get_label() + 1
        #return abs(y-predt)/(np.minimum(predt, y)+1))
        grad = np.zeros(y.shape)
        for i, val in enumerate(y):
            if predt[i] > y[i]:
                grad[i] = 1/(y[i] + 1)
            else:
                grad[i] = -(y[i] + 1)/(predt[i]**2 + 1)
        #y[y < 0] = 10**10
        #grad = abs((((y-predt)**2)**0.5)/(np.minimum(predt, y) + 1))
       
        return grad
        #return (np.log1p(y) - np.log1p(predt)**2)/ (np.log1p(y) - 1)
   
    def hessian_c(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''

        return np.ones(predt.shape)
   
    def custom_obj(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        grad = gradient_c(predt, dtrain)
        hess = hessian_c(predt, dtrain)
        return grad, hess
    
    def cust_eval(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        y = dtrain.get_label()
        #predt[predt < -1] = -1 + 1e-6
        elements = np.zeros(y.shape)
        for i, pred_val in enumerate(predt):
            elements[i] = abs(y[i] -pred_val)/min(pred_val, y[i])
        return 'cust_eval', elements.mean()
    
    ################### start of script
    from xgboost import XGBRegressor
    X, y, keys, idx = fetch_data()
    #y = ravel(y)
    #y = y.reshape(-1,1)
    # X_train, X_test, y_train, y_test = train_test_split(\
    #                 X, y, test_size=0.2, random_state=0)
        
    y_weight = y
    
    y = y + 3
    #y = set_classes(y_weight, 10, 10)
    X_train, X_test, y_train, y_test, keys_train, keys_test = manTTS(keys, X, y, idx)
    
    #X_train, X_test, y_weight, y_weight_test, keys_train, keys_test = manTTS(keys, X, y_weight)
    # train and fit the svd transformer
    svd = TruncatedSVD(1500)
    
    #w = y_weight switched to oversampling
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    data_dmatrix_X_test = xgb.DMatrix(data=X_test)
    #X_train, y_train = oversample(X_train, y_train, 'smote')
    
    print('explained variance ratio is ' + str(sum(svd.explained_variance_ratio_)))
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    
    #params = {"objective":"reg:logistic"}
    # params = {
    # 'eta': 0.3, 
    # 'max_depth': 3,
    # 'num_class': 2}
    
    model = XGBRegressor(objective = "reg:squaredlogerror")
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=15,
                      verbose = 2, eval_metric = "rmsle", eval_set = eval_set)
                      # try aucpr
    data_dmatrix_X_test = xgb.DMatrix(data=X_test)
    preds = model.predict(X_test)
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    preds = model.predict(X_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    
    res_dct = packagePreds_v2(preds, keys_test, y_test)
    save_pickles([res_dct], ["res_dct"], "model_related_outputs")

    return model

def run_pipe():
    """ main script to run pipe with svd and predictor
    """
    from sklearn.ensemble import RandomForestRegressor
    
    X, y, keys, idx = fetch_data()
    y = y.reshape(-1,1)
    # X_train, X_test, y_train, y_test = train_test_split(\
    #                 X, y, test_size=0.2, random_state=0)
        
    X_train, X_test, y_train, y_test, keys_train, keys_test = manTTS(keys,X,y, idx)
    
    # the diimensionality reducer needs a better way to find num components
    #dim_reducer_model = DimensionalityReducer(TruncatedSVD)
    #dim_reducer_model.fit(X)
    pipe = Pipeline([
                    ('svd', TruncatedSVD(100)),
                    ('estimator', Ridge())
        ])
    
    pipe.fit(X_train, y_train)
    
    print(pipe.score(X_test, y_test))
    
    preds = pipe.predict(X_test)
    
    preds = preds.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    res = preds - y_test
    
    preds_hist = np.histogram(preds)
    hist_bins = preds_hist[1][0:-1]
    plt.plot(hist_bins, preds_hist[0])
    
    
    y_test_hist = np.histogram(y_test)
    hist_bins = preds_hist[1][0:-1]
    plt.plot(hist_bins, y_test_hist[0])
    
    res_hist = np.histogram(res)
    hist_bins = res_hist[1][0:-1]
    
    plt.plot(hist_bins, res_hist[0])
    
    res_dct = packagePreds_v2(preds, keys_test, y_test)
    save_pickles([res_dct], ["res_dct"], "model_related_outputs")
    
    
    return pipe
    
def res_analysis(preds, y_test, keys_test):
    res = preds - y_test
    
    preds_hist = np.histogram(preds)
    hist_bins = preds_hist[1][0:-1]
    plt.plot(hist_bins, preds_hist[0])
    
    y_test_hist = np.histogram(y_test)
    hist_bins = preds_hist[1][0:-1]
    plt.plot(hist_bins, y_test_hist[0])
    
    res_hist = np.histogram(res)
    hist_bins = res_hist[1][0:-1]
    
    plt.plot(hist_bins, res_hist[0])
    
    res_dct = packagePreds_v2(preds, keys_test, y_test)
    save_pickles([res_dct], ["res_dct"], "model_related_outputs")


if __name__ == '__main__':
    
    #run_script()
    run_pipe = run_xgb_pipe_classify()

    # saving resDict



    # shuffling
    #dicMainKeys, X, y = shuffle(dicMain, X, y)
    
    #X = X.tocsr(copy=True)
    
    ####### spliting up inputs

    
    ## min max scaler doesn't work
    #sparseInput = setUp4Keras(Xm_train, Xm_test, ym_train, ym_test)
    
    
    # model = trainKeras(Xm_train, Xm_test, ym_train, ym_test)
    # print(model.count_params())
    
    # model = trainKeras(Xm_train, Xm_test, ym_train, ym_test)
    # print(model.count_params())
    
    # import os
    # cwd = os.getcwd()
    # model.save(cwd)
    
    # preds = predictKeras(model,Xm_test)
    
    # predsOut = tempFlattenPreds(preds)
    
    # resDict, misplaced = packagePreds(predsOut,keys_test,ym_test,dicMain)
    
    # mean_squared_log_error(ym_test, predsOut)
    
    # from sklearn.utils import shuffle
    # ym_test_shuffle = shuffle(ym_test, random_state=1)
    
    # #mean_squared_log_error(ym_test, ym_test_shuffle)
    
    # mean_squared_error(ym_test, predsOut)
    
    # #mean_absolute_error(ym_test, predsOut)
    
    # np.median(ym_test)
    
    # import pickle
    # pickle.dump(resDict, open( "resDict_2-26_MAE.p", "wb" ) )
    
    #putResInDf(preds,cited_test,keys_test,dicMain)
    

