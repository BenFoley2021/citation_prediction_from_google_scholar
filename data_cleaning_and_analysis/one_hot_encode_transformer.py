# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:53:58 2021

5/10, fixed error, .fit needs to return self
https://github.com/dmlc/xgboost/issues/1818


rewritting this as a transformer

input is the dataframe

the multiprocessing is causing problems, haven't narrowed it down beyond that


@author: Ben Foley
"""
from get_data_from_author_crawl import *
import pandas as pd
#import spacy
import numpy as np
import time
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
#from generic_func_lib import *
from scipy.sparse import csr_matrix, hstack
#from pandarallel import pandarallel
import math
import multiprocessing
from multiprocessing import Pool
from functools import partial
import os
import pickle
from sklearn import base
from sklearn import base
from typing import Tuple

class MainTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, transformers: dict):
        # self.process_title = transformers['title_transformer']
        # self.process_authors = transformers['author_transformer']
        # self.process_journal = transformers['journal_transformer']
        # self.process_publisher = transformers['publisher_transformer']
        # self.process_pages = transformers['pages_transformer']
        # self.process_year = transformers['year_transformer']
        self.transformers_present = set()
        self.transformers = {}
        for key, val in transformers.items():
            if val:
                self.transformers_present.add(key)
                self.transformers[key] = val
    
        #self.transformer = transformers['title_transformer']
        
    def fit(self, df, y = None):

        # for transformer in self.transformers:
        #     self.transformers[transformer].fit(df)
        for transformer in self.transformers_present:
            print('started fiting ' + transformer)
            self.transformers[transformer].fit(df)
            
        return self
        
    def transform(self, df, y = None):
        count = 0
        for transformer in self.transformers_present:
            print('started transforming ' + transformer)
            if count == 0:
                sparse_df = self.transformers[transformer].transform(df)
                count += 1
            else:
                sparse_df = sparse_df.join(self.transformers[transformer].transform(df), \
                                            lsuffix='_left', rsuffix='_right')
            
        # sparse_df = self.transformer.transform(df)
        # print('transformed df')
        return sparse_df

class TitleTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.title_words = None
        self.synonyms = None
    
    def fit(self, df, y = None):
        
        
        df['titleID_list'] = str_col_to_list_par(" ", df['titleID'])
        
        #df['titleID_list'] = parallelize_on_rows(df['titleID_list'], make_lower)
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x: make_lower(x))
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x:  custom_clean_tokens(x))
        
        #title_words = general_multi_proc(get_all_cat, df['titleID_list'])
        
        self.title_words = get_all_cat(df['titleID_list'])
        
        self.synonyms = buildCustomLookup(self.title_words) # this is pretty slow
        self.title_words = remove_synonym(self.title_words, self.synonyms)
            
        df['titleID_list2'] = merge_synonym_par(self.synonyms, df['titleID_list'])
            
        title_words_df = one_hot_encode_multi(self.title_words, df['titleID_list2'])
        title_words_df = threshold_sparse_df(title_words_df, 20)
        
        self.title_words = set(title_words_df.columns)
        

        print('fit title transformer')
        
        return self

    def transform(self, df, y = None):
        # the first portion process_title function one_hot_encode_parrallel
        # needed to return the transformed data
        df['titleID_list'] = str_col_to_list_par(" ", df['titleID'])
        
        #df['titleID_list'] = parallelize_on_rows(df['titleID_list'], make_lower)
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x: make_lower(x))
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x:  custom_clean_tokens(x))

        df['titleID_list2'] = merge_synonym_par(self.synonyms, df['titleID_list'])
            
        title_words_df = one_hot_encode_multi(self.title_words, df['titleID_list2'])

        #print('transformed title words')
        return title_words_df

class AuthorTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.authors = None
        
    def fit(self, df, y = None):
        df['Authors_list'] = str_col_to_list_par(",", df['Authors'])
        df['Authors_list'] = df['Authors_list'].apply(lambda x: make_lower(x))
        self.authors = get_all_cat(df['Authors_list'])
        author_df = one_hot_encode_multi(self.authors, df['Authors_list'])
        author_df = threshold_sparse_df(author_df, 2)
        
        self.authors = set(author_df.columns)
        return self
    
    def transform(self, df, y = None):
        df['Authors_list'] = str_col_to_list_par(",", df['Authors'])
        df['Authors_list'] = df['Authors_list'].apply(lambda x: make_lower(x))
        author_df = one_hot_encode_multi(self.authors, df['Authors_list'])
        return author_df


class JournalTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.journals = None

    def fit(self, df, y = None):
        
        df['Journal'] = df['Journal'].apply(lambda x: clean_journal(x))
        
        journal_df = pd.get_dummies(df['Journal'], sparse = True)
        print('got dummies journal')
        journal_df = threshold_sparse_df(journal_df, 7)
        
        self.journals = set(journal_df.columns)
        
        return self
        
    def transform(self, df, y = None):
        
        df['Journal'] = df['Journal'].apply(lambda x: clean_journal(x))
        df['Journal'] = df['Journal'].apply(lambda x: [x])
        journal_df = one_hot_encode_multi(self.journals, df['Journal'])

        return journal_df   

class PublisherTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.publishers = None

    def fit(self, df, y = None):
        
        publisher_df = pd.get_dummies(df['publisher'], sparse = True)
        #print('got dummies journal')
        publisher_df = threshold_sparse_df(publisher_df, 5)
        self.publishers = set(publisher_df.columns)
        return self
        
    def transform(self, df, y = None):
        
        publisher_df = one_hot_encode_multi(self.publishers, df['publisher'].apply(lambda x: [x]))
        return publisher_df 

class AuthorIdTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.author_ids = None
        
    def fit(self, df, y = None):
        
        # str_col_to_list_par(" ", df['titleID'])
        
        df['Author_id_list'] = str_col_to_list_par(" ", df['author_ids'])
        
        #df['Author_id_list'] = parallelize_on_rows(df['Author_id_list'], make_lower)
        
        author_ids = get_all_cat(df['Author_id_list'])
        author_id_df = one_hot_encode_multi(author_ids, df['Author_id_list'])
        author_id_df = threshold_sparse_df(author_id_df, 5)
        
        self.author_ids = set(author_id_df.columns)
        
        return self
        
    def transform(self, df, y = None):
        
        df['Author_id_list'] = str_col_to_list_par(" ", df['author_ids'])
        author_id_df = one_hot_encode_multi(self.author_ids, df['Author_id_list'])
        return author_id_df

class YearTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        years = np.linspace(1990,2021,32).astype(int).tolist()
        years = [str(i) for i in years]
        years = set(years)
        self.years = years
    
    def fit(self, df, y = None):
        return self

    def transform(self, df, y = None):
        df['year'] = df['year'].astype(str)
        return pd.get_dummies(df['year'].apply(lambda x: clean_year(x, self.years)), \
                              sparse = True)

class PageTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df, y = None):
        return self
    def transform(self, df, y = None):
        def custom1(x):
            if x == 'cant read from scrapper out' or x == "couldnt find":
                return 1
            else: 
                return 0
            
        cats = ['pages', 'vol', 'issue']
        for cat in cats:
            df[cat] = df[cat].apply(lambda x: custom1(x))
        
        return df[cats].astype((pd.SparseDtype("int", 0)))
    
class runXGB(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, idx):
        self.model = XGBClassifier()
        self.svd = None
        self.keys = None
        self.idx = idx
        
    def fit(self, X, y):
        #X2, y2, keys, idx = fetch_data()
        self.keys = X.index.to_list()
        X = X.sparse.to_coo()
        X = X.tocsr()
        X_train, X_test, y_train, y_test, keys_train, keys_test = manTTS(self.keys, X, y, self.idx)
        y_train = set_classes(y_train, 10, 10)
        y_test = set_classes(y_test, 10, 10)
        #X_train, X_test, y_weight, y_weight_test, keys_train, keys_test = manTTS(keys, X, y_weight, idx)
        # train and fit the svd transformer
        self.svd = TruncatedSVD(4000)
        
        #w = y_weight# switched to oversampling
        X_train = self.svd.fit_transform(X_train)
        X_test = self.svd.transform(X_test)
        X_train, y_train = oversample(X_train, y_train, 'normal')
        
        
        print('explained variance ratio is ' + str(sum(self.svd.explained_variance_ratio_)))
        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

        eval_set = [(X_test, y_test)]
        self.model.fit(X_train, y_train, early_stopping_rounds=15,
                          verbose = 2, eval_set = eval_set)
                          # try aucpr
        return self

    def predict(self, X_test, y = None):
        X_test = self.svd.transform(X_test)
        data_dmatrix_X_test = xgb.DMatrix(data=X_test)
        preds = self.model.predict(X_test)
        
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        
        preds = self.model.predict(X_test)
        
        if y:
            best_preds = np.asarray([np.argmax(line) for line in preds])
            
            print("Precision = {}".format(precision_score(y, best_preds, average='macro')))
            print("Recall = {}".format(recall_score(y, best_preds, average='macro')))
            print("Accuracy = {}".format(accuracy_score(y, best_preds)))
            print(classification_report(y,preds))
            custom_metrics(preds)
            #res_analysis(preds, y_test, keys_test)
        return preds
    
    
class runXGBRegress(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, idx):
        self.model = XGBRegressor(objective = "reg:squaredlogerror")
        self.svd = None
        self.keys = None
        self.idx = idx
        
    def fit(self, X, y):
        #X2, y2, keys, idx = fetch_data()
        self.keys = X.index.to_list()
        X = X.sparse.to_coo()
        X = X.tocsr()
        y = y + 2
        X_train, X_test, y_train, y_test, keys_train, keys_test = manTTS(self.keys, X, y, self.idx)
        #X_train, X_test, y_weight, y_weight_test, keys_train, keys_test = manTTS(keys, X, y_weight, idx)
        # train and fit the svd transformer
        self.svd = TruncatedSVD(5000)
        
        #w = y_weight# switched to oversampling
        X_train = self.svd.fit_transform(X_train)
        X_test = self.svd.transform(X_test)
       # X_train, y_train = oversample(X_train, y_train, 'normal')
        
        
        print('explained variance ratio is ' + str(sum(self.svd.explained_variance_ratio_)))
        #data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

        eval_set = [(X_test, y_test)]
        self.model.fit(X_train, y_train, early_stopping_rounds=15,
                          verbose = 2, eval_set = eval_set, eval_metric = "rmsle")
                          # try aucpr
        return self

    def predict(self, X_test, y = None):
        X_test = self.svd.transform(X_test)
        #data_dmatrix_X_test = xgb.DMatrix(data=X_test)
        preds = self.model.predict(X_test)

        
        preds = self.model.predict(X_test)
        

        return preds


def gradient_c(predt: np.ndarray, dtrain) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    ''' plugged  d/d(predt) abs(y-predt)/(np.minimum(predt, y)+1) into wolfram'''
    '''' if we just do relative to actual d/d pred (pred - actual)/actual) then
        then the first order is 1/actual, 2nd is just 0
    '''
    def get_grad_1():
        if predt[i] > y[i]:
            grad[i] = 1/(y[i] + 1)
        else:
            grad[i] = -(y[i] + 1)/(predt[i]**2 + 1)
            
    def get_grad_2():
        grad[i] = -((predt[i]/y[i] - 1)**0.5)/y[i]
        
        
    y = dtrain
    #return abs(y-predt)/(np.minimum(predt, y)+1))
    grad = np.zeros(predt.shape)
    for i, val in enumerate(predt):
        get_grad_2()
    #y[y < 0] = 10**10
    #grad = abs((((y-predt)**2)**0.5)/(np.minimum(predt, y) + 1))
   
    return grad
    #return (np.log1p(y) - np.log1p(predt)**2)/ (np.log1p(y) - 1)
   
def hessian_c(predt: np.ndarray, dtrain) -> np.ndarray:
    '''Compute the hessian for squared log error.'''

    return np.ones(predt.shape)
   
def custom_obj(predt: np.ndarray,
                dtrain) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    grad = gradient_c(predt, dtrain)
    hess = hessian_c(predt, dtrain)
    return grad, hess


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

def manTTS(keys,X,y, ind): 

    Xm_train = X[0:ind,:]
    Xm_test = X[ind:-1,:]
    
    ym_train = y[0:ind]
    ym_test = y[ind:-1]
    
    keys_train = keys[0:ind]
    keys_test = keys[ind:-1]
    
    
    return Xm_train, Xm_test, ym_train, ym_test, keys_train, keys_test

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




def prep_df(df: pd.DataFrame, which = None):
    
    if which == 'from_email':
            cols_to_drop = ['title_main', 'Conference', 'Source', 'book', \
                'urlID', 'citedYear']
    else:            
        cols_to_drop = ['Unnamed: 0', 'title_main', 'Conference', 'Source', 'book', \
                    'urlID', 'citedYear']
    df = df.drop(labels = cols_to_drop, axis = 1)
    
    if not which:
        df['year'] = df['year'].astype(int)
        df['date'] = df['date'].apply(lambda x: x.split(" ")[0])
        df = df[df['date'] < '2020-04-01']
        df = df.sort_values(by = ['date']) ### added this to manually select the more recent articles ad the test set
        df = df.reset_index()
        
        df['year'] = df['year'].astype(str)
    df = add_author_ids(df)
    
    cols_to_str = ['Journal', 'Authors']
    for col in cols_to_str:
        df[col] = df[col].astype(str)
    
    df = df[(df['cites_per_year'] != np.inf) & (df['cites_per_year'] != -np.inf)]

    y = df['cites_per_year'].to_numpy()

    return df, y

def getTok2(dfIn, col, col_2_write):
    """ Using spacy to get the tokens from the title
    """
    tokens = []
    #start = time.time()
    #disable = ["tagger", "parser","ner","textcat"]
    for doc in nlp.pipe(df[col].astype('unicode').values):#,disable = disable):
        #doc.text = doc.text.lower()
        #tokens.append([token.text.lower() for token in doc if not token.is_stop])
        tempList = ([token.lemma_ for token in doc])
        for i,string in enumerate(tempList):
            try:
                tempList[i] = string.lower()
            except:
                print(string)
    
        tokens.append(tempList)
    dfIn[col_2_write] = tokens
    return dfIn

def removeFromDict(removeSet,dicIn): #### removes things add hoc from the dictionary
    #my_dict.pop('key', None) https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
    for thing in removeSet:
        dicIn.pop(thing,None)

    return dicIn

def customProcTok(bowDic: dict,symbol2Rem: set) -> dict:
    """ removes tokens which have lots of number or nonsense characters (those in symbol2Rem)
    
        custom text parsing to remove some of the trash that made it through spacy tokenization
        loop through the dicBow
        if key has len 1 remove it
        if it has any of the characters in symbol2rem
        if it can be converted into a float
        more than 2 characters of the string can be converted into a float

    Parameters
    ----------
    bowDic : Dict
        The dict which keeps track of the words (or tokens) which are going into the 
        one hot encoded matrix
    symbol2Rem : Set
        Set of symbols, if a token contains these symbols, we remove it.

    Returns
    -------
    tempDic : Dict
        The updated dicBow after removeing tokens.

    """

    years = np.linspace(1990,2021,32).astype(int).tolist()
    years = [str(i) for i in years]
    years = set(years)
    
    okSet = set(['1d','2d','3d','4d'])
    okSet.update(years)
    
    tempDic = {}
    for key in bowDic:
        count = 0
        write = True  
        if key in okSet:
            pass
        else: 
            try:
                for n in key:
                    try:
                        float(n)
                        count = count +1
                    except:
                        pass
        
                    if n in symbol2Rem or count > 1:
                        write = False
                        break
                try:
                    float(key)
                    write = False
                    continue
                except:
                    pass
                if len(key) < 1:
                    write = False
                    continue    
                if len(key) < 3 and key not in okSet:
                    write = False
                    continue  
            except:
                write = False

        if write == True:
            tempDic[key] = bowDic[key]

    return tempDic 
    
def remTok(bowDic,set2Rem): ### i guess theres another ad hoc function to remove things from a dict
    """ Removes keys in the set2Rem variables from the dictionary. 
    
    Parameters:
        dowDic: dicBow
        set2Rem: Set. contains keys which are to be removed from bowDic if they
        are found there
        
    Returns:
        bowDic: The dictionary after desired keys have been removed
    
    """    
    for thing in set2Rem:
        try:
            bowDic.pop(thing,None)
        except:
            pass
    return bowDic

def saveOutput2(fListIn,NListIn,outLoc):
        ###pickles the outfile and saves it to a dir
    import pickle
    import os
    
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
            
def dropCust1(dfIn):
    ### removes the rows for which 'cited' cant be converted to an int
    def testIfInt(thing):
        
        try:
            int(thing)
            return True
        except:
            return False
    
    #
    dfIn = dfIn[~dfIn['title'].str.contains("Unexpected error: ")] #
    
    ########## this needs to be redone using apply instead of iterrows and drop
    for index, row in dfIn.iterrows():
        try:
            if testIfInt(row['cited']) == False or testIfInt(row['year']) == False:
                dfIn.drop(index, inplace=True)
        except:
            print(row)
            
    return dfIn

def buildCustomLookup(dicBow):
    # this can probably be done in 1/20 the lines with re 
    def checkSim(word1: str, word2: str) -> list: #returns list of strs
        """ if similar return true, else false
        the exact criteria is tricky
        if word 2 is longer it returns the differnce at the end as part of the list
        if word 1 is longer it returns none in that place
        
        """
        # if either word is 3 chars or shorter we're not bothering with them
        word2Long = False
        if len(word1) < 4 or len(word2) < 4:
            return [str(),str(),str()]
        #need to make the longer work word1
        if len(word2) >= len(word1):
            word2Long = True
            wordLong = word2
            wordShort = word1
        else:
            wordLong = word1
            wordShort = word2
        
        #want to get the common part and the difference. get back three strings
        base = str()
        wordLongDif = str()
        wordShortDif = str()
        count = 0
        for i, char in enumerate(wordShort): ### looping over shorter word
            if wordLong[i] == char:
                count += 1 # keeping track of number of same chars
                base +=char # if equal then we keep adding to the base
            else: # if not add to the differences
                if count < 3: # if we hit a character thats different and theres only 2 that have been the same it's not the same base word, don't need to keep looking
                    return [str(), str(), str()]
                else:
                    wordLongDif += wordLong[i]
                    wordShortDif += wordShort[i]
        
        if len(wordLong) > len(wordShort): # if word1 is longer than 2, put the rest of 1 into word1dif. this is sorta silly since there shouldn't be any duplicate words
            wordLongDif += wordLong[i+1:]
        ################## got the differences, now what?
        if word2Long == True:
            return [base,wordLongDif, word2]
        else:
            return [base,None, word2]
        
    def findLemma(wordRegion: list) -> dict: #
        #lets start with the most basic implementation, compare every word and
        #which is a base of another. scales terrible but conceptually simple
        tempDict = {} #look up table to be built
        reverseTempDict = {} 
        #word has [base, ]
        okEnds = set(['s','ic','ics','ed'])
        for word1 in wordRegion:
            for word2 in wordRegion:
                compareResults = checkSim(word1,word2)
                # if wordLongDiff in okEnds then word2 points to word1. word1 must be longer
                # like if word1 is cheese and word2 is cheeses
                # the trick is dealing with cases like:
                    # ferromagnetics ferromagnetic ferromagnet
                    # if a value is the same as a key (easy to check) forward that val
                if compareResults[1] in okEnds:
                    if word1 in tempDict: # word 1 is already a key in the dict
                        # word2 needs to point to the value of word1
                        tempDict[word2] = tempDict[word1]
                        reverseTempDict = {v: k for k, v in tempDict.items()}
                        
                    if word2 in tempDict:
                        #word2 is already in tempDict.
                        # this means that word2 was previously tried as word2, since
                        # we only reach this stage once with a given word as word2
                        print("do'h, see the findLemma subFunc")
                        
                    if word2 in reverseTempDict: #
                        # if the word that points is being pointed to by something
                        # else, need to update the val of that something else 
                        # ferromagnetic is word 2, ferromagnet is word 1
                        # need to go back and update ferromagnetics
                        
                        tempWordDict[word2] = word1
                        thingToBeUpdated = reverseTempDict[word1]
                        tempWordDict[thingToBeUpdated] = word1
                        reverseTempDict = {v: k for k, v in tempDict.items()}
                    
                    tempDict[word2] = word1
                
            return tempDict
                
    ###### preparing inputs
    lookUpDict= {}
    dicBowKeys = [] 
    
    if type(dicBow) == dict:
        for key in dicBow.keys(): 
            dicBowKeys.append(key) 
            
    elif type(dicBow) == set:
        for key in dicBow: 
            dicBowKeys.append(key) 
        
    dicBowKeys.sort() 
    
    point1 = 0 # initializing pointers 1 and 2
    point2 = 1
    currentWords = [] # this list will store the base and differences for regions of similar words
    
    while point2 < len(dicBowKeys):
        moveCond = True
        currentWords.append(checkSim(dicBowKeys[point1],dicBowKeys[point2]))
        if len(currentWords[-1][0]) == 0: # see the if count < X: condition in checkSim. we are checking to see if the latest results form checkSim say that the words are totally different
            moveCond = False    
        #else:
            #print('need to fill this in, do I need to do anything here?')
        if moveCond == False and len(currentWords) < 2: # if point1 and 2 are at different words AND theres only 1 word in current words, then we have nothing to compare
            point1 += 1 #words are different, keep steping through
            point2 += 1
            currentWords = [] #reset current words
        elif moveCond == False and len(currentWords) > 2: ### this condition triggers the main part of the function, actually figuring out the base words (lemma)
            wordRegion = dicBowKeys[point1:point2+1]
            lookUpDict.update(findLemma(wordRegion))
            point1 += len(currentWords)
            point2 = point1 + 1
            currentWords = []
        else: ## if movecondition is true
            point2 += 1
        
    return lookUpDict

def buildCustomLookup2(dicBow):
    # trying to break this down and make it more readable, clearly havent done it yet
    # then need to test more throughly and make sure it works
    def checkSim(word1: str, word2: str) -> list: #returns list of strs
        """ if similar return true, else false
        the exact criteria is tricky
        if word 2 is longer it returns the differnce at the end as part of the list
        if word 1 is longer it returns none in that place
        
        """
        # if either word is 3 chars or shorter we're not bothering with them
        word2Long = False
        if len(word1) < 4 or len(word2) < 4:
            return [str(),str(),str()]
        #need to make the longer work word1
        if len(word2) >= len(word1):
            word2Long = True
            wordLong = word2
            wordShort = word1
        else:
            wordLong = word1
            wordShort = word2
        
        #want to get the common part and the difference. get back three strings
        base = str()
        wordLongDif = str()
        wordShortDif = str()
        count = 0
        for i, char in enumerate(wordShort): ### looping over shorter word
            if wordLong[i] == char:
                count += 1 # keeping track of number of same chars
                base +=char # if equal then we keep adding to the base
            else: # if not add to the differences
                if count < 3: # if we hit a character thats different and theres only 2 that have been the same it's not the same base word, don't need to keep looking
                    return [str(), str(), str()]
                else:
                    wordLongDif += wordLong[i]
                    wordShortDif += wordShort[i]
        
        if len(wordLong) > len(wordShort): # if word1 is longer than 2, put the rest of 1 into word1dif. this is sorta silly since there shouldn't be any duplicate words
            wordLongDif += wordLong[i+1:]
        ################## got the differences, now what?
        if word2Long == True:
            return [base,wordLongDif, word2]
        else:
            return [base,None, word2]
        
    def findLemma(wordRegion: list) -> dict: #
        #lets start with the most basic implementation, compare every word and
        #which is a base of another. scales terrible but conceptually simple
        tempDict = {} #look up table to be built
        reverseTempDict = {} 
        #word has [base, ]
        okEnds = set(['s','ic','ics','ed'])
        for word1 in wordRegion:
            for word2 in wordRegion:
                compareResults = checkSim(word1,word2)
                # if wordLongDiff in okEnds then word2 points to word1. word1 must be longer
                # like if word1 is cheese and word2 is cheeses
                # the trick is dealing with cases like:
                    # ferromagnetics ferromagnetic ferromagnet
                    # if a value is the same as a key (easy to check) forward that val
                if compareResults[1] in okEnds:
                    if word1 in tempDict: # word 1 is already a key in the dict
                        # word2 needs to point to the value of word1
                        tempDict[word2] = tempDict[word1]
                        reverseTempDict = {v: k for k, v in tempDict.items()}
                        
                    if word2 in tempDict:
                        #word2 is already in tempDict.
                        # this means that word2 was previously tried as word2, since
                        # we only reach this stage once with a given word as word2
                        print("do'h, see the findLemma subFunc")
                        
                    if word2 in reverseTempDict: #
                        # if the word that points is being pointed to by something
                        # else, need to update the val of that something else 
                        # ferromagnetic is word 2, ferromagnet is word 1
                        # need to go back and update ferromagnetics
                        
                        tempWordDict[word2] = word1
                        thingToBeUpdated = reverseTempDict[word1]
                        tempWordDict[thingToBeUpdated] = word1
                        reverseTempDict = {v: k for k, v in tempDict.items()}
                    
                    tempDict[word2] = word1
                
            return tempDict
                
    ###### preparing inputs
    lookUpDict= {}
    dicBowKeys = [] 
    for key in dicBow.keys(): 
        dicBowKeys.append(key) 
    dicBowKeys.sort() 
    
    point1 = 0 # initializing pointers 1 and 2
    point2 = 1
    currentWords = [] # this list will store the base and differences for regions of similar words
    
    while point2 < len(dicBowKeys):
        moveCond = True
        currentWords.append(checkSim(dicBowKeys[point1],dicBowKeys[point2]))
        if len(currentWords[-1][0]) == 0: # see the if count < X: condition in checkSim. we are checking to see if the latest results form checkSim say that the words are totally different
            moveCond = False    
        #else:
            #print('need to fill this in, do I need to do anything here?')
        if moveCond == False and len(currentWords) < 2: # if point1 and 2 are at different words AND theres only 1 word in current words, then we have nothing to compare
            point1 += 1 #words are different, keep steping through
            point2 += 1
            currentWords = [] #reset current words
        elif moveCond == False and len(currentWords) > 2: ### this condition triggers the main part of the function, actually figuring out the base words (lemma)
            wordRegion = dicBowKeys[point1:point2+1]
            lookUpDict.update(findLemma(wordRegion))
            point1 += len(currentWords)
            point2 = point1 + 1
            currentWords = []
        else: ## if movecondition is true
            point2 += 1
        
    return lookUpDict

def removeBadScrap(dfIn):
    
    def customRemBad(x):
        if x in badSet:
            return False
        else:
            return True
        
    import pickle
    badSet = pickle.load(open("badSetMade_2-25.pckl", "rb" ))
    dfIn = df[dfIn['ids'].apply(lambda x: customRemBad(x)) == True]
                     
    return dfIn

def custom_clean_tokens(tok_list: list) -> list:
    """ removes tokens which have lots of number or nonsense characters (those in symbol2Rem)
    
        custom text parsing to remove some of the trash that made it through spacy tokenization

        if key has len 1 remove it
        if it has any of the characters in symbol2rem
        if it can be converted into a float
        more than 2 characters of the string can be converted into a float

    Parameters
    ----------
    tok_list : list
            the string to be broken up 
    symbol2Rem : Set
        Set of symbols, if a token contains these symbols, we remove it.

    Returns
    -------
    tempDic : Dict
        The updated dicBow after removeing tokens.

    """
    #stop_words = inputs['stop_words']
    stop_words = set(['and', 'the', 'a', 'to', 'with', 'for', 'then', 'there', 'what',\
                      '\n  ','--',"",',',"\to","..."])
    symbol2Rem = set(['%','$','{','}',"^","/", "\\",'#','*',"'",\
                  "''", '_','(',')', '..',"+",'-',']','[']) # remove 
    
    #symbol2Rem = inputs['set_to_remove']
    years = np.linspace(1990,2021,32).astype(int).tolist()
    years = [str(i) for i in years]
    years = set(years)
    
    okSet = set(['1d','2d','3d','4d'])
    okSet.update(years)
    toks_out = []

    for key in tok_list:
        count = 0
        write = True  
        if key in okSet:
            pass
        elif key in stop_words:
            write = False
        else: 
            try:
                for n in key:
                    try:
                        float(n)
                        count = count +1
                    except:
                        pass
        
                    if n in symbol2Rem or count > 1:
                        write = False
                        break
                try:
                    float(key)
                    write = False
                    continue
                except:
                    pass
                if len(key) < 1:
                    write = False
                    continue    
                if len(key) < 3 and key not in okSet:
                    write = False
                    continue  
            except:
                write = False

        if write == True:
            toks_out.append(key)
    return toks_out 

def sparse_dummies(df, column):
    """Returns sparse OHE matrix for the column of the dataframe"""
    # a much simpler one hot encoder if I'm not worried about customization
    
    from pandas import Categorical

    categories = Categorical(df[column])
    #column_names = np.array([f"{column}_{str(i)}" for i in range(len(categories.categories))])
    column_names = np.array([str(cat) for cat in categories.categories])

    N = len(categories)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N,))
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

def get_all_cat(df_col: pd.Series):
    """ Converts a df columns column where each row contains a list of tokens,
        into a set 

    Parameters
    ----------
    df_col : pd.Series
        The column of a data frame containing strs to be split apart, and the
        the parts converted to a set.
    splitter : str
        The character to split each item in series by.

    Returns
    -------
    all_authors : set
        A set of all unique items in the columns

    """
    
    all_cats_set = set()
    
    #all_cat_list = [item for sublist in regular_list for item in df_col.to_list()]
    all_cat_list = [item for sublist in df_col.to_list() for item in sublist]

    for cat in all_cat_list:
        all_cats_set.add(cat)
        
    return all_cats_set

def one_hot_encode(df_col, cats_to_use: set):
    # this doesn't work right now
    # one hot encodes the column  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    from sklearn.preprocessing import LabelBinarizer
    if len(cats_to_use) != 1:
        lb = LabelBinarizer(classes = list(cats_to_use))
    else: lb = LabelBinarizer()

    output = pd.DataFrame.sparse.from_spmatrix(
                lb.fit_transform(df_col),
                index=df_col.index,
                columns=lb.classes_)
    
    return output

def one_hot_encode_multi(cats_to_use: set, df_col):
    """ df(col) -> sparse df
            # this works but its also really slow
         #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

         # running this in parallel makes it much slower
    Parameters
    ----------
    df : pd.DataFrame
        DF containing all the relevant data.
    col : Str
        The column containg a list of things to be one hot encoded.
    cats_to_use : set
        The catagories for one hot encoding.

    Returns
    -------
    output : sparse pd.DataFrame
        The one hot encoded column from df. 

    """
    
    from sklearn.preprocessing import MultiLabelBinarizer 
    mlb = MultiLabelBinarizer(sparse_output=True, classes = list(cats_to_use))

    output = pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df_col),
                index=df_col.index,
                columns=mlb.classes_)
    
    return output


def str_col_to_list(x, splitter): # meant to be used with apply or lambda function

    return [y for y in x.split(splitter)]
    
def merge_synonym_par(synonyms: dict, df_col: pd.Series):
    
    return df_col.apply(lambda x: merge_synonym(x, synonyms))

def merge_synonym(x: list, synonyms: dict):
    #meant to be called by a lambda function
    
    for i, thing in enumerate(x):
        if thing in synonyms:
            x[i] = synonyms[thing]
            
    return x

def remove_synonym(setIn: set, synonyms: dict):
    # removes anything from the set which was found to be a synonym
    temp_set = set()
    for thing in setIn:
        if thing in synonyms:
            temp_set.add(thing)

    for thing2 in temp_set:
        setIn.remove(thing2)

    return setIn

def make_lower(x):
    return [y.lower() for y in x]

def threshold_sparse_df(dfIn, threshold):
    #need to drop columns with less then threshold counts
    return dfIn.drop(dfIn.columns[dfIn.apply(lambda col: col.sum() < threshold)], axis=1)
                          
#############3
def parallelize(data, func, num_of_processes=4):
    """ func is the partial function
    """
    data_split = np.array_split(data, multiprocessing.cpu_count())
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def run_on_subset_mod(func, data_subset):
    return data_subset.apply(func)

def parallelize_on_rows(data, func, num_of_processes=4):
    return parallelize(data, partial(run_on_subset_mod, func), num_of_processes)
#################
def str_col_to_list_par(splitter, df_col): # meant to be used with apply or lambda function
    def custom(x, splitter):
        return [y for y in x.split(splitter)]
    
    return df_col.apply(lambda x: custom(x, splitter))
#### 
def general_multi_proc(func, interable, *args):
    """ func is the function we want to run, interable is the thing (usually
        a df or array) that we want to split up and run the function on in
        parrallel, *args are additional arguments that need to be passed to 
        func.
    
    Parameters
    ----------
    func : Function
        The function to be run, takes iterable and *args as arguments.
    interable : Anything which can be split up with np.array_split
        The thing to be split up and run in parallel.
    *args : arguments needed by func
        DESCRIPTION.
    Returns
    -------
    None.

    """
    
    def recombine(output: list):
        """ process the output to desired format
            e.g. list of series -> 1 series
            list of dfs -> 1 df
            list of sets combined into 1 set

        Parameters
        ----------
        output : List
            list of the outputs from parrallel compution to be recombined.

        Returns
        -------
        recombined_output: iteritable, variable data types
            The output list recombined (eg pd.concat) to a single variable.
            The data types currently supported are in the if strucutre. If none 
            of these, the output is the input list. 

        """
        if type(output[0]) == pd.DataFrame or type(output[0]) == pd.Series:
            recombined_output = pd.concat(output)
            
        elif type(output[0]) == set:
            recombined_output = set.union(*output)
            
        else: recombined_output = output
        
        return recombined_output
    
    
    #print(*args)
    fp = partial(func, *args)
    data_split = np.array_split(interable, multiprocessing.cpu_count())
    process_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    output = process_pool.map(fp, data_split) 
    
    process_pool.close()
    process_pool.join()
    
    return recombine(output)



def get_dummies_for_par(cats_to_use, df):
    journal_df = pd.get_dummies(df['Journal'], sparse = True)
    return journal_df

def clean_journal(x):
    """ removes all numeric characters from the journal string
        There are a lot of 17th annual..... or arxiv:73648742
    """
    y = str()
    for char in x.strip().lower():
        if char.isdigit() == False:
            y += char
    return y

def clean_year(x, years):
    if x.isdigit():
        x = int(x)
        if x in years:
            return str(x)
        else:
            return 'old'
    else:
        return 'not number'

def process_cols(df, test_date, svd = False):
    # converts each col to one hot encoded, calls functions to get
    # all the cats, clean cats, and finally uses the set of cats for each
    # col to get sparse mats. 
    def process_title(df):
        # path = os.getcwd()
        # path_file = path + "\\other data to be loaded\\" + 'stop_words.txt'
        # stop_words = set()
        # with open(path_file, 'rb') as f:
        #     for line in f:
        #         stop_words.add(str(line))
        # stop_words = set(['the', 'a', 'to', 'with', 'for', 'then', 'there', 'what'])
        # set_to_remove = set(['!','%','&','[',']','^'])

        #df['titleID_list'] = general_multi_proc(str_col_to_list_par, df['titleID'], " ")
        
        df['titleID_list'] = str_col_to_list_par(" ", df['titleID'])
        
        #df['titleID_list'] = parallelize_on_rows(df['titleID_list'], make_lower)
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x: make_lower(x))
        
        df['titleID_list'] = df['titleID_list'].apply(lambda x:  custom_clean_tokens(x))
        
        #title_words = general_multi_proc(get_all_cat, df['titleID_list'])
        
        title_words = get_all_cat(df['titleID_list'])
        
        synonyms = buildCustomLookup(title_words) # this is pretty slow
        title_words = remove_synonym(title_words, synonyms)
        
        df['titleID_list2'] = general_multi_proc(merge_synonym_par, df['titleID_list'], \
                                             synonyms)
            
        df['titleID_list2'] = merge_synonym_par(synonyms, df['titleID_list'])
            
        title_words_df = one_hot_encode_multi(title_words, df['titleID_list2'])
        
        title_words_df = threshold_sparse_df(title_words_df, 20)
        return title_words_df
    
    def process_authors(df):
    
        df['Authors_list'] = general_multi_proc(str_col_to_list_par, df['Authors'], ",")
        df['Authors_list'] = parallelize_on_rows(df['Authors_list'], make_lower)
        authors = general_multi_proc(get_all_cat, df['Authors_list'])
        author_df = one_hot_encode_multi(authors, df['Authors_list'])
        author_df = threshold_sparse_df(author_df, 2)
    
        return author_df
    
    def process_author_ids(df):
    
        df['Author_id_list'] = general_multi_proc(str_col_to_list_par, df['author_ids'], " ")
        df['Author_id_list'] = parallelize_on_rows(df['Author_id_list'], make_lower)
        authors = general_multi_proc(get_all_cat, df['Author_id_list'])
        author_id_df = one_hot_encode_multi(authors, df['Author_id_list'])
        author_id_df = threshold_sparse_df(author_id_df, 5)
    
        return author_id_df
    
    def process_journals(df):
        temp = set(['a'])
        #journal_df = one_hot_encode(df['Journal'], temp) doesn't work
        df['Journal'] = df['Journal'].apply(lambda x: clean_journal(x))
        journal_df = pd.get_dummies(df['Journal'], sparse = True)
        print('got dummies journal')
        journal_df = threshold_sparse_df(journal_df, 7)
        return journal_df

    def process_year(df):
        years = np.linspace(1990,2021,32).astype(int).tolist()
        years = [str(i) for i in years]
        years = set(years)
        df['year'] = df['year'].astype(str)
        return pd.get_dummies(df['year'].apply(lambda x: clean_year(x, years)), \
                              sparse = True)
        
    def process_publisher(df):
        publisher_df = pd.get_dummies(df['publisher'], sparse = True)
        #print('got dummies journal')
        publisher_df = threshold_sparse_df(publisher_df, 5)
        return publisher_df
    
    def process_pages(df):
        # if the page numbers are anything other than "can't read from scrapper"
        # return 1, else 0
        def custom1(x):
            if x == 'cant read from scrapper out' or x == "couldnt find":
                return 1
            else: 
                return 0
            
        cats = ['pages', 'vol', 'issue']
        for cat in cats:
            df[cat] = df[cat].apply(lambda x: custom1(x))
        
        return df[cats].astype((pd.SparseDtype("int", 0)))
        
    
    
    def do_svd(sparse_df, n_comp):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_comp)
        
        sparse_mat = sparse_df.sparse.to_coo()
        X_svd = svd.fit_transform(sparse_mat)
        
        return X_svd

    def sort_by_date(df):
        """ sort the outpput by date and return the sparse, print which row the "test data"
            starts on"
        """
        
        idx_start = df.date.eq(test_date).idxmax()
        #idx_end = df.date.eq('2020-04-01').idxmax()
        return idx_start

    def save_outputs(svd):
        
        labels = df['cites_per_year'].to_numpy()
        paper_ids = sparse_df.index.to_list()
        outLoc = 'one_hot_encoded_data_v2'
        
        if svd == False:
            col_names = sparse_df.columns.to_list()
            bow_mat_X = sparse_df.sparse.to_coo()
            bow_mat_X = bow_mat_X.tocsc()
            idx = sort_by_date(df)
            fList = [col_names, labels, paper_ids, idx]
            nList = ["col_names", "labels", "paper_ids", "idx"]
            path = outLoc + '//' + 'bow_mat_X.npz'  
            scipy.sparse.save_npz(path, bow_mat_X)
        
        else:
            fList = [X_svd, labels, paper_ids]
            nList = ["X_svd", "labels", "paper_ids"]
        
        saveOutput2(fList,nList,outLoc) 
        return None

    sparse_df = process_title(df).join(process_journals(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_authors(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_year(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_author_ids(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_publisher(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_pages(df), lsuffix='_left', rsuffix='_right')
    # converting to sparse 

    if svd == True:
        X_svd = do_svd(sparse_df, 100)

    save_outputs(svd)

    return None

def add_author_ids(df):
    """ loads pickle with latest author id, adds these ids to the df
        as a list in a column
    """
    
    def attach_authorIds(x):
        if x['titleID'] in authors_by_paper:
            return authors_by_paper[x['titleID']]
        else:
            return 'None'
        
    path = os.getcwd()
    path += "//data_subset//" + "authors_for_paper.pickle"
    
    with open(path, 'rb') as f:
        authors_by_paper = pickle.load(f)
    
    df['author_ids'] = df.apply(attach_authorIds, axis = 1)
    
    return df
    #trying to use a series is really slow, idk why
    # index = []
    # vals = []
    # for key, value in authors_by_paper.items():
    #     index.append(key)
    #     vals.append(vals)
    #s_authors = pd.Series(data = vals, index = index)
    #df = df.join(s_authors)
    
def is_float(x):
    

    if isinstance(x, float):
        return True
    else:
        return False
    
def run_script():
    """runs everything in if __name__ == "__main__"
        want to be able to call this whole process from the build model predict script


    Returns
    -------
    None.

    """
    
    file_name = "df_select_06_5-4.csv"
    cwd = os.getcwd()
    path = cwd + "\\data_subset\\"
    df = pd.read_csv(path + file_name)
    
    cols_to_drop = ['Unnamed: 0', 'title_main', 'Conference', 'Source', 'book', \
                'urlID', 'citedYear']
    df = df.drop(labels = cols_to_drop, axis = 1)
    
    df['year'] = df['year'].astype(int)
    df = df[df['date'] < '2020-06-01']
    df = df.sort_values(by = ['date']) ### added this to manually select the more recent articles ad the test set
    df = df.reset_index()
    
    df['year'] = df['year'].astype(str)
    df = add_author_ids(df)
    
    cols_to_str = ['Journal', 'Authors']
    for col in cols_to_str:
        df[col] = df[col].astype(str)
    #df = df.sample(frac = 1) # commented this out 4-29
    
    #df = df.iloc[0:1000]
    #df = df.iloc[0:1000]
    
    #df = pd.read_csv("cleaned_data//df_for_results__28-03-2021 10_02_35.csv")
    #df = df.iloc[0:10000]
    
    ### droping some cols we aren't using in the hopes that this runs faster
    
    #df.reset_index()
    #categorical_features = ['year','Authors','Journal']
    test_date = '2019-05-01'
    df = df[(df['cites_per_year'] != np.inf) & (df['cites_per_year'] != -np.inf)]
    process_cols(df, test_date)
    
def load_from_email_predict(pipe):
    cols_to_return = ['titleID',"title_main", 'Authors', 'Journal', "Conference", "Source", "book", \
                   'publisher', "vol", "issue", "pages", 'cited_num', \
               'cites_per_year', 'date', 'scrap_auth_id', "citedYear", "urlID", 'abstract']
    df_latest =  run_script_for_loading_from_email_bot()
    #df_latest = clean_df(df_latest, cols_to_return)
    df_latest, y = prep_df(df_latest, which = 'from_email')
    df_latest.to_csv('latest_df_to_ML.csv')
    return pipe.predict(df_latest.copy())

    
def get_idx_for_year(df, test_date):
    idx_start = df.date.eq(test_date).idxmax()
        #idx_end = df.date.eq('2020-04-01').idxmax()
    return idx_start
    
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    from sklearn.decomposition import TruncatedSVD
    from typing import Tuple
    import numpy as np
    from sklearn.metrics import classification_report,confusion_matrix
    
    import xgboost as xgb
    from xgboost import XGBClassifier
    from xgboost import XGBRegressor
    #run_script()
    cwd = os.getcwd()
    path = cwd + "\\data_subset\\"
    file_name = "df_select_06_5-4.csv"
    df = pd.read_csv(path + file_name)
    
    df, y = prep_df(df)

    idx = get_idx_for_year(df, '2019-07-01')
    
    transformers_to_use = {'TitleTransformer': TitleTransformer(),\
                           'JournalTransformer': JournalTransformer(),
                           'AuthorIdTransformer': AuthorIdTransformer(),
                           'YearTransformer': YearTransformer(),
                           'PublisherTransformer': PublisherTransformer(),
                           'PageTransformer': PageTransformer(),
                           'AuthorTransformer': AuthorTransformer()}
        
    transformer = MainTransformer(transformers_to_use)
    
    pipe = Pipeline([
        ('main_transformer', transformer),
        ('predictor', runXGBRegress(idx))
    ])

    pipe.fit(X = df.copy(), y = y)
    
    df_test = df.iloc[idx:]
    y_test = y[idx:]
    test_out = pipe.predict(df_test.copy())
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, accuracy_score


else:
    print('load the data, pipe, and call predict')
    # y_test = set_classes(y_test, 10, 10)

    # print("Precision = {}".format(precision_score(y_test, test_out, average='macro')))
    # print("Recall = {}".format(recall_score(y_test, test_out, average='macro')))
    # print("Accuracy = {}".format(accuracy_score(y_test, test_out)))
    # print(classification_report(y_test, test_out))
    # custom_metrics(test_out)
