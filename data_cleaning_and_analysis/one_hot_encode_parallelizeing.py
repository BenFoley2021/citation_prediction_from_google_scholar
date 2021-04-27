# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:53:58 2021

trying to write a general function which makes it easy to parrallelize all the
df operations

currently on TypeError: string indices must be integers

multilabelbinarizer with dask
https://stackoverflow.com/questions/55487880/achieve-affect-of-sklearns-multilabelbinarizer-with-dask-dataframe-map-partitio

The parrallizing works ok for apply. its actually slower for the one hot encoding using the 
multilabel binarizer

notes on size of things
np array of float 32
.shape = (38525, 50)
.size = 1926250
.nbytes = 15410000

each thing takes up 8 bytes
so if have 4*10^6 * 

@author: Ben Foley
"""

import pandas as pd
#import spacy
import numpy as np
import time
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
from generic_func_lib import *
from scipy.sparse import csr_matrix, hstack
#from pandarallel import pandarallel
import math
import multiprocessing
from multiprocessing import Pool
from functools import partial
import os
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

def test_func1(x):
    return x**x

def get_dummies_for_par(cats_to_use, df):
    journal_df = pd.get_dummies(df['Journal'], sparse = True)
    return journal_df

def clean_journal(x):
    """ removes all numeric characters from the journal string
        There are a lot of 17th annual..... or arxiv:73648742
    """
    y = str()
    for char in x:
        if char.isdigit() == False:
            y += char
    return y

def process_cols(df, svd = False):
    # converts each col to one hot encoded, calls functions to get
    # all the cats, clean cats, and finally uses the set of cats for each
    # col to get sparse mats. 
    def process_title(df):
        df['titleID_list'] = general_multi_proc(str_col_to_list_par, df['titleID'], " ")
        df['titleID_list'] = parallelize_on_rows(df['titleID_list'], make_lower)
        title_words = general_multi_proc(get_all_cat, df['titleID_list'])
        synonyms = buildCustomLookup(title_words) # this is pretty slow
        title_words = remove_synonym(title_words, synonyms)
        df['titleID_list2'] = general_multi_proc(merge_synonym_par, df['titleID_list'], \
                                             synonyms)
        title_words_df = one_hot_encode_multi(title_words, df['titleID_list2'])
        title_words_df = threshold_sparse_df(title_words_df, 10)
        return title_words_df
    
    def process_authors(df):
    
        df['Authors_list'] = general_multi_proc(str_col_to_list_par, df['Authors'], ",")
        df['Authors_list'] = parallelize_on_rows(df['Authors_list'], make_lower)
        authors = general_multi_proc(get_all_cat, df['Authors_list'])
        author_df = one_hot_encode_multi(authors, df['Authors_list'])
        author_df = threshold_sparse_df(author_df, 5)
    
        return author_df
    
    def process_journals(df):
        temp = set(['a'])
        #journal_df = one_hot_encode(df['Journal'], temp) doesn't work
        journal_df = pd.get_dummies(df['Journal'], sparse = True)
        print('got dummies journal')
        journal_df = threshold_sparse_df(journal_df, 5)
        return journal_df

    def do_svd(sparse_df):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(100)
        
        sparse_mat = sparse_df.sparse.to_coo()
        X_svd = svd.fit_transform(sparse_mat)
        
        return X_svd

    def save_outputs(svd):
        
        labels = df['cites_per_year'].to_numpy()
        paper_ids = sparse_df.index.to_list()
        outLoc = 'one_hot_encoded_data_v2'
        
        if svd == False:
            col_names = sparse_df.columns.to_list()
            bow_mat_X = sparse_df.sparse.to_coo()
            bow_mat_X = bow_mat_X.tocsc()
            fList = [col_names, labels, paper_ids]
            nList = ["col_names", "labels", "paper_ids"]
            path = outLoc + '//' + 'bow_mat_X.npz'  
            scipy.sparse.save_npz(path, bow_mat_X)
        
        else:
            fList = [X_svd, labels, paper_ids]
            nList = ["X_svd", "labels", "paper_ids"]
        
        saveOutput2(fList,nList,outLoc) 
        return None

    sparse_df = process_title(df).join(process_journals(df), lsuffix='_left', rsuffix='_right')
    sparse_df = sparse_df.join(process_authors(df), lsuffix='_left', rsuffix='_right')
    # converting to sparse 

    if svd == True:
        X_svd = do_svd(sparse_df)

    save_outputs(svd)

    return None
    
def run_script():
    """runs everything in if __name__ == "__main__"
        want to be able to call this whole process from the build model predict script
    

    Returns
    -------
    None.

    """
    
    file_name = "batteries__15-04-2021 22_33_32.csv"
    cwd = os.getcwd()
    path = cwd + "\\data_subsets\\"
    df = pd.read_csv(path + file_name)
    #df = df.iloc[0:1000]
    #df = df.iloc[0:1000]
    
    #df = pd.read_csv("cleaned_data//df_for_results__28-03-2021 10_02_35.csv")
    #df = df.iloc[0:10000]
    
    ### droping some cols we aren't using in the hopes that this runs faster
    cols_to_drop = ['Unnamed: 0', 'title_main', 'Conference', 'Source', 'book', \
                    'publisher', 'vol', 'issue', 'pages', 'urlID', 'citedYear']
    
    df = df.drop(labels = cols_to_drop, axis = 1)
    
    df.reset_index()
    categorical_features = ['year','Authors','Journal']
    
    df = df[(df['cites_per_year'] != np.inf) & (df['cites_per_year'] != -np.inf)]
    
    process_cols(df)
    
    
    
    

if __name__ == "__main__":
    ############ load the data, keeping the ids col as a str
    #df_list = load_all_dfs("cleaned_data")
    
    # # for testing we want this to run fast
    #df_list = df_list[0:1]
    
    #df = cat_dfs(df_list)
    run_script()

