# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:53:58 2021

want to try other one hot encoding strategies to see if they are faster
likely some packages which are better than my solution

https://www.kaggle.com/alexandrnikitin/efficient-xgboost-on-sparse-matrices

possible resources for dealing with names https://towardsdatascience.com/python-tutorial-fuzzy-name-matching-algorithms-7a6f43322cc5
https://regex101.com/r/tR7kV2/1
https://stackoverflow.com/questions/31248856/regex-full-name-to-abbreviated-name

converts the input df into one hot encoded outputs

need to rewrite the encoder using built in (and faster functionality). If deal with each column
indiviudally instead of bundling them all together, can write smaller more specfic functions which
make best use of built in functionality of pandas. 
    
    spacy is really slow, and it's not that useful because theres so many science words not in its
    database. get rid of it and rely on the custom lookup function to do things like qubits -> qubit'
    
    expand each column in as sparse array, stack the sparse array for each row to get a sparse matrix,
    concat with exisiting sparse matrices resuling from processing other cols. expand out columns until done.
    this way columns like authors and title can be treated seperately.
    
other one hot encoders https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
can pass this one a list of expected catagories. can do custom processing on the cols first to 
considate synomnys and authors, then use this to get the sparse mat for each col
    

current progress on other one hot methods: one_hot_encode_col(df, col, cats_to_use)
    works, but is still pretty slow


custom lookup
    generally looks good, does miss some things though
        mass -> mask

@author: Ben Foley
"""

import pandas as pd
import spacy
import numpy as np
import time
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
from generic_func_lib import *
from scipy.sparse import csr_matrix, hstack
        
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

def updateBowDict(tokens,dicBow):
    ##### processing words
    ##### update dict with dic[word] = 0
    #tokens = processTxt(textIn)
    for tok in tokens:
        if tok not in dicBow:
            dicBow[tok] = 0
            
        elif tok in dicBow:
            dicBow[tok] = dicBow[tok] + 1
            #print(dicBow[tok])
    return dicBow
    
def setUpBow2(dfIn, extraWords):  #### adding the extra words to the list after the fact
    # transfering data in the df to dictionary formatt, getting tokens or lemma of the title
    # adding other info like year published and category to the tokens
    dicBow = {} ##### the dictionary storing the words for the BOW
    dicMain = {} #### dictionary with the row (or paper) id as the key, times cited, tokens, and number of tokens to be actually used in the model as values
    print('starting on getting toks')
    dfIn = getTok2(dfIn, 'titleID', 'tokens')  ##### getting tokens (or lemma) and adding the list as a new col in the df. # this is fairly slow
    print('got tokens')

    def get_toks_from_non_title_cols(x):
        #### adds to the list of toks in the tokens column
        ### extra info not in the title is added so it can be used in the bag of words
        tempList= []
        for col in extraWords:
            if col == 'Authors':
                tempList = tempList + x[col].split(',')
            else:
                if type([x[col]]) == str:
                    x[col] = x[col].lower()
                tempList = tempList +  [x[col]]
                
        return x['tokens'] +  tempList

    def iter_df_make_bow_and_main(dfIn, dicBow, dicMain):
        for i,row in dfIn.iterrows(): ###### looping through the dataframe (now with tokens/lemma) and building the dictionaries
            if i%2000 == 0:
                print(i)
            
            dicBow = updateBowDict(row['tokens'],dicBow)  #### update the bag of words dictionary 
            ##### add the extraWords in here
        
            ### updating the dict with row id, tokens, and cited by info
            dicMain[str(i)] = [row['cites_per_year'],row['tokens'],0]#### the zero will be updated with the word count for that ids later

        return dicBow, dicMain

    dfIn['tokens'] = dfIn.apply(get_toks_from_non_title_cols, axis=1)
    dicBow, dicMain = iter_df_make_bow_and_main(dfIn, dicBow, dicMain)

    return dicBow, dicMain                                 

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

def popBow(dicBow: dict, dicMain: dict, intermediate_dict: dict): ####### this is the main loop which contructs the one hot encoded matrix for input to
    """ constructs the one hot endoded input
    """

    def make_word_to_vec_and_toks_per_paper():
        """ making a dictionary which converts tokens to the position in the 
            vector. Also making a copy of dicBow which keeps track of which papers
            are using which tokens. 
        """
        dicWordPaper = dicBow.copy()  #### the keys are the words, values list of paper ids
        dicWord2Vec = {} ### keeps track of which word corresponds to which col in bowVec
        for i,key in enumerate(dicBow): 
            dicWord2Vec[key] = i #putting the location of each word into the dict
            dicWordPaper[key] = [] ### changing the value to emptylists
        
        return dicWord2Vec, dicWordPaper

    def loop_tokens(key, dicMain, dicWordPaper, wordVec):
        """ loops through the tokens contained in each item of dicMain
        """
        tempSet = set()
        dicBowTemp = dicBow.copy() ### getting copy of the bow and setting all keys to zero, will use this to count words for the current row
        dicBowTemp = dict.fromkeys(dicBowTemp, 0) #setting all keys to 0 https://stackoverflow.com/questions/13712229/simultaneously-replacing-all-values-of-a-dictionary-to-zero-python
            
        for tok in dicMain[key][1]: ### looping through the tokens stored in dicMain
            #### need to check if the tok is in the intermediate dictionary
            # if this token got merged with another one (e.g. batteries -> battery)
            # then we change the tok to its base (lemma, root, whatever), as thats 
            # what will be in dicBow
            if tok in intermediate_dict: # intermediate_dict is the hashmap made by build custom lookup table
                tok = intermediate_dict[tok]
        
            if tok in dicBowTemp:
                dicBowTemp[tok] = dicBowTemp[tok] + 1 #### #times the word is present
                dicWordPaper[tok].append(key)  ###### adding the id of the paper which has that word
                tempSet.add(tok) ###adding the token to a set which will be used to update the sparse array later
                dicMain[key][2] = dicMain[key][2] + 1 #### keeping track of how many words from this ids are still going to be used

        return dicMain, dicBowTemp, dicWordPaper, tempSet

    def update_bowVec(tempSet, dicWord2Vec, bowVec, dicBowTemp, indi):
        
        for item in tempSet:  ### looping through the tokens encountered for this row (paper) and updating the one hot encoding 
            indj = dicWord2Vec[item] ## getting the index (or column in the sparse matrix)
            try: #### at one point I was having trouble with indexing, so that's why its a try block
                bowVec[indi,indj] = dicBowTemp[item]
                if dicBowTemp[item] < 0:
                    print('warning! negative word count in dicBowTemp')
                if bowVec[indi,indj] < 0:
                    print('waring! negative word count in bowVec')
            except:
                print(str(indi) + ' ' + str(indj))

        return bowVec

    def loop_dicMain(dicMain, bowVec, dicWordPaper):
        """ loops through the main dictionary
        """
        
        indi = -1
        for key in dicMain: # loop used to populate bowVec, as well update dicMain with the number of factors for each row (paper) that are being used in the model
            indi = indi +1
            label.append(dicMain[key][0]) # label is the number of citations, what I will try to predict later
            dicMain, dicBowTemp, dicWordPaper, tempSet = loop_tokens(key, dicMain, dicWordPaper, wordVec)
            bowVec = update_bowVec(tempSet, dicWord2Vec, bowVec, dicBowTemp, indi)

        return label, bowVec, dicWord2Vec, dicWordPaper

    # main function script, declaring variables 
    label = [] #### label is the number of citations, what I will try to predict later
    bowVec = lil_matrix((len(dicMain), len(dicBow)), dtype=np.int8) ## this will hold the one hot encoded bow    
    wordVec = np.zeros(len(dicBow))  ### is the same length as the num of words in the bow
    
    dicWord2Vec, dicWordPaper = make_word_to_vec_and_toks_per_paper()
    
    label, bowVec, dicWord2Vec, dicWordPaper = loop_dicMain(dicMain, bowVec, dicWordPaper)

    return label, bowVec, dicWord2Vec, dicWordPaper  #### dicWordPaper is obsolete in this version

def analyzeFreq(dicIn):   
    ## this is also obsolete now
    
    tempDic = dicIn.copy()
    tempList= []
    #empty df with cols
    #https://www.kite.com/python/answers/how-to-create-an-empty-dataframe-with-column-names-in-python#:~:text=Use%20pandas.,an%20empty%20DataFrame%20with%20column_names%20.
    
    column_names = ["word", "count", "ids"]
    tempdf = pd.DataFrame(columns = column_names)
    
    
    for key in dicIn:
        tempDic[key] = len(dicIn[key])
        tempList.append(len(dicIn[key]))
        
        
        
    return tempDic,tempList

def histList(listIn): #makes histogram of list
    import numpy as np
    # import random
    from matplotlib import pyplot as plt
    
    # data = np.random.normal(0, 20, 1000) 
    
    # fixed bin size
    bins = np.arange(1, 100, 2) # fixed bin size
    
    plt.xlim([min(listIn)-5, max(listIn)+5])
    
    plt.hist(listIn, bins=bins, alpha=0.5)
    plt.title('Random Gaussian data (fixed bin size)')
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('count')
    
    plt.show()
    
def trimBow2(bowDic,thresh):  ### removes words from bag of words if they present in less than thresh rows (papers)
    tempDic = {}
    for key in bowDic:
        if bowDic[key] > thresh:
            #bowDic.pop(key,None)
            tempDic[key] = bowDic[key]
            
    return tempDic

def trimBow3(bowDic,authorSet,journalSet,thresh,threshAuthor,threshJournal):  ### removes words from bag of words if they present in less than thresh rows (papers)
    """ need to apply different thresholds to different groups
        The theshold changes depending on which group the key belongs to
        
    """
    tempDic = {}
    for key in bowDic:
        tempThresh = thresh
        if key in authorSet:
            tempThresh = threshAuthor
        elif key in journalSet:
            tempThresh = threshJournal
            
        if bowDic[key] > tempThresh:
            #bowDic.pop(key,None)
            tempDic[key] = bowDic[key]
            
    return tempDic

def wordCount(curBowDic,mainDic): #### i think this is also obsolete now
    #checks to see how many of the words for each paper are in the current bow dic.
    # need to know if have eliminated all or most words from anything
    
    tempDic = mainDic.copy()
    
    for key in mainDic:
        #print(key)
        count = 0
        for tok in mainDic[key][1]:
            if tok in curBowDic:
                count = count +1
        
        
        tempDic[key] = [mainDic[key][0],mainDic[key][1],count]
            
    return tempDic

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

def prepToks(dfIn,extraWords):
    #adds the extra words to the title page
    #https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    
    def custom1(x):
        tempStr = str()
        for col in extraWords:
            if col == 'cat':
                tempList = x[col].split(' ')
                for thing in tempList:
                    tempStr = tempStr + ' ' + str(thing)
                
            elif col =='year':
                
                tempStr = tempStr + ' ' +  x[col]
        
        return x['title'] + ' ' + tempStr
    
    df['title'] = df.apply(custom1, axis=1)
    
    return df

            
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

def condenseDicBow(dicBow,customLookUpDict):
    """ converts tokens which were decided to have a root into that root.
        eg. batteries -> battery
        
    PARAMS
        dicBow: dict. The keys all tokens to be used in the one hot encoded model. 
                        The values are the number of times that token has been used
        customLookUpDict: dict. Maps tokens to their root, such as a key of "batteries"
                            with a value of "battery".
    """
    
    for key in customLookUpDict:
        if key in dicBow:
            try:
                dicBow[customLookUpDict[key]] += dicBow[key] # since we are combing the tokens, also need to combine the number of times it's present
                dicBow.pop(key) # remove the key. (eg batteries is merged with battery, then the batteries key removed)
            except:
                print('couldnt find ' + key)

    return dicBow

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

def compositionOfDicBow(dicBow,authorSet,journalSet):
    """ figuring out which source is contributing how many factors to dicBow
        right now we just the tokens form the title, the authors, and journals
    

    Parameters
    ----------
    dicBow : TYPE
        DESCRIPTION.
    authorSet : TYPE
        DESCRIPTION.
    journalSet : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    outDict = {}
    authorCount = 0
    journalCount = 0
    otherCount = 0
    
    for key in dicBow:
        if key in authorSet:
            authorCount += 1
        elif key in journalSet:
            journalCount += 1
        else:
            otherCount += 1
            
    outDict['author'] = authorCount
    outDict['journal'] = journalCount
    outDict['other'] = otherCount
    
    return outDict

def predictOne(MLinput):
    """ return 
    """
    import keras
    import os 
    
    model = keras.models.load_model(os.getcwd()) #### probably want to load the model in flask,
    # dont want to load it everytime need to predict
    
    return model.predict(MLinput)

def words2ModelInput(inputDict, dicWord2Vec):
    """ converting a list of words to input for model
    
    """
    import spacy
    nlp = spacy.load('en_core_web_sm')
    
    X = np.zeros(len(dicBow))
    
    tokens = []
    factUsed = []
    
    tokens = ([token.lemma_ for token in nlp(inputDict['title']) if not token.is_stop])
    
    for thing in inputDict['otherFactor']:
        tokens.append(thing)
    
    
    for word in tokens:
        #word = word.strip().lower()
        #print(word)
        if word in dicWord2Vec:
            X[dicWord2Vec[word]] += 1
            factUsed.append(word)
    return X, factUsed

def clean_authors(df):
    """ currently: this is more work than I realized, not sure if it's work doing
        until have more results from the intial fits. should try other less, less adhoc 
        methods of dimensionality reduction before investing more time here (note on 4/1)
    
        It's common that authors are listed with both full name and initials. would like
        to combine these.
        
        strategy: if we take the data scrapped from one author, it's not likely that
                there will be cases where authors have the same or same initials. So if 
                map initials back to the author they likely came from only with the set
                of papers for each author scrapped, should be able to expand the author
                abbreviatinos into full name if present
                
        implementation:
            define allowed transformations
                initial can be expanded into full name if
                    * + lastName = firstName + lastName
                    * + ^ + lastName = firstName + middleName + lastName
                    
                * = any from list of all first names 
                ^ = from list of all middle names

            single letter = initial, try and sub *
            single letter + "." = initial, try and sub *
            double letter = initials, try and sub * + ' ' + ^
            
            split df into groups of authors scrapped
            for each of those, get lists of authors
            
            go through each author, if the authors name has initials,
            make an entry with the generic character 
        
            if there are multiple things it could be, we want to add that group
            (name with initials and full names that it could be) to a list of authors
            which we can't resolve. 
        
    """
    


    def custom_name_clean(x):
        # break authors in list, make lower, strip, recombine into str seperated
        # by commas
        x_list = x.split(",")
        y = str()
        for name in x_list:
            name = name.lower().strip()
            y += name + ","
        
        return y
    
    def custom_replace_name(x):
        # this was meat to be the .apply or lambda for replacing the authors
        # in the actual df, haven't done it yet
        return x
    
    def map_initials_to_names(df_temp):
        # for now we are just dealing with the case where theres a first,
        # maybe middle, and last name.
        def update_dict(dictIn, key, value):
            if key in dictIn:
                dictIn[key].add(str(value))
            else:
                dictIn[key] = set()
                dictIn[key].add(str(value))
            return dictIn
        
        def make_name_sets():
            """ loops through the authors of each paper makes lists of first,
                middle, last names as well as any that seem to be abbrievations
                
                A dictionary with names with abbreiations where the abbreiations
                are converted to wildwards is returned, along with dictionaries 
                with the wildcard-ized version of each first and middle name
                
                based somewhat on the word ladder type problems
            """
            
            first_names = {}
            middle_names = {}
            last_names = {}
            all_names = {}
            name_to_generic = {}
            names_to_title = {}
            for _, row in df_temp.iterrows():
                for name in row['Authors'].split(","):
                    num_words = len(name.split(" "))
                    words = name.split(" ")
                    temp_name_to_generic = {}
                    if len(words) == 2 or len(words) == 3: 
                        all_names.update({name: 1})
                        for i, word in enumerate(words):
                            # this isn't a very good way to do this
                            # could do:
                            # for each thing try transform. return transformed thing
                                # or thing if no transform
                            # add all things back into name, if no transforms done
                            # don't put it in the generic_to_name dict.
                            # the if structure is really hard to follow
                            
                            
                            if i == 0:
                                if len(word) > 2: # its a word not an abbrievation.
                                    update_dict(first_names, (word[0] + "*"), word)
                                elif len(word) == 1:
                                    temp_name_to_generic[name] = word + "*" + " "
                                elif len(word) == 2 and word[1] == ".":
                                    temp_name_to_generic[name] = word[0] + "*" + " "

                            elif i == 1 and num_words == 3:
                                if len(word) > 2: # its a word not an abbrievation.
                                    update_dict(middle_names, word[0] + "*", word)
                                elif len(word) == 1:
                                    temp_name_to_generic[name] += word + "*" + " "
                                elif len(word) == 2 and word[1] == ".":
                                    temp_name_to_generic[name] += word[0] + "*" + " "
                                
                            elif (i == 2 and num_words == 3) or (i == 1 and num_words == 2):
                                if len(word) > 2:
                                    last_names.update({word: 1})
                                    if bool(temp_name_to_generic) == True:
                                        temp_name_to_generic[name] += word
                                    
                    name_to_generic.update(temp_name_to_generic)
                    if bool(temp_name_to_generic) == True: # if this name had abbrievations, add it so we how to look at that paper to try and resolve the abbreivation
                        names_to_title.update({name: row['titleID']})
            generic_to_name = {v: k for k, v in name_to_generic.items()}
            
            return all_names, first_names, middle_names, last_names, name_to_generic, \
                generic_to_name, names_to_title

        def get_possible_names(abrv_name):
            """ gets all possible names based on the options for the * characters
            """
            pos_first_names = None 
            pos_middle_names = None 
            pos_names = []
            if abrv_name.split(" ")[0] in first_names:
                pos_first_names = first_names[abrv_name.split(" ")[0]]
            if len(abrv_name.split(" ")) == 3 and abrv_name.split(" ")[1] \
                in middle_names:
                pos_middle_names = middle_names[abrv_name.split(" ")[1]]
    
            # now loop through all the possible combinations. can't break when find, cause
            # need to see if there's more than 1
            if pos_first_names:
                for f_name in pos_first_names:
                    if pos_middle_names:
                        for m_name in pos_middle_names:
                            pos_names.append(f_name + " " + m_name + " " + \
                                             abrv_name.split(" ")[2])
                    else:
                        pos_names.append(f_name + " " + abrv_name.split(" ")[1])
                    
            return pos_names

        all_names, first_names, middle_names, last_names, name_to_generic, generic_to_name, \
            names_to_title = make_name_sets()
    
        # loop through the generic_to_names (or names_to_generic) and see if the abbreviations match any of the other full names
        abrv_names_to_full = {} 
        for abrv_name in generic_to_name:
            pos_names = get_possible_names(abrv_name)
            for pos_name in pos_names:
                if pos_name in all_names:
                    abrv_names_to_full = update_dict(abrv_names_to_full, abrv_name, pos_name)
    
    
        return abrv_names_to_full
    
    df['Authors'] = df['Authors'].apply(custom_name_clean)
    scrape_authors = df['scrap_auth_id'].unique()
    
    for author in scrape_authors:
        df_temp = df[df['scrap_auth_id'] == author]
        abrv_names_to_full = map_initials_to_names(df_temp)
        
        
        #updated_authors = map_initials_to_names(df_temp)
    
    return df

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

def one_hot_encode(x, cats_to_use):
    # one hot encodes the column  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore', categories = list(cats_to_use))
    
    enc.fit(temp)
    
    return None

def one_hot_encode_col(df_col, cats_to_use: set):
    """ df(col) -> sparse df
            # this works but its also really slow
         #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

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
                index=df.index,
                columns=mlb.classes_)
    
    
    #df = df.join(
    #            pd.DataFrame.sparse.from_spmatrix(
    #                mlb.fit_transform(df.pop(col)),
    #                index=df.index,
    #                columns=mlb.classes_))
    
    return output


def one_hot_encode_for_col(df_col, cats_to_use):
    # one hot encodes the column  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
    
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore', categories = list(cats_to_use))
    df_col_list= df_col.to_list()
    enc.fit(df_col_list)
    
    return None

def str_col_to_list(x, splitter): # meant to be used with apply or lambda function

    return [y for y in x.split(splitter)]
    
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


def process_cols(df):
    # converts each col to one hot encoded, calls functions to get
    # all the cats, clean cats, and finally uses the set of cats for each
    # col to get sparse mats. 
    
    
    # need a good way to get the dummies for normal cols 
    # https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
    
    df['Authors_list'] = df['Authors'].apply(lambda x: str_col_to_list(x, ","))
    df['titleID_list'] = df['titleID'].apply(lambda x: str_col_to_list(x, " "))
    
    df['Authors_list'] = df['Authors_list'].apply(lambda x: make_lower(x))
    df['titleID_list'] = df['titleID_list'].apply(lambda x: make_lower(x))
    # getting catagories
    authors = get_all_cat(df['Authors_list'])
    title_words = get_all_cat(df['titleID_list'])

    # getting synonyms from title words
    synonyms = buildCustomLookup(title_words)
    
    # consolidating synomyms
    df['titleID_list2'] = df['titleID_list'].apply(lambda x: \
                                                   merge_synonym(x, synonyms))
    
    title_words = remove_synonym(title_words, synonyms)
        
    title_words_df = one_hot_encode_col(df['titleID_list2'], title_words) # this is a sparse df
    
    title_words_df2 = threshold_sparse_df(title_words_df, 2)
    
    journal_df = pd.get_dummies(df['Journal'], sparse = True)
    
    journal_df = threshold_sparse_df(journal_df, 3)
    
def custom_encoder(df, categorical_features):
    
    nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    customize_stop_words = ['-','Single']  ##### adding custom stopwords
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    extraWords = ['year','Authors','Journal'] ####### cols from the df to be added to the bow
    toks2Rem = ['\n  ','--',"",',',"\to","..."] # if we seem these tokens get rid of them
    symbol2Rem = set(['%','$','{','}',"^","/", "\\",'#','*',"'",\
                  "''", '_','(',')', '..',"+",'-',']','[']) # remove 
    
    start = time.time()
    dicBow,dicMain = setUpBow2(df,categorical_features) ### get initial data
    end = time.time()### it doesn't look like I acutally use tok set
    
    setUpBow1 = end - start
    ####################
    dicBow = customProcTok(dicBow,symbol2Rem) #### this isn't working right
    print('did customProcTok')

    print('trimmed bow')
    ###############
    toks2Rem = ['\n  ','--',"",',',"\to","...","the", "what", "that", "then", \
                "and", "for", "couldnt find", "from", 'with', "non", "in", "of"]
    dicBow = remTok(dicBow,toks2Rem)
    ############### work on this later once we have more words
    #dicBow = customLookUpSpacy(dicBow)
    word_to_root = buildCustomLookup(dicBow)
    
    start = time.time()
    print('starting popBow')
    """ this is also for later
    """
    #labelY,bowVecY,dicWord2Vec,dicWordPaper = popBow(dicBow,futureDict)
    
    thresh = 10
    dicBow = trimBow2(dicBow, thresh)
    
    labelX,bowVecX,dicWord2Vec,dicWordPaper = popBow(dicBow,dicMain, word_to_root)
    
    print('finished pop Bow')
    end = time.time()
    popBow1 = end - start
        
    outLoc = 'one_hot_encoded_data'

    fList = [dicBow,dicMain,labelX,dicWord2Vec]
    nList = ['dicBow','dicMain','labelX', 'zDic2WordVec']

    saveOutput2(fList,nList,outLoc)  ###### saving variables in the list with the desired names
    
    ############### converting bowvec to csc and saving it
    bowVecX = bowVecX.tocsc()
    path = outLoc + '/' + 'bowVecSparseX.npz'  
    scipy.sparse.save_npz(path, bowVecX)  ##### saving the sparse matrix
    
    return dicBow, dicMain

if __name__ == "__main__":
    ############ load the data, keeping the ids col as a str
    #df_list = load_all_dfs("cleaned_data")
    
    # # for testing we want this to run fast
    # df_list = df_list[0:1]
    
    #df = cat_dfs(df_list)
    
    df = pd.read_csv("cleaned_data//df_for_results__28-03-2021 10_02_35.csv")
    df = df.iloc[0:10000]
    
    ### droping some cols we aren't using in the hopes that this runs faster
    cols_to_drop = ['Unnamed: 0', 'title_main', 'Conference', 'Source', 'book', \
                    'publisher', 'vol', 'issue', 'pages', 'urlID', 'citedYear']
    
    df = df.drop(labels = cols_to_drop, axis = 1)
    
    df.reset_index()
    categorical_features = ['year','Authors','Journal']
    
    process_cols(df)
    
    #dicBow, dicMain = ustom_encoder(df, categorical_features)
    
    matrices = []
    all_column_names = []
    # creates a matrix per categorical feature
    for c in categorical_features: #    https://www.kaggle.com/alexandrnikitin/efficient-xgboost-on-sparse-matrices

        matrix, column_names = sparse_dummies(df, c)
        matrices.append(matrix)
        all_column_names.append(column_names)
    


    train_sparse = hstack(matrices, format="csr")
    feature_names = np.concatenate(all_column_names)
    del matrices, all_column_names



    #clean_authors(df)
    ### loading spacy library and stopwords
    nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    customize_stop_words = ['-','Single']  ##### adding custom stopwords
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    extraWords = ['year','Authors','Journal'] ####### cols from the df to be added to the bow
    toks2Rem = ['\n  ','--',"",',',"\to","..."] # if we seem these tokens get rid of them
    symbol2Rem = set(['%','$','{','}',"^","/", "\\",'#','*',"'",\
                  "''", '_','(',')', '..',"+",'-',']','[']) # remove 
    
    start = time.time()
    dicBow,dicMain = setUpBow2(df,extraWords) ### get initial data
    end = time.time()### it doesn't look like I acutally use tok set
    
    setUpBow1 = end - start
    ####################
    dicBow = customProcTok(dicBow,symbol2Rem) #### this isn't working right
    print('did customProcTok')

    print('trimmed bow')
    ###############
    toks2Rem = ['\n  ','--',"",',',"\to","...","the", "what", "that", "then", \
                "and", "for", "couldnt find", "from", 'with', "non", "in", "of"]
    dicBow = remTok(dicBow,toks2Rem)
    ############### work on this later once we have more words
    #dicBow = customLookUpSpacy(dicBow)
    word_to_root = buildCustomLookup(dicBow)
    
    start = time.time()
    print('starting popBow')
    """ this is also for later
    """
    #labelY,bowVecY,dicWord2Vec,dicWordPaper = popBow(dicBow,futureDict)
    
    thresh = 10
    dicBow = trimBow2(dicBow, thresh)
    
    labelX,bowVecX,dicWord2Vec,dicWordPaper = popBow(dicBow,dicMain, word_to_root)
    
    print('finished pop Bow')
    end = time.time()
    popBow1 = end - start
        
    outLoc = 'one_hot_encoded_data'

    fList = [dicBow,dicMain,labelX,dicWord2Vec]
    nList = ['dicBow','dicMain','labelX', 'zDic2WordVec']

    saveOutput2(fList,nList,outLoc)  ###### saving variables in the list with the desired names
    
    ############### converting bowvec to csc and saving it
    bowVecX = bowVecX.tocsc()
    path = outLoc + '/' + 'bowVecSparseX.npz'  
    scipy.sparse.save_npz(path, bowVecX)  ##### saving the sparse matrix
    
    # bowVecY = bowVecY.tocsc()
    # path = outLoc + '/' + 'bowVecSpares.npz'  
    # scipy.sparse.save_npz(path, bowVecY)  ##### saving the sparse matrix
    ########### temp analysis 2-24 8 pm
    
    # compDicBow = compositionOfDicBow(dicBow,authorSet,journalSet) # i don't remeber what this does
    
    # tots = compDicBow['other']  + compDicBow['journal'] + compDicBow['author'] 
        
