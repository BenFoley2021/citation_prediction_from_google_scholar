# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:53:58 2021

converts the input df into one hot encoded outputs

misc to do:
    strip (the autors have blank space)

@author: Ben Foley
"""

import pandas as pd
import spacy
import numpy as np
import time
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
from generic_func_lib import *

def getYear(dfAll):
    import numpy as np

    def custom1(y):
    
        years = np.linspace(1990,2020,31).astype(int).tolist()
        for i,year in enumerate(years):
            years[i] = str(year)
            if years[i] in y:
                #print(years[i])
                return years[i]
    
    
    dfAll['year'] = ''
    dfAll['year'] = dfAll['jref'].apply(lambda y: custom1(y))
    
    dfAll = dfAll[dfAll['year'] != None]
    dfAll = dfAll.dropna(axis = 0, subset =['year'])
    dfAll['year'] = dfAll['year'].astype(str)
    
    return dfAll

def getTok2(dfIn):
    
    tokens = []
    #start = time.time()
    #disable = ["tagger", "parser","ner","textcat"]
    for doc in nlp.pipe(df['titleID'].astype('unicode').values):#,disable = disable):
        #doc.text = doc.text.lower()
        #tokens.append([token.text.lower() for token in doc if not token.is_stop])
        
        tempList = ([token.lemma_ for token in doc])
    
    
        for i,string in enumerate(tempList):
            try:
                tempList[i] = string.lower()
            except:
                print(string)
    
        tokens.append(tempList)
    
    dfIn['tokens'] = tokens
    
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
    dfIn = getTok2(dfIn)  ##### getting tokens (or lemma) and adding the list as a new col in the df. # this is fairly slow
    print('got tokens')

    def custom2(x):
        #### adds to the list of toks in the tokens column
        ### extra info not in the title is added so it can be used in the bag of words
        tempList= []
        for col in extraWords:
            if col == 'Authors':
                tempList = tempList + x[col].split(',')
            else:
                tempList = tempList +  [x[col]]
                
        return x['tokens'] +  tempList

    dfIn['tokens'] = dfIn.apply(custom2, axis=1)

    for i,row in dfIn.iterrows(): ###### looping through the dataframe (now with tokens/lemma) and building the dictionaries
        if i%2000 == 0:
            print(i)
        
        dicBow = updateBowDict(row['tokens'],dicBow)  #### update the bag of words dictionary 
        ##### add the extraWords in here
    
        ### updating the dict with row id, tokens, and cited by info
        dicMain[str(i)] = [row['cites_per_year'],row['tokens'],0]#### the zero will be updated with the word count for that ids later

    return dicBow, dicMain                                 

def removeFromDict(removeSet,dicIn): #### removes things add hoc from the dictionary
    #my_dict.pop('key', None) https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
    for thing in removeSet:
        dicIn.pop(thing,None)

    return dicIn

def customProcTok(bowDic,symbol2Rem):
    #### custom text parsing to remove some of the trash that made it through spacy tokenization
    ### loop through the dicBow
    ### if key has len 1 remove it
    ### if it has any of the characters in symbol2rem
    ### if it can be converted into a float
    ### more than 2 characters of the string can be converted into a float
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
    
    for thing in set2Rem:
        try:
            bowDic.pop(thing,None)
        except:
            pass
    
    return bowDic


def popBow(dicBow: dict, dicMain: dict): ####### this is the main loop which contructs the one hot encoded matrix for input to
    # various models
    ## constructing a scipy sparse matrix


    label = [] #### label is the number of citations, what I will try to predict later

    
    bowVec = lil_matrix((len(dicMain), len(dicBow)), dtype=np.int8) ## this will hold the one hot encoded bow
    
    dicWord2Vec = {} ### keeps track of which word corresponds to which col in bowVec
    wordVec = np.zeros(len(dicBow))  ### is the same length as the num of words in the bow
    
    ########adding another dict which keeps track of which papers have which words
    dicWordPaper = dicBow.copy()  #### the keys are the words, values list of paper ids
    
    for i,key in enumerate(dicBow): 
        dicWord2Vec[key] = i #putting the location of each word into the dict
        dicWordPaper[key] = [] ### changing the value to emptylists
    
    indi = -1
    for key in dicMain: ################## loop used to populate bowVec, as well update dicMain with the number of factors for each row (paper) that are being used in the model
        indi = indi +1
        dicBowTemp = dicBow.copy() ### getting copy of the bow and setting all keys to zero, will use this to count words for the current row
        dicBowTemp = dict.fromkeys(dicBowTemp, 0) #setting all keys to 0 https://stackoverflow.com/questions/13712229/simultaneously-replacing-all-values-of-a-dictionary-to-zero-python
        
        wordVecTemp = wordVec.copy()
        label.append(dicMain[key][0]) ### label is the number of citations, what I will try to predict later
        tempSet = set()
        for tok in dicMain[key][1]: ### looping through the tokens stored in dicMain
            
            try: #### the token may have been removed as a model input, so I need to check if it's still in dicBow
                dicBowTemp[tok] = dicBowTemp[tok] + 1 #### #times the word is present
                
                dicWordPaper[tok].append(key)  ###### adding the id of the paper which has that word
                tempSet.add(tok) ###adding the token to a set which will be used to update the sparse array later
                dicMain[key][2] = dicMain[key][2] + 1 #### keeping track of how many words from this ids are still going to be used
            except:
                pass
        
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
        
        # for j,key in enumerate(dicBowTemp): ############## this is the scaling problem.
        # # only need to non zero ones
        #     wordVecTemp[j] = dicBowTemp[key]
        
        #bowVec.append(wordVecTemp)

    return label,bowVec,dicWord2Vec,dicWordPaper  #### dicWordPaper is obsolete in this version

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

def getJournalName2(dfIn):
    """ this approach is to take all characters from before the first numeric
        in the jref string, and set them as the journal name"
        
        this approach missed occasionally. if jref starts with num then 
        it returns "" for that row
        
        can refine later 

    Parameters
    ----------
    dfIn : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    def getStrBeforeNumeric(x):
        """returns the string to left of the first numberic character
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        Str

        """
        
        tempStr = str()
        for i in x:
            #print(i)
            if i.isdigit() == False:
                tempStr += i
            else:
                break
        return tempStr
        
        
        
    #dfAll['jref'].apply(lambda y: custom1(y))
    dfIn['jName'] = dfIn['jref'].apply(lambda x: getStrBeforeNumeric(x))
    
    
    return dfIn
    
def cleanJName(dfIn):
    """
    particulary problematic
    Ap. J.  Ap.J. ApJ ApJ, Vol. ApJ Letters, ApJL
    
    #dont have anything to deal with advphys advances or advances phys
    """    

    
    def custom3(x):
        """ removing begining and end spaces, ","  '.'
        cleaning sequential, basic functions like strip first

        """
        x = x.lower()
        x = x.strip()
        x = x.replace('vol',"")
        x = x.replace('volume',"")       
        # while x[-1] == "." or x[-1] == ",":
        #     x = x[:-1]
    
    dfIn['jName'] = dfIn['jName'].apply(lambda x: custom3(x))
    
    return dfIn['jName']
    
def tempCleanJName(dfIn):
    """ container for test code, trying combine the various permutations of a
        journals name into one string
        
        at the moment this contains ad hoc things and isn't meant to be a 
        permanate solution'

    """
    def custom4(x):
        
        try:
            if x[-1] == 'v' and x[-2] == ' ':
                x = x[:-3]
        except:
            pass
            
        return x
    
    dfTemp = dfIn['jName']
    
    
    # months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', \
    #           'august', 'september','october','december']
    
    # for month in months:
    #     dfTemp.str.replace(month,"")
    
   #pickle.dump(df, open( "textTest2.p", "wb" ) )
    
    dfTemp = dfTemp.str.lower()
    dfTemp = dfTemp.str.strip()
    dfTemp = dfTemp.str.replace('vol',"")
    dfTemp = dfTemp.str.replace('volume',"")
    dfTemp = dfTemp.str.replace('(',"")
    dfTemp = dfTemp.str.replace(")","")
    dfTemp = dfTemp.str.replace(".","")
    dfTemp = dfTemp.str.replace(",","")
    dfTemp = dfTemp.str.replace("ume","")
    dfTemp = dfTemp.str.replace(":","")
    dfTemp = dfTemp.str.replace("[","")
    dfTemp = dfTemp.str.replace("&","and")
    dfTemp = dfTemp.str.replace(";","")
    dfTemp = dfTemp.str.replace("doi","")
    dfTemp = dfTemp.str.replace("~","")
    dfTemp = dfTemp.str.replace(" lett","letter")
    dfTemp = dfTemp.str.replace("letters","letter")
    dfTemp = dfTemp.str.replace("letterers","letter")
    dfTemp = dfTemp.str.replace("adv in","advances in")
    dfTemp = dfTemp.str.replace("adv","advanced")
    
    dfTemp = dfTemp.str.replace("grav","gravity")
    
    dfTemp = dfTemp.str.replace("scienc","science")
    dfTemp = dfTemp.str.replace("theoret","theoretical")
    dfTemp = dfTemp.str.replace("techno","technology")
    dfTemp = dfTemp.str.replace("techn","technology")
    dfTemp = dfTemp.str.replace("technol","technology")
    dfTemp = dfTemp.str.replace("technologies","technology")
    dfTemp = dfTemp.str.replace("issn","")
    dfTemp = dfTemp.str.replace("isn","")
    dfTemp = dfTemp.str.replace("commun","communication")
    dfTemp = dfTemp.str.replace("commu","communication")
    dfTemp = dfTemp.str.replace("communications","communication")
    dfTemp = dfTemp.str.replace("physic","phys")
    dfTemp = dfTemp.str.replace("physics","phys")
    dfTemp = dfTemp.str.replace(" geom"," geometry")
    
    dfTemp = dfTemp.str.replace('march',"")
    dfTemp = dfTemp.str.replace('january',"")
    dfTemp = dfTemp.str.replace('february',"")
    dfTemp = dfTemp.str.replace('april',"")
    dfTemp = dfTemp.str.replace('may',"")
    dfTemp = dfTemp.str.replace('june',"")
    dfTemp = dfTemp.str.replace('july',"")
    dfTemp = dfTemp.str.replace('august',"")
    dfTemp = dfTemp.str.replace('september',"")
    dfTemp = dfTemp.str.replace('october',"")
    dfTemp = dfTemp.str.replace('december',"")
    dfTemp = dfTemp.str.replace('novemeber',"")
    
    dfTemp = dfTemp.apply(custom4)
    
    
    dfTemp = dfTemp.str.strip()
    
    dfTempU = dfTemp.value_counts()
    
    dfTempU = dfTempU[dfTempU > 4]
    
    jNameSet = set(dfTemp.to_list())
    
    
    return dfTemp

def customLookUpSpacy(dicBow):
    """ trying to combine as many lemma as possible, keys in customLookUpDict
        point to the root (lemma (ex gentics => gentic). this function declares custom
        word => lemma and updates dicBow by moving all counts of the word to the
        lemma a poping the work. Ex all counts of gentics are added to gentic, then
        gentics is popped
        
        the custom lookup table here will be combined another lookUpDict generated
        elsewhere in future versions
    """
    import pickle
    #lemma i skipped on the first pass 
    # bais, binary, bipd, bivariant, bounds, braching
    # it seems like it would be possible to take of all the plurals by looking through
    # and ordered list with two pointers. can also extend to other suffixes
    
    customLookUpDict = {'batteries':'battery','bayesian':'bayes', \
                     'behavioral':'behaviour', 'behaviors': "behaviour", \
                     'behaviours': 'behaviour', 'behavioural': 'behaviour', \
                     'beliefs': 'believe', 'beliefs': 'believe', \
                     'benchmarking': 'benchmark', 'benchmarks':'benchmark', \
                     'bifurcation': 'bifurcate', 'bifurcations': 'bifurcate', \
                     'bilayered': 'bilayer', 'bilayers': 'bilayer', \
                     'billiards': 'billiard', \
                     'biometrics': 'biometric', 'biosignatures': 'biosignature', \
                     'biphotons': 'biphoton', 'bipolarons': 'bipolaron', \
                     'birefringent': 'birefringence', \
                     'blockchains': 'blockchain', \
                     'caching': 'cache',
                     'calculations': 'calculation', 'calculated':'calculation', \
                     'calibrated': 'calibrate', 'calibration':'calibrate', 'calibrations': 'calibrate', \
                     'calorimeters': 'calorimetry', 'calorimetric': 'calorimetry', 'calorimeter': 'calorimetry', \
                     'calorons': 'calorons', 'cameras': 'camera', \
                     'campaigns': 'campaign', \
                     'cancelation': 'cancellation', 'cancellations': 'cancellation', \
                     'capacitance': 'capacitive', 'capacitively': 'capacitive', \
                     'capacitors' : 'capacitive', 'capacitor': 'capacitive', \
                     'ceramics': 'ceramic', 'chalcogenides': 'chalcogenide', \
                     'chromodynamics': 'chromodynamic', 'chromospheric': 'chromosphere', \
                     'colloidal': 'colloid', 'colloids': 'colloid',
                     'combinatorics': 'combinatorics', 'comets': 'comet', \
                     'elastodynamics': 'elastodynamic', 'elastomers': 'elastomer', \
                     'elastoplasticity': 'elastoplastic', \
                     'electrocatalytic': 'electrocatalyst', \
                     'electrochemical': 'electrochemistry', 'electrochemically': 'electrochemistry', \
                     'electrodynamic': 'electrodynamics', \
                     'electroluminescence': 'electroluminescent',
                     'electrolytes': 'electrolyte', \
                     'electromechanical': 'electromechanics', \
                     'electromigrated': 'electromigration',
                     'electrophoretic': 'electrophoresis',
                     'electrostatically': 'electrostatic', 'electrostatically': 'electrostatic', \
                     'emitters': 'emitter', 'emotions': 'emotion', \
                     'emulsions': 'emulsion', \
                     'endomorphisms': 'endomorphism', \
                     'enriched': 'enrich', \
                     'ensembles': 'ensemble',
                     'entangled': 'entangle',
                     'ferromagnets': 'ferromagnet', 'ferromagnetically': 'ferromagnetic', \
                     'gaussians' : 'gaussian', 'gaussianitie': 'gaussianities', \
                     'genomics': 'genomic', 'genetics': 'genetic',
                     'geodesics': 'geodesic', \
                     'neutrinos': 'neutrino'}
        
        
    

    #dicBowKeys = pickle.load(open("dicBowKey.pckl", "rb" ))
    #dicBowKeys = dicBow.keys()

    for key in customLookUpDict:
        if key in dicBow:
            try:
                dicBow[customLookUpDict[key]] += dicBow[key]
                dicBow.pop(key)
            except:
                print('couldnt find ' + key)

    return dicBow



def buildCustomLookup(dicBow):

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
    for key in customLookUpDict:
        if key in dicBow:
            try:
                dicBow[customLookUpDict[key]] += dicBow[key]
                dicBow.pop(key)
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


if __name__ == "__main__":
    ############ load the data, keeping the ids col as a str
    df_list = load_all_dfs("cleaned_data")
    df = cat_dfs(df_list)
    
    ### loading spacy library and stopwords
    nlp = spacy.load('en_core_web_sm')

    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    
    #toRemove = ['-']
    customize_stop_words = ['-','Single']  ##### adding custom stopwords
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    extraWords = ['year','Authors','Journal','publisher'] ####### cols from the df to be added to the bow
    
    toks2Rem = ['\n  ','--',"",',',"\to","..."]
    
    symbol2Rem = set(['%','$','{','}',"^","/", "\\",'#','*',"'",\
                  "''", '_','(',')', '..',"+",'-',']','['])
    
    start = time.time()
    dicBow,dicMain = setUpBow2(df,extraWords) ### get initial data
    end = time.time()### it doesn't look like I acutally use tok set
    
    setUpBow1 = end - start
    ####################
    dicBow = customProcTok(dicBow,symbol2Rem) #### this isn't working right
    print('did customProcTok')
    
    
    ### building a custom dictionary to key track of lemma
    """ need to come back to this later when we have more stuff
    """
    
    # custLemmaLookUP = buildCustomLookup(dicBow)
    # dicBow = condenseDicBow(dicBow,custLemmaLookUP)
    ####remove words with low counts here
    thresh = 5
    dicBow = trimBow2(dicBow, thresh)
    
    # threshAuthor = 9
    # threshJournal = 9
    # authorSet = set(df['submit'].to_list())
    # journalSet = set(df['jName'].to_list())
    
    # this is the option for if we want different thresholds for different categories
    #dicBow = trimBow3(dicBow,authorSet,journalSet,thresh,threshAuthor,threshJournal)
    
    
    
    print('trimmed bow')
    ###############
    toks2Rem = ['\n  ','--',"",',',"\to","...","the", "what", "that", "then", \
                "and", "for", "couldnt find", "from", 'with', "non", "in", "of"]
    dicBow = remTok(dicBow,toks2Rem)
    ############### work on this later once we have more words
    dicBow = customLookUpSpacy(dicBow)
    
    start = time.time()
    print('starting popBow')
    """ this is also for later
    """
    #labelY,bowVecY,dicWord2Vec,dicWordPaper = popBow(dicBow,futureDict)
    
    labelX,bowVecX,dicWord2Vec,dicWordPaper = popBow(dicBow,dicMain)
    
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
        
