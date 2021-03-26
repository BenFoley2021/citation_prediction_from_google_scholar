# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:13:49 2021

Get the data output by the author_list crawler. This is stored in the outputs subfolder, with filename paperDictA______.pckl
Then combine all data into lists and convert lists into a df. Baisc cleaning and formatting steps are also performed on the data
so that it is ready to be used by the one_hot_encode script
    1) reformat the date to a datetime object
    2) convert any citation numbers listed as nan to 0
    3) get time difference between published date and scrapped date
    4) remove papers which don't hae an english title
    5) perfrom any custom processing of the cited_num (may be desired so we don't get divide by 0
                                                          error in downstream scripts')
    6) calculate citations per year
                                                       
resulting df then saved

This very slow to run, meant as a one time processing step. The bottle neck seems to be  checking if
the title is in english                                                       
                                                       
@author: Ben Foley
"""

import pandas as pd
import numpy as np

import pickle
import os
import pickle as pckl
import sqlite3

from langdetect import detect, detect_langs

from datetime import datetime

def get_key_vals(keysToGet: set,dictIn: dict) -> dict:
    def delKeys(singleDict):
        for key in list(singleDict.keys()):
            if key not in keysToGet:
                del singleDict[key]
        return singleDict
    
    if len(dictIn) == 0:
        return dictIn
    
    elif len(dictIn) == 1:
        dictIn = delKeys(dictIn)
            
    else:
        for outerKey in dictIn:
            dictIn[outerKey] = delKeys(dictIn[outerKey])
            
    return dictIn
    
    
def getPaths(main_dir: str, folderType: str, extension: str or None) -> list:
    """ make a list of directories based on folderType, a string which is used to
        identify folders
    
        the extension is added to the end of the path. in each paperGraph_X subfolder
        the output data is placed in the outputs folder

    Parameters
    ----------
    folderType : str
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    tempList = []
    for dirc in os.listdir(main_dir):
        if os.path.isfile(dirc) == False:
            print(dirc)
            if folderType in dirc:
                if extension:
                    tempList.append(main_dir + dirc + "\\" + extension)
                else:tempList.append(dirc)
                    
    return tempList
    

def load_dicts_from_dir(fileID: str, paths: list, keysToGet = None) -> dict:
    """ loads all pickle files which contain fileID into dictionary
        
        loops through all the .pckl files in path and updates a dictionary
        with the contents of each. returns this dict
    
    Parameters
    ----------
    fileID : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    tempDict = {}
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.pckl') and fileID in file:
                #pathFile = path + "\\" + file
                print(file)
                with open(path + "\\" + file,'rb') as f:
                    if keysToGet:
                        ### loadin file, removing unwanted keys, updating tempDict
                        tempDict.update(get_key_vals(keysToGet, pickle.load(f)))
    
                    else:                                     
                        tempDict.update(pickle.load(f))
    return tempDict
    
def load_dicts_from_dir_to_list(fileID: str, paths: list, colNames: dict) -> list:
    """ modifying load_dicts_from_dir to output a list instead of dict. the list
        will then be made into a df
    

    Parameters
    ----------
    fileID : str
        DESCRIPTION.
    paths : list
        DESCRIPTION.
    keysToGet : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    list
        DESCRIPTION.

    """

    list_form = []
    col_name_list = []
    out_list = []
    for key in colNames.keys():
        list_form.append([])
        col_name_list.append(key)
    
    out_list.append(col_name_list) # putting this lots of places so make sure everything is
                                    # is labeled correctly, it's easy to get rid of later
    
    
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.pckl') and fileID in file:
                #pathFile = path + "\\" + file
                print(file)
                
                with open(path + "\\" + file,'rb') as f:
                    temp_dict = pickle.load(f)
                    
                    # this is a nested dict
                    for topKey in temp_dict.keys():
                        temp_list = list_form.copy()
                        for key in colNames.keys():
                            try:
                                temp_list[colNames[key]] = temp_dict[topKey][key]
                            except: 
                                temp_list[colNames[key]] = 'cant read from scrapper out'
                        out_list.append(temp_list)
                    # replace this block
                    
                    # tempList. append (new row)
                    
    return out_list

def loadPickled(path,fileName):
    #import pickle
    fileName = path + "\\" +  fileName
    with open(fileName,'rb') as f:
        outName = pckl.load(f)
    
    return outName

def readMergeDicts(dir2Read, keysToGet): ### obsoleted
    """ merges all the .pickle files with "paperDict" into one dict,
        and all with "citedByDict"

    Parameters
    ----------
    dir2Read : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    allPapersDict= {}
    allCitedByDict = {}
    
    for file in os.listdir(dir2Read):
        if file.endswith('.pckl'):
            if 'paperDict' in file:
                tempVar = get_key_vals(keysToGet, loadPickled(dir2Read,file))
                try:
                    allPapersDict.update(tempVar)
                except:
                    print('error updating allPapersDict')
                    
            elif 'citedByIdDict' in file:
                tempVar = loadPickled(dir2Read,file)
                try:
                    allCitedByDict.update(tempVar)
                except:
                    print('error updating allCitedByDict')
                
    return allCitedByDict, allPapersDict
    
    
def getFirstAuthor(paperDict: dict) -> set:
    """ getting the first 


    """
    
    firstAuthorSet = set()
    
    for paper in paperDict:
        if len(list(paperDict[paper]['authors'].keys())) > 0:
            firstAuthorSet.add(list(paperDict[paper]['authors'].keys())[0])
            
    
    return firstAuthorSet
    
    
def allPapersDict_todf(allPapersDict: dict) -> pd.DataFrame:
    """ converting the dict to dataFrame. want the author IDs as a list
        right now they are a dict in the dict. loop through and update the
        keys, and get rid of cols we don't want. (dont need "raw" in 
        the df)

    Parameters
    ----------
    allPapersDict : dict
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    df = pd.DataFrame() 
    #dict2 = {'urlId': None, 'cited', None, 'title': None, ''}
    allPapersDict = backUpPaperDict.copy()
    for key in allPapersDict:
        allPapersDict[key]['authors'] = \
            list(allPapersDict[key]['authors'].keys())
        del allPapersDict[key]['raw']
    
        
    df = pd.DataFrame.from_dict(allPapersDict)
    df = df.transpose()
    
    return df
    
def dictToDF(dictIn, cols):
    """ trying to make a more generalized function to make a df from the dicts
    

    Parameters
    ----------
    dictIn : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    df = pd.DataFrame() 
    #dict2 = {'urlId': None, 'cited', None, 'title': None, ''}
    # allPapersDict = backUpPaperDict.copy()
    # for key in allPapersDict:
    #     allPapersDict[key]['authors'] = \
    #         list(allPapersDict[key]['authors'].keys())
    #     del allPapersDict[key]['raw']
    
        
    df = pd.DataFrame.from_dict(dictIn)
    df = df.transpose()
    
    return df
    
def isInDb(id_1: str, table: str, con: object) -> bool:
    
    """ checks to see if the id_1 is present in the index of the "table" TABLE
        in the database which is connected to "con"
    """
    try: 
        sql = "SELECT EXISTS(SELECT 1 FROM " + table + " WHERE id=?)"
        
        #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
        key = [(id_1)]
        with con:
            data = con.execute(sql, key)
            for row in data:
                tempVar = row
                
        tempVar = list(tempVar)
        
        if tempVar[0] == 1:
            print('already found this paper')
            return True
        else:
            return False
        
    except:
        print('error in isInDb')
        return False
    
def updateDbFirstAuthor(firstAuthors: set) -> None:
    table = 'firstauthors'
    # can i use an exisiting database function to insert 
    dbPath = "C:\\Users\\bcyk5\OneDrive\\Documents\GitHub\\citation_prediction_from_google_scholar\\data_mining\\main_db_A.db"
    con = sqlite3.connect(dbPath)

    data = []
    for author in firstAuthors:
        if isInDb(author, table, con) == False:
            data.append((author, 'placeHolder', "false"))
    
    sql = 'INSERT INTO '+ table + ' (id, value, scrapped) values (?, ?, ?)'
    
    # the code to try and insert is the same regardless of whether its papers or authors, args to con. execute prepared in above code
    try: # program can't stop for data base errors. 
        with con:
            con.executemany(sql, data)
    
    except: # if something goes wrong trying to insert everything at once, try one at a time, skip any that don't work
        print('error writing paperID to db')
        try:
            for dataRow in data:
                with con:
                    con.execute(sql, data)
        except:
            pass
    
    return None
    
def paperlist_to_df(paper_list):
    """ takes the list of papers and converts to df.
        
        how to use the columns names
            could loop through list until hit another col name
            place everything before into a df, cat with cur df
            
            repeat this may be slow, but meh
            
            i think other ways would require an intermediate formatting step
            
            
                colNames = {"titleID": 0, "title_main": 1, "cited": 2, "authors": 3, "pubDate": 4, \
                "journal": 5, "vol": 6, "issue": 7, "pages": 8, "publisher": 9, \
                "description": 10, "citedYear": 11, "allscrap_auth_id": 12, "urlID": 13, \
                "all": 14}
            
            
    """
    
    df = pd.DataFrame(paper_list[1:], columns = paper_list[0])
    
    return df
    
def clean_df(df):
    """ using lots of functions from raw data to to bag of words
        
        want all the basic cleaning and formatting steps to be in one function.
        
        need to get get
            1) year
            2) date
            3) citations
            4) cits per year
    
            5) remove confrences and books
                remove any journals or sources that have "confrence in them"
                
            push df with title, authors, journal, year, cites/year to one hot encoder
                the encoder will do second round of cleaning 
            
            the round of cleaning in this function is to remove artifacts from scrapping 

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    def custom_get_date(x):
        # getting date time from scrapped pubDate. if no day or month, it's set to one
        x_pieces = x.split("/")
        
        try:
        
            if len(x_pieces) == 3:
                return datetime(int(x_pieces[0]), int(x_pieces[1]), int(x_pieces[2]))
            if len(x_pieces) == 2:
                return datetime(int(x_pieces[0]), int(x_pieces[1]), 1)
            if len(x_pieces) == 1:
                return datetime(int(x_pieces[0]), 1, 1)
            
        except:
            return np.nan
            
    def custom_get_cited(x):
        if x == "NA":
            return 0
        
        return int(x.split(" ")[2])
    
    def custom_get_year_difference(x):
        try:
            return ((d_scrape - x).days)/365
        except:
            return np.nan
    
    def custom_is_en(x):
        # check and see if the title is enlisgh
        try:
            major_lang = detect_langs(x)[0]
            if major_lang.lang == 'en':
                return True
            else:
                return False
        except:
            return False
    
    def mod_cited_data(df):
    
        df['cited_num_mod'] = df['cited_num'] + 1
        
        df['cites_per_year_mod'] = df['cited_num_mod']/ df['date'].apply(lambda x: custom_get_year_difference(x))
        
        return df    
    
    # need to drop duplicates
    df = df.drop_duplicates(subset=['titleID'])

    cols_to_drop = ['all', 'title_main', 'Conference', 'Source', 'vol', 'issue', 'pages', 'urlID', 'description', \
                    'scrap_auth_id', 'citedYear']
    
    cols_to_return = ['titleID', 'Authors','Journal', 'publisher', 'year', 'cited_num', \
                      'cites_per_year', 'date']
        
    df = df.drop(cols_to_drop, axis = 1)
    
    df = df[df['Journal'] != "cant read from scrapper out"]
    
#    df['is_en'] = df['titleID'].apply(lmbda x: custom_is_en(x))
    
    df = df[df['titleID'].apply(lambda x: custom_is_en(x)) == True] ### this is really slow
    
    # getting eyar
    df['year'] = df['pubDate'].apply(lambda x: x.split("/")[0])
    
    
    df['date'] =  df['pubDate'].apply(lambda x: custom_get_date(x))

    d_scrape = datetime(2021, 3, 20) ## datetime of scrape (approx)
    
    df['cited_num'] = df['cited'].apply(lambda x: custom_get_cited(x))
    
    df['cites_per_year'] = df['cited_num']/ df['date'].apply(lambda x: custom_get_year_difference(x))
    
    
    ### final formatting
    df['year'] = df['year'].astype(str)

    df = df[cols_to_return]
    
    df = df[df['cites_per_year'] >= 0]
    
    df = mod_cited_data(df)

    df = df[df['cites_per_year'] >= 0]

    return df

if __name__ == "__main__":
    ### loading all the papers we've gotten so far from crawling the papers by citation
    
    main_dir = "C:\\Users\\bcyk5\OneDrive\\Documents\GitHub\\citation_prediction_from_google_scholar\\data_mining\\"
    cur_dir = os.getcwd()
    
    # this tells which position in the list corresponds to
    colNames = {"titleID": 0, "title_main": 1, "cited": 2, "Authors": 3, "pubDate": 4, \
                "Journal": 5, "Conference": 6, "Source": 7, "vol": 8, "issue": 9, "pages": 10, "publisher": 11, \
                "description": 12, "citedYear": 13, "scrap_auth_id": 14, "urlID": 15, \
                "all": 16}
    
    dirs_to_read = getPaths(main_dir, "authorList_", "outputs")
    
    ##### temp to readuce amount of data
    #dirs_to_read = dirs_to_read[len(dirs_to_read) - 3:-1]
    
    keysToGet = set(['titleID', 'cited', 'authors', 'authors', 'pubDate', 'journal'])
    papersDict = load_dicts_from_dir("paperDictA", dirs_to_read, keysToGet)
    
    papers = load_dicts_from_dir_to_list("paperDictA", dirs_to_read, colNames)
    
    df = paperlist_to_df(papers)
    
    
    df = clean_df(df)
    
    df.to_csv('current_df_to_be_one_hot_encoded.csv')
    #firstAuthors = getFirstAuthor(papers)
    # author_dirs = getPaths("authorList", "outputs")
    # author_papers = load_dicts_from_dir("paperDictA", author_dirs)
    
    #updateDbFirstAuthor(firstAuthors)

    
    #paper_graph_dirs = getPaths("paperGraph", "outputs")
    
    
    
    ### loading all the papers we've gotten so far by crawling the authors gs pages
    # tempDict = {}
    # for i, thing in enumerate(papers):
    #     if i > 20:
    #         break
    #     i += 1
        
    #     tempDict.update({thing: papers[thing]})
        
        

