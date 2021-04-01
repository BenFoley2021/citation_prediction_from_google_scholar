# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:13:49 2021

the defualt behavoir is to save the files read after each run. when the script is
run again, the files read is loaded. files in files_read are skipped when loading
the paperDicts.



@author: bcyk5
"""

import pandas as pd
import numpy as np
import pickle
import os
import pickle as pckl
import sqlite3

from langdetect import detect, detect_langs

from datetime import datetime

from generic_func_lib import *



def get_key_vals(keysToGet: set,dictIn: dict) -> dict: # i think this is also obsoleted
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
        A key word which determines whether the folder is added to the list. eg folders with 
        "latest_crawler_runfile" will be added to the list so they can be later checked for data.

    Returns
    -------
    list
        The list of paths from which data can be loaded from by load_dicts_from_dir_to_list.

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
    
    
def load_dicts_from_dir_to_list(fileID: str, paths: list, colNames: dict, files_read: set) -> list:
    """ modifying load_dicts_from_dir to output a list instead of dict. the list
        will then be made into a df
    

    Parameters
    ----------
    fileID : str
        The key work which determines if the file is read (eg if "I_have_data" in file, it gets read).
    paths : list
        The list of paths to check for files which have fileID in their name.
    colNames : list
        The names of all attributes that we want to get from the list. Corresponds to keys in the 
        dictionary files.
    files_read: set
        The files that we have already processed. If the file name is in this list we skip it.

    Returns
    -------
    out_list : list
        The processed data, where each item in the list is a data point, and the contains the attributes
        of that data point. strings and objects are contained in the list representing each row. This is
        what will be converted to a df later
        
    files_read : set
        The set given as an argument after being updated to include all files loaded during this function
        call.
        
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
                if file not in files_read:
                    print(file)
                    files_read.add(file)
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
                    
    return out_list, files_read
    
def getFirstAuthor(paperDict: dict) -> set: # this will need to be rewritten to work with a df
    """ getting the first author of each paper
        Technically just getting the first author in the list. The authors appear
        in the order they are listed on the paper (the first author will be first 
        in the list). However, if the first author doesn't have a gs profile then 
        highest ranked author who does will be first on the list'


    """
    
    firstAuthorSet = set()
    
    for paper in paperDict:
        if len(list(paperDict[paper]['authors'].keys())) > 0:
            firstAuthorSet.add(list(paperDict[paper]['authors'].keys())[0])
            
    
    return firstAuthorSet
    
def isInDb(id_1: str, table: str, con: object) -> bool:
    
    """ checks to see if the id_1 is present in the index of the "table" TABLE
        in the database which is connected to "con"
        
        add hoc function sometimes used by me to check whats where
        
    Parameters:
        id_1: Str, the value to be checked to see if it's a primary key in the table
        table: Str, the table in the database to check for id_1
        con: sqllite3 database connection
        
    Returns:
        None
        
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
    """ This is for manually adding more first authors to the firstauthors table
    

    Parameters
    ----------
    firstAuthors : set
        The authors to be added to the sqlite3 db.

    Returns
    -------
    None
        data base updated, no return balue needed.

    """
    table = 'firstauthors'
    # can i use an exisiting database function to insert 
    dbPath = 'C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\dataBase\\main_db_A.db'
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
    
def paperlist_to_df(paper_list: list) -> pd.DataFrame:
    """ takes the list of papers and converts to df.

        This turned out just to be a one liner, leaving as funciton in case more logic needs to be
        added later                    
            
                expected column names
                colNames = {"titleID": 0, "title_main": 1, "cited": 2, "authors": 3, "pubDate": 4, \
                "journal": 5, "vol": 6, "issue": 7, "pages": 8, "publisher": 9, \
                "description": 10, "citedYear": 11, "allscrap_auth_id": 12, "urlID": 13, \
                "all": 14}
            
        Parameters:
            paper_list: A list where each item in the list is the row, and sublists are
                        values of that row. The first item in the list contains the columns names
                        (these were hardcoded by load_dicts_from_dir_to_list)
            
    """
    
    df = pd.DataFrame(paper_list[1:], columns = paper_list[0])
    
    return df
    
def clean_df(df, cols_to_return):
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
    
    def mod_cited_data(df, cols_to_return):
    
        df['cited_num_mod'] = df['cited_num'] + 1
        
        df['cites_per_year_mod'] = df['cited_num_mod']/ df['date'].apply(lambda x: custom_get_year_difference(x))
        
        return df    
    
    
    
    # need to drop duplicates
    df = df.drop_duplicates(subset=['titleID'])
    
    #df = df[df['Journal'] != "cant read from scrapper out"]
    
#    df['is_en'] = df['titleID'].apply(lmbda x: custom_is_en(x))
    print('started removing non en')
    df = df[df['titleID'].apply(lambda x: custom_is_en(x)) == True] ### this is really slow
    print('done removing non en')
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
    
    #df = mod_cited_data(df)

    #df = df[df['cites_per_year'] >= 0]

    return df


if __name__ == "__main__":
    ### loading all the papers we've gotten so far from crawling the papers by citation
    
    main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
    cur_dir = os.getcwd()
    
    out_dir = cur_dir + "\\cleaned_data\\"
    
    files_read = load_one_pickled('files_read.pckl', out_dir)
    #files_read = set()
    # this tells which position in the list corresponds to
    colNames = {"titleID": 0, "title_main": 1, "cited": 2, "Authors": 3, "pubDate": 4, \
                "Journal": 5, "Conference": 6, "Source": 7, "book":8, "vol": 9, "issue": 10, "pages": 11, "publisher": 12, \
                "description": 13, "citedYear": 14, "scrap_auth_id": 15, "urlID": 16, \
                "all": 17}
    
    dirs_to_read = getPaths(main_dir, "authorList_V2-7", "outputs")
    
    ##### temp to readuce amount of data
    #dirs_to_read = dirs_to_read[len(dirs_to_read) - 3:-1]
    
    keysToGet = set(['titleID', 'cited', 'authors', 'authors', 'pubDate', 'journal'])
    #papersDict, files_read = load_dicts_from_dir("paperDictA", dirs_to_read, keysToGet)
    
    papers, files_read = load_dicts_from_dir_to_list("paperDictA", dirs_to_read, colNames, files_read)
    
    df = paperlist_to_df(papers)
    
    # only keeping these cols
    cols_to_return = ['titleID',"title_main", 'Authors', 'Journal', "Conference", "Source", "book", \
                      'publisher', "vol", "issue", 'year', "pages", 'cited_num', \
                  'cites_per_year', 'date', 'scrap_auth_id', "citedYear", "urlID"]
    
    df = clean_df(df, cols_to_return)
    
    ############ going to save each df seperately for now
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    fileName = "df_for_results_"
    fileName = fileName + '_' + dt    
    
    df.to_csv(out_dir + fileName + '.csv')
    #df.to_csv(out_dir + 'current_df_to_one_hot_encode2.csv')
    
    save_pickles([files_read],['files_read'],out_dir)
    
    
    

        

