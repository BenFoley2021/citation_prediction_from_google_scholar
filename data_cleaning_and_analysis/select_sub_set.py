# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:50:57 2021

want to be able to select a subset of all the data, so can have more managle sized data
for experimenting. This will be done so the sampling is still as dense as possible (not randomly selecting)

first pass, use key words to select. If key words are in title, abstract, or 

example use general_multi_proc
df['titleID_list'] = general_multi_proc(str_col_to_list_par, df['titleID'], " ")

4-19 7pm rewritting so this is more flexible
    9pm, locked while trying to check if table exists.
        going to skip this and do it manually for now
        


@author: Ben
"""
from generic_func_lib import *
import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import os

#from one_hot_encode_parallelizeing import general_multi_proc
from get_data_from_author_crawl import *

def loop_folders_do_thing(fileID: str, paths: list, accumulator, \
                                 keywords: list, locations: list
                                     ) -> list:
    key2get = 'authors'
    def check_words(x):
        for place in locations:
            for word in keywords:
                try:
                    if word in x[place].lower():
                        return True
                except:
                    return False
    
    def get_values_as_list(temp_dict, key2get):
        return list(temp_dict[key2get].keys())
    
    def file_ops_get_keyword_authors(temp_dict, author_set):
        temp_set = set()
        for topKey in temp_dict.keys():
            if check_words(temp_dict[topKey]) == True:
               for thing in get_values_as_list(temp_dict[topKey], key2get):
                   author_set.add(thing)
        
        return author_set
        
    def open_file_ops(f, file):
        try:
            temp_dict = pickle.load(f)
            f.close()
        except:
            print(file)
            temp_dict = None
        # maybe more logic?
        return temp_dict
    
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.pckl') and fileID in file:
                with open(path + "\\" + file,'rb') as f:
                    temp_file = open_file_ops(f, file)
                    if temp_file:
                        accumulator = file_ops_get_keyword_authors(temp_file, accumulator)
                    
                    
    return accumulator
    
def are_keywords_there(df, keywords, locations):
    """ Checks if keywords are found in the df at locations.
    """
    def check_words(x):
        for place in locations:
            for word in keyowrds:
                if word in x[place].lower():
                    return True


    df = df[df.apply(check_words) == True]
    #more logic needed?
    return df

def load_dicts_from_dir_to_list_condition(fileID: str, paths: list, colNames: dict, \
                                 keywords: list, locations: list) -> list:
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
    def check_words(data_point):
        for place in locations:
            for word in keywords:
                if word in data_point[place].lower():
                    return True
        return False
        

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

                with open(path + "\\" + file,'rb') as f:
                    temp_dict = pickle.load(f)
                    # this is a nested dict
                    for topKey in temp_dict.keys():
                        if check_words(temp_dict[topKey]) == True:
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



def update_db_create_table(firstAuthors: set, table) -> None:
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
    def check_if_table(table, con):
        #check if the table exists, create it if not
        with con:
            con.execute("""
                        CREATE TABLE perovskiteAuthors (
                            id TEXT NOT NULL PRIMARY KEY,
                            value TEXT,
                            scrapped TEXT
                            );
                        """)
                
                
            # sql = " SELECT count(name) FROM sqlite_master WHERE type='table' AND name=(table); values(?)"
            # con.execute(sql, table)
#if the count is 1, then table exists

    # can i use an exisiting database function to insert 
    dbPath = 'C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\dataBase\\db_C.db'
    con = sqlite3.connect(dbPath)
    #check_if_table(table, con)
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
        print('error writing thing to db')
        try:
            for dataRow in data:
                with con:
                    con.execute(sql, data)
        except:
            pass
    
    return None

def get_all_authors_keywords():
    out_dir = "data_subset\\"
    
    main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
    cur_dir = os.getcwd()

    dirs_to_read = getPaths(main_dir, "paperGraph", "outputs")

    keywords = ['perovskite', 'hoips', 'mapi', 'organic-inorganic', 'lead halide', 'csPbX', 'MAPI']

    locations = ['abstractWords', 'title']
    
    authors = set()
    
    authors = loop_folders_do_thing("paperDict", dirs_to_read, authors, keywords, locations)

    return authors

def get_abstract(x):
    """ selects the abstract based on the text which comes before and after
        Doing it this way because I didn't save raw html, just text
    """
    
    start = 'Description\n'
    if x['cited'] == 0:
        end = '\nScholar articles'
    else:
        end = '\nTotal citations\nCited by'
    
    return x['all'][x['all'].find(start)+len(start):x['all'].rfind(end)]


def get_subset():
    out_dir = "data_subset\\"
    
    main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
    cur_dir = os.getcwd()
    
    dirs_to_read = getPaths(main_dir, "authorList_V2-7", "outputs")
    colNames = {"titleID": 0, "title_main": 1, "cited": 2, "Authors": 3, "pubDate": 4, \
            "Journal": 5, "Conference": 6, "Source": 7, "book":8, "vol": 9, "issue": 10, "pages": 11, "publisher": 12, \
            "description": 13, "citedYear": 14, "scrap_auth_id": 15, "urlID": 16, \
            "all": 17}
    cols_to_return = ['titleID',"title_main", 'Authors', 'Journal', "Conference", "Source", "book", \
                   'publisher', "vol", "issue", 'year', "pages", 'cited_num', \
               'cites_per_year', 'date', 'scrap_auth_id', "citedYear", "urlID", 'abstract']
        
        
    keysToGet = set(['titleID', 'cited', 'authors', 'authors', 'pubDate', 'journal', 'abstract'])
        
    keywords = ['perovskite', 'hoip', 'organic-inorgancic', 'lead-halide', 'CH 3 NH 3 PbI 3', \
                'CsPbX', 'MAPI']
        
    locations = ['all']
    
    papers = load_dicts_from_dir_to_list_condition("paperDictA", dirs_to_read, colNames, keywords, locations)

    df = paperlist_to_df(papers)
    """ use apply to get the abstract text from all, create abstract column
    """
    df['abstract'] = df.apply(get_abstract, axis = 1)
    df = clean_df(df, cols_to_return)
    
    
    
    
    ############ going to save each df seperately for now
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    fileName = "df_for_results_perovskites"
    fileName = fileName + '_' + dt    
    
    df.to_csv(out_dir + fileName + '.csv')


def get_titles_by_author_id():
    """ need to make a list of each authorID for authors with a gs profile
        listed on a paper
    """
    
    

if __name__ == "__main__":
    #get_subset()
    
    get_subset()
    
    #authors = get_all_authors_keywords()
    #out_dir = "data_subset\\"
    
    #pickle.dump(authors, open( 'perovskite authors 4-19.pickle', "wb" ) )
    
    #authors = pickle.load(open("data_subset\\" + 'perovskite authors 4-26.pickle', "rb"))
    
    #update_db_create_table(authors, 'perovskiteAuthors')
    
    #authors = get_all_authors_keywords()
    