# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:50:57 2021

5-3 leaving off on get_all_from_select_authors. need to filter df by rows which have 
an author in the selected authors

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
import pickle
#from one_hot_encode_parallelizeing import general_multi_proc
from get_data_from_author_crawl import *

def check_words(x, locations, keywords):
    for place in locations:
        for word in keywords:
            try:
                if word in x[place].lower():
                    return True
            except:
                return False

def get_values_as_list(temp_dict, key2get):
    return list(temp_dict[key2get].keys())

def file_ops_get_keyword_authors(temp_dict, author_set, key2get, locations, keywords):
    #temp_set = set()
    for topKey in temp_dict.keys():
        if check_words(temp_dict[topKey], locations, keywords) == True:
           for thing in get_values_as_list(temp_dict[topKey], key2get):
               author_set.add(thing)
    
    return author_set

def file_ops_get_authorsId_for_paper(temp_dict, accumulator, locations):
    def organize_paper_author(papers):
        for paper in papers:
            if paper in accumulator:
                accumulator[paper] += " " + topKey
            else:
                accumulator[paper] = topKey

        return accumulator
    
    for topKey in temp_dict.keys():
        organize_paper_author(temp_dict[topKey][locations[0]])
    
    return accumulator
    
def open_file_ops(f, file):
    try:
        temp_dict = pickle.load(f)
        f.close()
    except:
        print(file)
        temp_dict = None
    # maybe more logic?
    return temp_dict
    
def loop_folders_do_thing_V2(fileID: str, paths: list, accumulator, \
                                 keywords: list, locations: list
                                     ) -> list:
    key2get = 'authors'
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.pckl') and fileID in file:
                with open(path + "\\" + file,'rb') as f:
                    temp_file = open_file_ops(f, file)
                    if temp_file:
                        accumulator = file_ops_get_authorsId_for_paper(
                            temp_file, accumulator, locations)
                    
    return accumulator
    

def loop_folders_get_authors_paperGraph(fileID: str, paths: list, accumulator, \
                                 keywords: list, locations: list
                                     ) -> list:
    key2get = 'authors'
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.pckl') and fileID in file:
                with open(path + "\\" + file,'rb') as f:
                    temp_file = open_file_ops(f, file)
                    if temp_file:
                        accumulator = file_ops_get_keyword_authors(
                            temp_file, accumulator, key2get, locations, keywords)
                    
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
    def check_words(data_point, keywords):
        for place in locations:
            for word in keywords:
                try:
                    if word in data_point[place].lower():
                        return True
                except:
                    pass
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
                        if topKey == "High efficiency planar-type perovskite solar cells with negligible hysteresis using EDTA-complexed SnO 2":
                            print('found paper of interest')
                        
                        if check_words(temp_dict[topKey], keywords) == True:
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
                        CREATE TABLE solarAuthors (
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
    check_if_table(table, con)
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
    
    keywords = ['solar', 'photovoltaic']
    
    locations = ['abstractWords', 'title']
    
    authors = set()
    
    authors = loop_folders_get_authors_paperGraph("paperDict", dirs_to_read, authors, keywords, locations)
    
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


def get_subset(dir_code, file_name):
    out_dir = "data_subset\\"
    
    main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
    cur_dir = os.getcwd()
    
    dirs_to_read = getPaths(main_dir, dir_code, "outputs")
    colNames = {"titleID": 0, "title_main": 1, "cited": 2, "Authors": 3, "pubDate": 4, \
            "Journal": 5, "Conference": 6, "Source": 7, "book":8, "vol": 9, "issue": 10, "pages": 11, "publisher": 12, \
            "description": 13, "citedYear": 14, "scrap_auth_id": 15, "urlID": 16, \
            "all": 17}
    cols_to_return = ['titleID',"title_main", 'Authors', 'Journal', "Conference", "Source", "book", \
                   'publisher', "vol", "issue", 'year', "pages", 'cited_num', \
               'cites_per_year', 'date', 'scrap_auth_id', "citedYear", "urlID", 'abstract']
        
        
    keysToGet = set(['titleID', 'cited', 'authors', 'authors', 'pubDate', 'journal', 'abstract'])
        
    keywords = ['perovskite', 'lead-halide', 'MAPI']
    #, 'hoip', 'organic-inorgancic', 'lead-halide', 'CH 3 NH 3 PbI 3', \
    #            'CsPbX', 'MAPI']
        
    locations = ['all']
    
    papers = load_dicts_from_dir_to_list_condition("paperDictA", dirs_to_read, colNames, keywords, locations)

    df = paperlist_to_df(papers)
    """ use apply to get the abstract text from all, create abstract column
    """
    # test paper gone at this point
    df['abstract'] = df.apply(get_abstract, axis = 1)
    df = clean_df(df, cols_to_return)
    
    ############ going to save each df seperately for now
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    fileName = file_name
    fileName = fileName + '_' + dt    
    
    df.to_csv(out_dir + fileName + '.csv')


def get_titles_by_author_id():
    """ need to make a list of each authorID for authors with a gs profile
        listed on a paper
    """
    out_dir = "data_subset\\"
    
    main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
    cur_dir = os.getcwd()
    dirs_to_read = getPaths(main_dir, "authorList_V3_B", "outputs")
    
    #papers = load_dicts_from_dir_to_list_condition("paperDictA", dirs_to_read, colNames, keywords, locations)
    
    accumulator = {}
    loop_folders_do_thing_V2('author_info', dirs_to_read, accumulator, \
                                 'none', ['all_papers']
                                     )
    
    path = out_dir + '/' + 'authors_for_paper.pickle'
    with open(path, 'wb') as f:
        pickle.dump(accumulator, f)
    return accumulator

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
    path += "\\data_subset\\" + "authors_for_paper.pickle"
    
    with open(path, 'rb') as f:
        authors_by_paper = pickle.load(f)
    
    df['author_ids'] = df.apply(attach_authorIds, axis = 1)
    
    return df

def get_as_f_of_author(thresh): #df_all, df_subset
    """ If an author meets some condition, get all their papers.
        author condition = f(authors papers). for this, if more than X of their
        papers have the perovskite keywords

        need the df of all papers, the df of perovskite papers, and the author dict
        with their papers as keys
        
        THIS IS THE FUNCTION USED TO SELECT A SUBSET FOR THE MOST RECENT SET OF DATA USED TO FIT
    """
    def file_ops_get_papers_for_author(temp_dict, accumulator, locations):
        def organize_paper_author(papers):
            try:
                if topKey == "Iz6tDlQAAAAJ": # ad hoc debugging stuff
                    print('found test case')
                if topKey in accumulator:
                    accumulator[topKey] += temp_dict[topKey][locations[0]]
                else:
                    accumulator[topKey] = temp_dict[topKey][locations[0]]
            except:
                print('debug')
        
        for topKey in temp_dict.keys():
            organize_paper_author(temp_dict[topKey][locations[0]])
        
    def get_authors_papers():
        """ loops through the output dirs and gets all authors
            and their papers from author info. If 
        """
        out_dir = "data_subset\\"
        main_dir = "C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\"
        cur_dir = os.getcwd()
        dirs_to_read = getPaths(main_dir, "authorList_V3", "outputs")
        locations = ['all_papers']
        accumulator = {}
        fileID = 'author_info'
        for path in dirs_to_read:
            for file in os.listdir(path):
                if file.endswith('.pckl') and fileID in file:
                    with open(path + "\\" + file,'rb') as f:
                        temp_file = open_file_ops(f, file)
                        if temp_file:
                            file_ops_get_papers_for_author(
                                temp_file, accumulator, locations)
        return accumulator
    
    def select_authors():
        """ returns a set with any author who had more 
            than "thresh" fraction of all their papers in perov_papers
            
            params: authors_papers
        """
        authors_selected = set()
        for author in authors_papers:
            score = 0
            authors_papers[author] = set(authors_papers[author])
            for paper in authors_papers[author]:
                if paper in perov_papers:
                    score += 1
            if score/len(authors_papers[author]) > thresh:
                authors_selected.add(author)
                
        return authors_selected
    

    def custom1(x):
        x = x.split(' ')
        for thing in x:
            if thing in authors_selected:
                return True
        return False #if the above condtion didn't occur return False
        
    def custom_test(x):
        x = x.split(' ')
        if len(x) > 1:
            return True
        else:
            return False
        
    authors_papers = get_authors_papers()
    df_subset = pd.read_csv("data_subset//perovskite_papers_from_dense_04-05-2021 16_04_08.csv")
    df_subset = add_author_ids(df_subset)
    df_all = pd.read_csv("data_subset//all_papers_with_authors_cleaned_5-4.csv")
    #df_all = add_author_ids(df_all)
    papers_w_authors = load_one_pickled("authors_for_paper.pickle", "data_subset")
    
    #authors = load_one_pickled("perovskite authors 4-26.pickle", "data_subset")
    perov_papers = set(df_subset['titleID'])
    authors_selected = select_authors()
    #need to get author info
    
    #authors_papers = get_authors_papers()
    df_select = df_all[df_all['author_ids'].apply(lambda x: custom1(x))]
    
    # df_select = df_all[df_all['author_ids'].apply(lambda x: len(x.split(' ')) > 1)]
    # df_test = df_all['author_ids'].apply(lambda x: custom_test(x))
    return df_select


def custom1(x):
    x = x.split(' ')
    for thing in x:
        if thing in authors_selected:
            return True
    return False #if the above condtion didn't occur return False
    
def custom_test(x):
    x = x.split(' ')
    if len(x) > 1:
        return True
    else:
        return False

def custom_make_set(x):
    x = ' '.join(set(x.split(' ')))
    return x

if __name__ == "__main__":

    # df_select = get_as_f_of_author(0.1)
    # df_select['author_ids'] = df_select['author_ids'].apply(lambda x: ' '.join(set(x.split(' '))))
    # df_select.to_csv('df_select_01_5-4')
    #authors_papers = get_as_f_of_author(0.3)
    #get_subset('authorList_V3', "perovskite_papers_from_dense") # gets papers with perovskite ids
    authors_for_paper = get_titles_by_author_id() # get the titles for each author, {author: [titles]}
    
    #authors = get_all_authors_keywords()
    #out_dir = "data_subset\\"
    
    #pickle.dump(authors, open( 'perovskite authors 4-30.pickle', "wb" ) )
    
    #authors = pickle.load(open("data_subset\\" + 'perovskite authors 4-26.pickle', "rb"))
    
    #update_db_create_table(authors, 'solarAuthors')
    
    #authors = get_all_authors_keywords()
    