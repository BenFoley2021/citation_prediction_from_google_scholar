# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:51:03 2021

initializing the db to be used by the scrapper programs
This script is meant to be run manually to create the data base

main_db_A needs authors and papers tables. The papers table is used by the paper_graph
scrapper to tell which papers it's already found. The authors table is used by
paper_graph to record newly found authors, and by the author_list crawler to
decide which author to go to next, and to record which authors have been done.

The firstauthors table is meant to contain the first author of each paper found by
paper_graph. This may allow author_list to run more effiecently by not bothering
with an author unless they are listed first on something. This is far from perfect,
as if the first author of the paper doesn't have a gs, then 2nd (or 3rd, etc) will look
like the first when that paper is scapped. This table needs to be populated by a 
custom function or through modification of paper_graph, and the db connection attributes of 
author_list updated.




@author: bcyk5
"""

import sqlite3
import os


def make_db():
    """ creates a database if one isn't already there. If it is, initializes 
        a connection to that database
    
    Returns
    -------
    con : sqlite3 database connection

    """
    

    db_name = 'main_db_A_example.db'
    path = os.getcwd()
    path = path + "\\" + db_name
    
    con = sqlite3.connect(path)

    return con

# create papers table
def make_tables(con):
    """ creates the tables which will be used by the paper_graph and author_list 
        crawlers
    

    Parameters
    ----------
    con : sqlite3 database connection
        connection to the database returned from make_db

    Returns
    -------
    None.

    """
    
    with con:
        con.execute("""
                    CREATE TABLE papers (
                        id TEXT NOT NULL PRIMARY KEY,
                        value TEXT
                        );
                    """)
                    
                    
    #write one testValue to papers
    sql = 'INSERT INTO papers (id, value) values (?, ?)'
    data = [('testPaper', 'testy_McTest')]
    
    with con:
        con.executemany(sql, data)
                    
    # create the authors table    
    with con:
        con.execute("""
                    CREATE TABLE authors (
                        id TEXT NOT NULL PRIMARY KEY,
                        value TEXT,
                        scrapped TEXT
                        );
                    """)
                    
    # write one test value to authors
    
    sql = 'INSERT INTO authors (id, value, scrapped) values (?, ?, ?)'
    data = [('testAuthor4', 'testy McTestyFace2', 'testFalse')]
    
    with con:
        con.executemany(sql, data)      
                    
    # create the first authors table, this isn't used in the defualt implementation
    # of data mining, but it can be a useful way to speed things up. Hence it's 
    # left in here
    with con:
        con.execute("""
                    CREATE TABLE firstauthors (
                        id TEXT NOT NULL PRIMARY KEY,
                        value TEXT,
                        scrapped TEXT
                        );
                    """)
                    
    # write one test value to first authors
    
    sql = 'INSERT INTO firstauthors (id, value, scrapped) values (?, ?, ?)'
    data = [('testFirstAuthor', 'testy McTestyFace2', 'testFalse')]
    
    with con:
        con.executemany(sql, data)      
    
def testUpdateDb():
    # want to test flipping the scrapped row in authors from False to True
    # https://www.sqlitetutorial.net/sqlite-update/
    """ misc code used for testing the database at various points in development.
        
    """    
    con = makeDb()
    
    sql = "UPDATE authors" + " SET" + " scrapped=" + '1' + " WHERE id=?"
    key = [("testAuthor2")]
    with con:
        data = con.execute(sql, key)


    sql = "SELECT * FROM " + "firstauthors" + " WHERE id=?"

#sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
    key = [("testFirstAuthor")]
    with con:
        data = con.execute(sql, key)
        for row in data:
            print(row)

# getting connection to bd (or making it if it doesnt exisit)

    
    
    
    sql = "SELECT EXISTS(SELECT 1 FROM " + "paperidcited" + " WHERE id=?)"
    
    #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
    key = [("paperid1")]
    with con:
        data = con.execute(sql, key)
        for row in data:
            print(row)
            
            
    sql = "SELECT * FROM " + "paperauthor" + " WHERE id=?"
    
    #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
    # d1gkVwhDpl0C # https://scholar.google.com/citations?user=IbDYT3AAAAAJ&hl=en&oi=sra
    # d1gkVwhDpl0C # https://scholar.google.com/citations?user=zARrOK0AAAAJ&hl=en
    key = [("d1gkVwhDpl0C")]
    with con:
        data = con.execute(sql, key)
        for row in data:
            print(row)
            
    sql = "SELECT EXISTS(SELECT 1 FROM " + "papers" + " WHERE id=?)"
    
    #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
    key = [("testPaper")]
    with con:
        data = con.execute(sql, key)
        for row in data:
            print(row)
            
    sql = "SELECT * FROM " + 'paperAuthor'
    
    with con:
        data = con.execute(sql)
        for row in data:
            print(row)
            
    sql = "SELECT * FROM " + 'authors' + " WHERE scrapped=?" + " LIMIT 10"
    key = [("0")]
    with con:
        data = con.execute(sql, key)
        for row in data:
            print(row)

con = make_db()

make_tables(con)


