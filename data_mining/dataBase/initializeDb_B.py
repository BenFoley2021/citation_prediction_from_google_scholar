# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:51:03 2021

db_B is used by the author_list scrapper to record which papers it's already seen.
Clicking on a paper waiting for it load is the bottle neck in this scrapper, and
so it's very inefficent to look at the same paper from through the gs page of
multiple authors. The paper title is used as the primiary key in the table. 

@author: bcyk5
"""

import sqlite3
import os

def make_db():
    """ creates a database if one isn't already there. If it is, initializes 
        a connection to that database. Mirror of the function in initializeDb.py
    
    Returns
    -------
    con : sqlite3 database connection

    """
    
    db_name = 'db_B_example.db'
    path = os.getcwd()
    path = path + "\\" + db_name
    
    con = sqlite3.connect(path)

    return con

# create papers table
def make_tables(con):
    """ creates the tables which will be used by the paper_graph and author_list 
    crawlers. mirror of the function in initializeDb.py
    
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
                    CREATE TABLE authorpapers (
                        id TEXT NOT NULL PRIMARY KEY,
                        value TEXT
                        );
                    """)
                    
                    
    #write one testValue to papers
    sql = 'INSERT INTO authorpapers (id, value) values (?, ?)'
    data = [('test A Paper', 'testy_McTest')]
    
    with con:
        con.executemany(sql, data)
                    
  

# getting connection to bd (or making it if it doesnt exisit)
con = make_db()

make_tables(con)


""" misc code run to test the data base is left in below
"""
# sql = "SELECT EXISTS(SELECT 1 FROM " + "authorpapers" + " WHERE id=?)"

#sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
# key = [("test A Paper")]
# with con:
#     data = con.execute(sql, key)
#     for row in data:
#         print(row)
        
        
# sql = "SELECT * FROM " + "authorpapers" + " WHERE id=?"

# #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
# key = [("test A Paper")]
# with con:
#     data = con.execute(sql, key)
#     for row in data:
#         print(row)
        
