# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:18:53 2021


@author: bcyk5
"""
import sqlite3
import traceback
import requests
from bs4 import BeautifulSoup
from scraper_api import ScraperAPIClient
client = ScraperAPIClient('XXXXXXXXXXXXXXXXXXXXXXXX')

import pickle
import os
import re
import time
import sys
from datetime import datetime
    
class pathHead():
    """ this is to make it easier for me to keep track of the url and where to go next
        contains all the info which needs to be passed around when searching the current paper
        and its connections. the object is returned to the mainCrawl function where the results 
        of crawling a paper are extracted, and then it passed back up to the crawlPaper function
    
        the object contains the url, the id of the citation page (I hope there's 1 unique
        Id for each papers cited by'). page is the current in pageination
        
        The methods are for modifying the url to go the next page or the next articles
        cited by
    """
    def __init__(self, citedID: str, pageNum: str, thresh: int, dbPath: str, pathID: str):
        """ constructing  object 
        """


        
        self.pathID = pathID # this ID will be used to keep track of which path found which papers and authors. the pathID arg will always be the same for this run of the crawl
        self.urlBase = "https://scholar.google.com/scholar?start="
        self.urlMid = "&hl=en&as_sdt=5,48&cites="
        self.urlEnd = "&scipsc="
        self.id = citedID
        self.page = pageNum
        self.paperEr = None ### for errors while scraping a paper
        self.pageEr = None ### for errors on a page
        self.page_yeild_error = 0
        self.apiEr = None ### error thrown by api. mostly this just tells use if we've run out of api calls
        self.genEr = None ### other errors to keep track of which will be dealt with in the mainLoop
        self.alreadySeen = 0 # counter for how many papers we've already seen
        self.yeild = 0 # number of papers sucessfully gotten from a page
        self.page_yeild = 0
        self.papersToStack = {} # there papers a added to the stack of nodes to visit
        self.papersNotStack = {} # these aren't going to be added to the stack
        self.leafThresh = thresh # if a paper has less citations than this, it's close enough to
                                # to being a leaf that we won't put it in the stack
        
        # the visted attribute is a connection to the database where all pathHeads report 
        # where they've been. this allows this to run in parrellel with other instances of 
        # the same program. I should probably have the value in visited by the pathHead id
        # then I can have different conditions for what's added to the stack depending on 
        # whether this pathHead visited it or another one did
        self.visited = sqlite3.connect(dbPath)
        
        
        self.url = str(self.urlBase) + str(pageNum) + str(self.urlMid) + \
            str(citedID) + str(self.urlEnd)
        
        
    def nextPage(self):
        """ go to the next page in pagination
        """

        self.page = str(int(self.page) + 10)
        
        self.url = str(self.urlBase) + str(self.page) + str(self.urlMid) + \
            str(self.id) + str(self.urlEnd)

        return self

def toSoup(r: object) -> object:
    """ converting the request object to a beatuiful soup. mmmmmm..... soup
    
    ok it turns out this didn't need to be a function'
    """
    soup = BeautifulSoup(r.text, 'html.parser')

    return soup

def findInSoup(soup: object, attrs: dict) -> list:
    listOut = []
    for hit in soup.findAll(attrs=attrs):
        #print(hit.text)
        listOut.append(hit)
        
    return listOut
    
def isLastPage(soup, curPathHead):
    """ checking to see if we are on the last page of citations for that paper
    """
    navNode = findInSoup(soup, {'id': "gs_nml"})
    if len(navNode) == 0: # papers with less than 10 citations wont have a nav tag 
        return True
        
    pageTags = findInSoup(navNode[0], {'class': "gs_nma"})

    if int(pageTags[-1].text)*10 - 10 <= int(str(curPathHead.page)): # if last tag bigger than curPathHead.page, at the end. 
        return True
    else:
        return False
    
def isFloat(numIn): # simple function to check if str can be converted to float
    try:
        float(numIn)
        return numIn
    except:
        #print('bad cited str')
        return None
        
def isInDb(id_1: str, table: str, curPathHead: object) -> bool:
    
    """ checks to see if the id_1 is present in the index of the "table" TABLE
        in the database which is connected to curPathHead
        
    """
    try: 
        sql = "SELECT EXISTS(SELECT 1 FROM " + table + " WHERE id=?)"
        
        #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
        key = [(id_1)]
        with curPathHead.visited as con:
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

def updateDbFlexible(table: str, attr: str, curPathHead) -> None:
    """ puts attr of the dictionaries into table
    

    Parameters
    ----------
    table : str
        DESCRIPTION.
    attr : str
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    allPapers = curPathHead.papersToStack
    allPapers.update(curPathHead.papersNotStack)
    keys = allPapers.keys()
    data = []
    for key in keys:
        if isInDb(allPapers[key][attr], table, curPathHead) == False:
            data.append((allPapers[key][attr], curPathHead.pathID, "false"))
    
    sql = 'INSERT INTO '+ table + ' (id, value, scrapped) values (?, ?, ?)'
    
    # the code to try and insert is the same regardless of whether its papers or authors, args to con. execute prepared in above code
    try: # program can't stop for data base errors. 
        with curPathHead.visited as con:
            con.executemany(sql, data)
    
    except: # if something goes wrong trying to insert everything at once, try one at a time, skip any that don't work
        print('error writing paperID to db')
        try:
            for dataRow in data:
                with curPathHead.visited as con:
                    con.execute(sql, data)
        except:
            pass
    
    return None
    

def updateDb(which: str, curPathHead):
    """ updates the sqlite database with the new papers or authors found. 
        which tells if the papers or authors are being updated
        
    Parameters
    ----------
    which : str
        DESCRIPTION.
    curPathHead : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    

    def prepareAuthorSet():
        """ making a set of unique author ids from all the authors found in the latest round of papers,
            and checking if they haven't been found before'. the author table of the db is read by the other 
            scrapper, and we only want to put new author values in there

        Returns
        -------
        authorSet: set, unique list of all the new authors found by crawing this curID 

        """
        authorSet = set()
        for key in keys: # looping through all keys
            for authorID in allPapers[key]['authors'].keys(): # getting the author IDs (the gs string for an author) and looping through
                if isInDb(authorID,'authors',curPathHead) == False: # if this author hasn't been 
                    authorSet.add(authorID)
                
        return authorSet
    
    # combining the two dicts for convience
    allPapers = curPathHead.papersToStack
    allPapers.update(curPathHead.papersNotStack)

    # need to get the keys to both dictionaries regardless of which
    keys = allPapers.keys()
    
    if which == 'papers':
        data = []
        for key in keys:
            if isInDb(key, 'papers', curPathHead) == False: #this should be a redundant check, but something could have been written to the database since the other check in parsePapers
                data.append((key,curPathHead.pathID))
            
        sql = 'INSERT INTO papers (id, value) values (?, ?)'
        
    elif which == 'authors':
        #updating the authors table which is used as inputs to the scraper reading the detailed 
        #paper data from the authors paper
        authorSet = prepareAuthorSet()
        data = []
        for author in authorSet: # i suppose this loop structure could also be shared with 'papers' for more polymorphic-ness
            data.append((author,curPathHead.pathID, 'False'))
            
        sql = 'INSERT INTO authors (id, value, scrapped) values (?, ?, ?)'
    
    # the code to try and insert is the same regardless of whether its papers or authors, args to con. execute prepared in above code
    try: # program can't stop for data base errors. 
        with curPathHead.visited as con:
            con.executemany(sql, data)
    
    except: # if something goes wrong trying to insert everything at once, try one at a time, skip any that don't work
        print('error writing ' + which + ' to db')
        try:
            for dataRow in data:
                with curPathHead.visited as con:
                    con.execute(sql, data)
        except:
            pass
    
    return None # the database has been updated, don't need to return anything

def getPage(curPathHead: object):
    """ having trouble with the client for the proxy, moving the process of 
        getting the page and making sure it's loaded to a seperate function.
        Tries the client, if we get a 200 response but no mainNode it tries requests once
        failing at that returns none for the soup

    """
    def clientOrRequests(curPathHead: object, which: bool):
        if which == True:
            r = client.get(url = curPathHead.url, render = True)
        else:
            r = requests.get(url = curPathHead.url)
            
        if str(r) == "<Response [200]>":
        
            soup = BeautifulSoup(r.text, 'html.parser')
        #### getting node with the papers as children
            mainNode = findInSoup(soup, {'id': "gs_res_ccl_mid"})
            
            if len(mainNode) != 1: ### just in case the page format changes, should only be one thing in the list
                print('main node not len 1')
                print(curPathHead.url)
                time.sleep(1)
                return None, str(r)
            else:
                return soup, str(r)
        else:
            return None, response
    
    which = True
    soup, response = clientOrRequests(curPathHead, which)
    
    if not soup:
        which = False
        soup, response = clientOrRequests(curPathHead, which)
        
    return soup, response
        
def saveCurrent(masterPaperDict, fileName):
    """ converting the current paperList to a dict and saving it
    """
    def savePickle(fileName, var2Save, outDir):
        fileName = fileName + '.pckl'
        path = outDir + '/' + fileName
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(var2Save, f)
        f.close()
        

    outDir = 'outputs'
    
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    
    fileName = fileName + '_' + dt    
    
    savePickle(fileName, masterPaperDict, outDir)

def parsePapers(papers: object, curPathHead: object) -> dict:
    """ convert the soup object for a list of papers into a list of "paperCl"
        objects which have the raw html and all desired items as attributes
        
        Also:
            checks to make sure a paper (node) hasn't already been visited'
            decides whether to place a paper into the stack or not based on
            how many citations it has
        
    Parameters
    ----------
    paper : bs4.element.Tag
        contains info on papers from the citations view

    Returns
    -------
    curPathHead: the main object, attirubutes updated to contain new papers

    """

    
    def getAuthorsAndJ(paper):
        authorDict = {}
        
        authorNode = findInSoup(paper, {'class': 'gs_a'})[0]
    
        authors = authorNode.findChildren()
        
        for author in authors:
            
            uInd = author['href'].find("?user=") + len("?user=")
            endInd = author['href'].find("&hl=")
            
            authorDict.update({author['href'][uInd:endInd]: author.text})
            
        startInd = authorNode.text.find("\xa0") + len("\xa0")
        #tempPaper.j = authorNode.text[startInd:]
        
        return authorDict, authorNode.text[startInd:]
        
    def getCited(textIn): # this is similar enough to getURLcited that they should be combined into one polymphic func
        """ The citations are a pain to grab using bs4, using text parsing instead
            if the cited by text isn't a number, return none'
        """ 
        
        startInd = textIn.find('Cited by ') + len('Cited by ')
        textIn = textIn[startInd:]
        endInd = textIn.find(' ')
        
        return isFloat(textIn[0:endInd])
        
    def getCitedUrlID(textIn):
        """ getting the urlID by text parsing. the url ID is needed to continue the crawl, so
            if it's not in the right format we return none'

        """
        startInd = textIn.find('href="/scholar?cites=') + len('href="/scholar?cites=')
        textIn = textIn[startInd:]
        
        endInd = textIn.find('&amp')
        
        return isFloat(textIn[0:endInd])

######### starting loop to get data from each paper that was passed to this func
    for paper in papers:
        
        tempDict = {}

        # try and get the info on the paper from the tag for that paper
        try: 
        # check to see if we already have this paper
            #if getCitedUrlID(str(paper)) not in curPathHead.visited and paper['data-cid'] not in curPathHead.visited:
            if isInDb(getCitedUrlID(str(paper)), 'papers', curPathHead) == False and \
                      isInDb(paper['data-cid'], 'papers', curPathHead) == False:
                          
                if getCitedUrlID(str(paper)) and getCited(paper.text): # if the citedURL and citations are there
                    tempDict['urlID'] = getCitedUrlID(str(paper))
                    tempDict['cited'] = getCited(paper.text)
                else:    # if not it's probably a leaf, set urlId as id, citations = 0. this field is set as the paper ID instead of the urlID for the citations. Paper id has letters in it
                    tempDict['urlID'] = paper['data-cid']
                    tempDict['cited'] = "0"
                
                tempDict['id'] = paper['data-cid']
                tempDict['raw'] = str(paper)
                tempDict['title'] = findInSoup(paper, {'class': 'gs_rt'})[0].text
                tempDict['abstractWords'] = findInSoup(paper, {'class': 'gs_rs'})[0].text
                
                authorDict, j = getAuthorsAndJ(paper)
                tempDict['authors'] = authorDict
                tempDict['j'] = j
            
                if int(tempDict['cited']) > curPathHead.leafThresh and isFloat(tempDict['urlID']): # if it has more than X cites and the 'urlID' can be a float its a useful lead, put is in the stack
                    curPathHead.papersToStack.update({str(tempDict['urlID']): tempDict})
                else:
                    curPathHead.papersNotStack.update({str(tempDict['urlID']): tempDict})
    
                curPathHead.yeild += 1 # keeping track of how many papers we actually got
                curPathHead.page_yeild += 1
            else: 
                curPathHead.alreadySeen += 1 # keeping track of how many papers we've already seen
        
        except: # if something goes wrong, we note it and move onto the next paper
            curPathHead.paperEr += 1
        
    return curPathHead
    
def crawlPaper(curPathHead: object) -> object:
    """ goes through all the citations of a paper
    
        goes through pagination and gets all the papers citing this paper
        
        starts from the cited by page for paper
        
        resonsible for dealing with pagination. this is the only function which
        gets the webpage
        
        error handling logic. if fail to get a page twice, move to next page. if also fail
        get that page twice, be done.
            count errors. if = 2, move to next page. if = 4, return
            if = 2 and move to next page, and that works, reset to 0
        
    args:
        start: the start url for the paper

    return:
        list: list of paperCl objects
        should also probably modify the stack which tells which node to go to next
        in the bfs
    """
    ###### initializing variables. most of the counters are reset when we start on a new node from the stack
    # i think this is redunant, each time we start a new paper to crawl with this function we get a new curPathHead object
    morePages = True
    curPathHead.paperEr = 0 
    curPathHead.alreadySeen = 0
    curPathHead.pageEr = 0
    curPathHead.yeild = 0 
    curPathHead.alreadySeen = 0
    curPathHead.papersToStack = {} # there papers a added to the stack of nodes to visit
    curPathHead.papersNotStack = {} # these aren't going to be added to the stack
    
    
    while morePages == True:
        # @@@@@@@@
        try:
            #r = requests.get(url = curPathHead.url)
            curPathHead.page_yeild = 0
            soup, response = getPage(curPathHead) 
            if response == '<Response [200]>':
                papers = findInSoup(soup,{'class': 'gs_r gs_or gs_scl'}) ### the nodes for each paer are under this parent
                # @@@@@@@@@
                if len(papers) == 0: # if no papers return and move onto the next in the stack
                    curPathHead.genEr = 'no papers'
                    return curPathHead
                
                curPathHead = parsePapers(papers, curPathHead) ###### getting the data on papers for the current page
                # @@@@@@@@@
                if curPathHead.page_yeild == 0: # if there weren't any papers with processable data on this page, we are done with the paper
                    curPathHead.page_yeild_error += 1
                    if curPathHead.page_yeild_error > 5:
                        curPathHead.genEr = 'no yeild yeild'
                        return curPathHead
                
                if isLastPage(soup, curPathHead) == True: ### checking to see if there's more papers on the next page
                    curPathHead.genEr = 'lastPage'    
                    return curPathHead # if that was the last page, done with this paper
                
                curPathHead =  pathHead.nextPage(curPathHead) ### moving to the url for the next page
                # @@@@@@@
                curPathHead.page_yeild_error = 0 
                curPathHead.paperEr = 0 # if we had an error on the previous page but not one this one, reset
                
            # @@@@@@@@@@@
            elif response == '<Response [403]>':
                print('out of API requests!!!!!!!!!!!')
                curPathHead.apiEr = '403'
                return curPathHead
            elif response == '<Response [429]>':
                print('requesting too fast')
                time.sleep(1)
            elif response == '<Response [404]>': ### problem with url, move onto next in stack
                curPathHead.apiEr = '404'
                return curPathHead
            elif response == '<Response [410]>': ### problem with url, move onto next in stack
                curPathHead.apiEr = '410'
                return curPathHead
            elif response == '<Response [500]>': ###### this has the potential to loop indefinetly
                print('500 error')
            else:
                morePages = False  # the defualt for if something else is wrong is skip to the next in stack
        #r = client.get(url = curPathHead.url)
        except:
            print(traceback.format_exc())
            curPathHead.paperEr += 1
            # if keep getting errors on a page, want to skip it. There are often papers with non-english
            # titles or incimplete info. If hit these, are likely near the end of the gs page. not really 
            # interested in citing articles which don't have meta data or aren't in english
            # if we get generic errors more than a few times in a row, move to next page
            # if that page has generic errors too, want to be done
            
            # first check and see if we are at stop condition, if two pages in a row are being weird we are done with the paper
            if curPathHead.paperEr > 3:
                curPathHead.genEr = 'paper error'
                return curPathHead
            
            elif curPathHead.paperEr >1: # if this page is being weird just skip it
                curPathHead =  pathHead.nextPage(curPathHead) 
                    
            time.sleep(1)
            print('exception')
            error = str(("Unexpected error: " +  str(sys.exc_info()[0])))
            print(error)
            
    return curPathHead # unlikely we ever end the function on this condition
    
def mainCrawl(startID: str, dbPath: str):
    """ manages the web crawl, decides which paper to look at next, eventually will decide
        when to terminate a path. this will be done by not adding a node to the stack
        
        this is only responible for dealing with papers
        
        also needs to dump the masterPaperList to disk when it gets too big
        
    """
    
    def updateDictsStack(masterPaperDict: dict, citedByIdDict: dict, stack: list, curPathHead: object) -> dict:
        """ moving the code which updates the dictionaries out of the main while loop for readaility
    
            # setting some logic for whether to put the latest batch of papers in the stack
            # want to avoid going to papers which connect to those we've seen already. If we've 
            # seen most of a papers connections before, the connections of those connections 
            # are more likely to be seen as well
    
        Parameters
        ----------
        masterPaperDict : dict
            Records of all papers and associated metadata found.
        stack: list, This is actually a queue
            list of nodes to go to next in the breadth first search
        citedByIdDict : dict
            The paper ids that each paper visited has been cited by. Can be used to reconstruct graph.
        curPathHead : custom object

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        if curPathHead.yeild != 0: # only bother with this step if we got something from crawlPaper
            # also want to avoid going down paths with lots of errors.
                if curPathHead.alreadySeen / (curPathHead.yeild + curPathHead.alreadySeen) < 0.9:
                    # for all papers found, if 90% are ones we've seen before stop this path
                    for key in curPathHead.papersToStack:
                        stack.append(key) # add the papers which cite the current paper to the bottom of the stack
                
                masterPaperDict.update(curPathHead.papersToStack) # update the main dictionary 
                masterPaperDict.update(curPathHead.papersNotStack) # update the main dictionary 
                
                citedByIdDict.update({curID : list(curPathHead.papersToStack.keys()) + \
                                      list(curPathHead.papersNotStack.keys())}) # the IDs of all papers which cited the current paper are added to the current papers data


        return masterPaperDict, citedByIdDict, stack
    
    
    # making a unique ID for this crawl, all papers  and authors found by this instance (run, whatever) will have this id 
    # attached to them when recorded in the db
    now = datetime.now()
    currentTime = now.strftime("%d/%m/%Y %H:%M")
    currentTime = currentTime.replace("/","-")
    currentTime = currentTime.replace(":","-")
    pathID = startID + ' ' + str(currentTime)
    
    try: # try to do the things, if something does wrong dump the variables outside the function so can look at them
        #id for neutron paper is 7459309748507948956
        itemsToGetBeforeSave = 2000 # when master paper dictionary is greater than this we dump it to disk
        leafThresh = 2 # if a paper only has two citations, don't put those citations in the stack
        # this is to try and explore the graph more efficently. following up on all papers with one or
        # two citations is slow, inefficent use of api calls, and biases the results toward recently
        # published papers with little citation history (not very useful for this project)
        
        masterPaperDict = {} ### holds all info for the paper. 
        citedByIdDict = {} # keeps track of which papers have cited the a paper for which citing articles crawled
        stack = [startID] ### stack which stores where to go next. 
        search = True
        count = 0
        
        while search == True: # this isn't just "while stack" becuase I'm likely going to add other stop conditions later
            
            curID = stack.pop(0)
            curPathHead = pathHead(curID, '0', leafThresh, dbPath, pathID) ### get a url object which contains the url for the current paper
            
            curPathHead = crawlPaper(curPathHead) # get all papers which cite the current paper
            
            # updating the ditionaries storing the results and deciding which papers go into the stack
            masterPaperDict, citedByIdDict, stack = updateDictsStack(masterPaperDict, citedByIdDict, stack, curPathHead)
            
            updateDbFlexible('paperidcited', 'id', curPathHead)
            updateDb('papers', curPathHead) ### updating dicts with seperate calls for now
            updateDb('authors', curPathHead)
            
            ###################
            
            if len(stack) == 0: # if there's nothing left in the stack, don't continue
                search = False
                
            elif curPathHead.apiEr == "403": # if We've run out of API calls stop
                search = False
            
            count += 1 ### print some updates to the command window
            print(count)
            print(curID)
            
            # if count > 1:
            #     print('actual ' + str(len(curPathHead.papersToStack) + len(curPathHead.papersNotStack)) \
            #           + ' expected ' + str(masterPaperDict[curID]['cited']))
    
            # dump the main dictionary and cited by ID list to disk when len() = X
            if len(masterPaperDict) > itemsToGetBeforeSave:
            
                saveCurrent(masterPaperDict, 'paperDict')
                del masterPaperDict
                masterPaperDict = {}
    
                saveCurrent(citedByIdDict, 'citedByIdDict')
                del citedByIdDict
                citedByIdDict = {}
    
        return citedByIdDict, masterPaperDict, curPathHead
    
    except:
        print('exception in main loop')
        print(traceback.format_exc())
        return masterPaperDict, citedByIdDict, curPathHead # this is just setup for adhoc debugging now

if __name__ == "__main__":

    startID = '9261126064257628754' # put the starting ID here
    dbPath = 'C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\dataBase\\main_db_A.db'
    #startID = "10431327697626759508" # this is for my nano letters paper, 20 citations

    masterPaperDict, citedByIdDict, curPathHead = mainCrawl(startID, dbPath)
    
    #saveCurrent(visited, 'visited')
    saveCurrent(citedByIdDict, 'citedByIdDict')
    saveCurrent(masterPaperDict, 'paperDict')
    
    def testDb(curPathHead):
        
        with curPathHead.visited as con:
            data = con.execute("SELECT * FROM authors")
            for row in data:
                print(row)
    
    
    
    